"""ASR roundtrip — OpenAI gpt-4o-transcribe + Levenshtein + IPA distance.

Per docs/plan.md § Tier 3 — Audio ASR roundtrip:

- Default model: `gpt-4o-transcribe` (LOCKED 2026-05 after A/B vs whisper-1,
  gpt-4o-mini-transcribe, and elevenlabs/scribe_v2 on 200 cached clips —
  see audit/ab_asr_summary.tsv). 99.0% pass-rate vs 95.5% for whisper-1
  at the same per-minute price.
- Pre-process clip (word clips < 1 sec hallucinate badly): pad 0.5s
  silence at start AND end before sending.
- Use `prompt = "Palavra em português brasileiro: {input}"` for word clips
  as a soft bias against English / YouTube garbage prior. No prompt on
  example clips (long enough that bias is unnecessary).
- temperature = 0.0 (suppress drift).
- Cross-check: if the biased pass returns input verbatim, run a second
  unbiased pass and verify — guards against prompt-trick false positives.

Length-aware decision policy:
- Inputs ≤ 3 chars normalized (function words like `o`, `de`, `em`):
  judge purely on phonetic distance (eSpeak IPA Levenshtein), threshold
  SHORT_INPUT_PHONETIC_PASS_DISTANCE = 0.40.
- Longer inputs: text Levenshtein with threshold 0.95 (top-1000) or 0.92
  (long-tail). Lower → REGEN (retry queued). After 2 retries → HUMAN.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI
from rapidfuzz.distance import Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.ipa import transcribe_word as ipa_transcribe  # noqa: E402

# Default ASR model — locked to gpt-4o-transcribe per A/B on 200 cached clips
# (see audit/ab_asr_summary.tsv and audit/ab_asr_disagreements.log):
#   - gpt-4o-transcribe       99.0% pass-rate (winner)
#   - whisper-1               95.5%
#   - gpt-4o-mini-transcribe  95.0% (occasional cross-language hallucinations
#                                    on isolated phonemes — Korean, Japanese)
#   - elevenlabs/scribe_v2    90.0%
# Same per-minute price as whisper-1 ($0.006/min); ~7× fewer false REGENs on
# short Portuguese function words.
DEFAULT_ASR_MODEL = "gpt-4o-transcribe"
# Backwards-compat alias for code that still imports the old name.
DEFAULT_WHISPER_MODEL = DEFAULT_ASR_MODEL
DEFAULT_LANGUAGE = "pt"
DEFAULT_TEMPERATURE = 0.0

# Decision thresholds (text-Levenshtein, normalized).
THRESHOLD_TOP1000 = 0.95
THRESHOLD_LONGTAIL = 0.92

# Phonetic distance is a secondary signal; only matters if text similarity
# passes by a hair. We don't fail on it alone (audio TTS that mispronounces
# something the ASR also mishears consistently is rare and caught at study).
PHONETIC_WARNING_DISTANCE = 0.30  # 1 - normalized_similarity

# Short-input policy: function words like `o`, `de`, `em` are too short for
# Whisper to transcribe reliably even with silence-padding + biased prompt
# (Whisper hallucinates "OU", "G", "PING" etc.). For inputs of ≤ this many
# normalized characters, judge purely on phonetic distance — a far more
# robust signal for isolated phonemes.
SHORT_INPUT_MAX_CHARS = 3
# Max acceptable phonetic distance for short-input PASS. Looser than text
# threshold because eSpeak-IPA representation has its own noise.
SHORT_INPUT_PHONETIC_PASS_DISTANCE = 0.40

# ASR API pricing (USD / minute audio) — used to log per-clip cost.
# As of 2026-05 OpenAI lists $0.006/min for whisper-1 and gpt-4o-transcribe,
# $0.003/min for gpt-4o-mini-transcribe.
ASR_PRICE_PER_MINUTE: dict[str, float] = {
    "whisper-1": 0.006,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
    "gpt-4o-transcribe-diarize": 0.006,
}
# Backwards-compat scalar for legacy callers; equals whisper-1's rate.
WHISPER_PRICE_PER_MINUTE = ASR_PRICE_PER_MINUTE["whisper-1"]


def _response_format_for(model: str) -> str:
    """gpt-4o-transcribe / gpt-4o-mini-transcribe accept ONLY `json`;
    whisper-1 accepts `text`/`json`/`verbose_json`/`srt`/`vtt`.

    We use `text` for whisper-1 (plain string return — easier) and `json`
    for the gpt-4o family (returns Transcription object with `.text` attr).
    """
    if model == "whisper-1":
        return "text"
    return "json"


def _extract_text(model: str, resp) -> str:
    """Pull the transcribed text out of whatever shape the SDK returned."""
    if model == "whisper-1":
        return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict):
        return resp.get("text", "")
    return str(resp)

# Padding applied before sending to Whisper.
SILENCE_PAD_START_S = 0.5
SILENCE_PAD_END_S = 0.5

# Retries for Whisper API.
DEFAULT_MAX_ATTEMPTS = 5
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class AsrResult:
    """Output of one ASR roundtrip on one clip."""

    transcript: str
    biased_transcript: str | None  # only set if a separate biased pass was made
    text_similarity: float         # normalized Levenshtein on cleaned text
    phonetic_distance: float | None  # 1 - normalized_similarity on IPA, or None
    decision: str                  # 'pass' | 'regen' | 'human'
    threshold_used: float
    cost_usd: float
    attempts: int                  # API calls made (1 or 2)
    notes: str = ""


# --- Text normalization --------------------------------------------------- #


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Normalize for Levenshtein similarity: lowercase, strip diacritics,
    collapse whitespace, drop punctuation.

    NOT used for IPA — that comparison happens on raw eSpeak output.
    """
    s = s.strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def text_similarity(a: str, b: str) -> float:
    """Levenshtein normalized similarity ∈ [0, 1] on normalized text."""
    return float(Levenshtein.normalized_similarity(normalize_text(a), normalize_text(b)))


def phonetic_distance(input_text: str, asr_transcript: str) -> float | None:
    """Compare eSpeak IPA of input vs ASR transcript; return 1 - similarity ∈ [0, 1].

    Returns None if either eSpeak run fails or produces empty IPA.
    """
    try:
        ipa_in = ipa_transcribe(input_text)
        ipa_out = ipa_transcribe(asr_transcript) if asr_transcript else ""
    except RuntimeError:
        return None
    if not ipa_in or not ipa_out:
        return None
    sim = float(Levenshtein.normalized_similarity(ipa_in, ipa_out))
    return round(1.0 - sim, 4)


# --- ffmpeg silence padding (uses the same ffmpeg discovered by loudness) -- #


def _ffmpeg_path() -> str:
    # local import so loudness.py is the single source of ffmpeg discovery
    from build.lib.loudness import FFMPEG  # noqa: E402

    return FFMPEG


def pad_mp3_with_silence(
    mp3_bytes: bytes,
    *,
    pad_start_s: float = SILENCE_PAD_START_S,
    pad_end_s: float = SILENCE_PAD_END_S,
) -> bytes:
    """Add silence padding to MP3 input; re-encode to MP3 192k.

    Used for word-only clips (< 1 sec) before sending to Whisper, to get the
    clip above the hallucination threshold.
    """
    delay_ms = int(pad_start_s * 1000)
    af = f"adelay={delay_ms}|{delay_ms},apad=pad_dur={pad_end_s}"
    args = [
        _ffmpeg_path(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-af", af,
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "1",
        "-f", "mp3",
        "pipe:1",
    ]
    proc = subprocess.run(args, input=mp3_bytes, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"silence padding failed: {proc.stderr.decode('utf-8', 'replace')[:300]}"
        )
    return proc.stdout


def _audio_duration_seconds(mp3_bytes: bytes) -> float:
    """Use ffprobe to read the (padded) clip duration. Cheap; reads container header."""
    from build.lib.loudness import FFPROBE  # noqa: E402

    proc = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "-i",
            "pipe:0",
        ],
        input=mp3_bytes,
        capture_output=True,
        check=False,
    )
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


# --- ASR client ----------------------------------------------------------- #


class AsrClient:
    """OpenAI ASR client (default `gpt-4o-transcribe`) with retry,
    biased-prompt option, silence-padding, and per-model response format."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_ASR_MODEL,
        language: str = DEFAULT_LANGUAGE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.language = language
        self.max_attempts = max_attempts

    def transcribe(
        self,
        mp3_bytes: bytes,
        *,
        prompt: str | None = None,
        filename: str = "clip.mp3",
    ) -> tuple[str, float]:
        """Transcribe MP3 bytes; return (transcript, cost_usd_for_this_call).

        Picks `response_format` per model: `text` for whisper-1, `json` for
        the gpt-4o-transcribe family (the new models reject `text`).
        """
        kwargs: dict = {
            "model": self.model,
            "language": self.language,
            "temperature": DEFAULT_TEMPERATURE,
            "response_format": _response_format_for(self.model),
            "file": (filename, mp3_bytes, "audio/mpeg"),
        }
        if prompt:
            kwargs["prompt"] = prompt[:240]  # truncated around 224 tokens
        last_exc: Exception | None = None
        price_per_min = ASR_PRICE_PER_MINUTE.get(self.model, WHISPER_PRICE_PER_MINUTE)
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._client.audio.transcriptions.create(**kwargs)
                text = _extract_text(self.model, resp)
                duration = _audio_duration_seconds(mp3_bytes)
                cost = (duration / 60.0) * price_per_min
                return text.strip(), round(cost, 5)
            except APIStatusError as exc:
                if exc.status_code in RETRYABLE_STATUS and attempt < self.max_attempts:
                    time.sleep(min(60.0, 2.0 ** (attempt - 1)))
                    last_exc = exc
                    continue
                raise
            except (APITimeoutError, APIConnectionError, APIError) as exc:
                if attempt < self.max_attempts:
                    time.sleep(min(60.0, 2.0 ** (attempt - 1)))
                    last_exc = exc
                    continue
                raise
        assert last_exc is not None
        raise last_exc


def _decision_for(similarity: float, threshold: float) -> str:
    return "pass" if similarity >= threshold else "regen"


def _word_prompt(input_text: str) -> str:
    return f"Palavra em português brasileiro: {input_text}"


def is_short_input(input_text: str, *, max_chars: int = SHORT_INPUT_MAX_CHARS) -> bool:
    """Whether to use the phonetic-only ASR path.

    Function words like `o`, `de`, `em` are too short for Whisper to transcribe
    reliably even with silence-padding + biased prompt; phonetic comparison
    against eSpeak IPA is far more robust for these cases.
    """
    return len(normalize_text(input_text)) <= max_chars


def _decide_phonetic_only(
    *,
    transcript: str,
    biased_transcript: str | None,
    input_text: str,
    threshold_dist: float,
    cost_total: float,
    attempts: int,
    extra_note: str = "",
) -> AsrResult:
    """Build an AsrResult judging on phonetic distance only.

    Falls back to text similarity if eSpeak can't produce IPA for either side
    (graceful — phonetic check is preferred but text is the safety net).
    """
    pdist = phonetic_distance(input_text, transcript)
    sim = text_similarity(transcript, input_text)
    if pdist is None:
        # eSpeak failed — fall back to text-based decision with a soft bound,
        # because for ≤3 chars text similarity is unreliable, so we use
        # PHONETIC_WARNING_DISTANCE (0.30) as a soft cutoff rather than the
        # tight text threshold.
        decision = "pass" if sim >= 0.99 else "regen"
        return AsrResult(
            transcript=transcript,
            biased_transcript=biased_transcript,
            text_similarity=round(sim, 4),
            phonetic_distance=None,
            decision=decision,
            threshold_used=0.99,  # nominal
            cost_usd=round(cost_total, 5),
            attempts=attempts,
            notes=("short-input fallback: eSpeak unavailable; " + extra_note).strip("; "),
        )
    decision = "pass" if pdist <= threshold_dist else "regen"
    note = f"short-input phonetic check (dist={pdist:.2f}, thresh={threshold_dist:.2f})"
    if extra_note:
        note = f"{note}; {extra_note}"
    return AsrResult(
        transcript=transcript,
        biased_transcript=biased_transcript,
        text_similarity=round(sim, 4),
        phonetic_distance=pdist,
        decision=decision,
        threshold_used=threshold_dist,
        cost_usd=round(cost_total, 5),
        attempts=attempts,
        notes=note,
    )


def asr_roundtrip(
    *,
    asr: AsrClient,
    mp3_bytes: bytes,
    input_text: str,
    clip_type: str,
    sense_id: str,
    is_top_1000: bool = False,
) -> AsrResult:
    """Run ASR, compare to input_text, return a decision.

    For word clips, applies silence padding and a biased prompt. Two policy
    branches by input length:

    - **Short inputs (≤ SHORT_INPUT_MAX_CHARS chars after normalization)**:
      judge purely on phonetic distance (eSpeak-IPA Levenshtein), because
      Whisper hallucinates badly on isolated phonemes. Threshold:
      SHORT_INPUT_PHONETIC_PASS_DISTANCE.
    - **Longer inputs**: text Levenshtein with the standard threshold; if
      biased transcript matches input verbatim, cross-check unbiased.
    """
    threshold = THRESHOLD_TOP1000 if is_top_1000 else THRESHOLD_LONGTAIL
    cost_total = 0.0
    attempts = 0

    if clip_type == "word":
        # Pad short clips before transcription to mitigate Whisper hallucination.
        try:
            send_bytes = pad_mp3_with_silence(mp3_bytes)
        except RuntimeError:
            send_bytes = mp3_bytes  # graceful fallback
        prompt = _word_prompt(input_text)
        biased_text, c1 = asr.transcribe(send_bytes, prompt=prompt, filename=f"{sense_id}-word.mp3")
        cost_total += c1
        attempts += 1

        # Length-aware policy: short inputs go through phonetic-only path.
        if is_short_input(input_text):
            return _decide_phonetic_only(
                transcript=biased_text,
                biased_transcript=biased_text,
                input_text=input_text,
                threshold_dist=SHORT_INPUT_PHONETIC_PASS_DISTANCE,
                cost_total=cost_total,
                attempts=attempts,
            )

        biased_sim = text_similarity(biased_text, input_text)

        if biased_sim >= 0.999:
            # Bias might be fooling us; cross-check unbiased.
            unbiased_text, c2 = asr.transcribe(send_bytes, prompt=None, filename=f"{sense_id}-word-unbiased.mp3")
            cost_total += c2
            attempts += 1
            unbiased_sim = text_similarity(unbiased_text, input_text)
            # Use the WORSE of the two to be conservative — if unbiased fails,
            # the biased pass was lying for us.
            if unbiased_sim < threshold:
                pdist = phonetic_distance(input_text, unbiased_text)
                return AsrResult(
                    transcript=unbiased_text,
                    biased_transcript=biased_text,
                    text_similarity=round(unbiased_sim, 4),
                    phonetic_distance=pdist,
                    decision=_decision_for(unbiased_sim, threshold),
                    threshold_used=threshold,
                    cost_usd=round(cost_total, 5),
                    attempts=attempts,
                    notes="unbiased cross-check disagreed with biased",
                )
            # Both agree — accept.
            pdist = phonetic_distance(input_text, unbiased_text)
            return AsrResult(
                transcript=unbiased_text,
                biased_transcript=biased_text,
                text_similarity=round(unbiased_sim, 4),
                phonetic_distance=pdist,
                decision="pass",
                threshold_used=threshold,
                cost_usd=round(cost_total, 5),
                attempts=attempts,
                notes="biased+unbiased agreed",
            )
        else:
            pdist = phonetic_distance(input_text, biased_text)
            return AsrResult(
                transcript=biased_text,
                biased_transcript=biased_text,
                text_similarity=round(biased_sim, 4),
                phonetic_distance=pdist,
                decision=_decision_for(biased_sim, threshold),
                threshold_used=threshold,
                cost_usd=round(cost_total, 5),
                attempts=attempts,
                notes="",
            )

    # Example clips: long enough not to need padding or bias.
    text, c1 = asr.transcribe(mp3_bytes, prompt=None, filename=f"{sense_id}-ex.mp3")
    cost_total += c1
    attempts += 1
    sim = text_similarity(text, input_text)
    pdist = phonetic_distance(input_text, text)
    return AsrResult(
        transcript=text,
        biased_transcript=None,
        text_similarity=round(sim, 4),
        phonetic_distance=pdist,
        decision=_decision_for(sim, threshold),
        threshold_used=threshold,
        cost_usd=round(cost_total, 5),
        attempts=attempts,
        notes="",
    )
