"""Gemini audio-input judge — wrapper around google-genai for BP-vs-EP
accent classification. Mirrors the shape of build/lib/audio_judge.py
(which targets OpenAI) so the two are drop-in interchangeable in tests.

Same tool schema:
    {
      pronunciation_verdict: "bp_ok" | "non_bp" | "unclear",
      drift:                 "EN" | "EP" | "ES" | "FR" | "other" | "none",
      severity:              "low" | "medium" | "high",
      confidence:            "low" | "medium" | "high",
      evidence:              str
    }

Pricing (2026-05) on `gemini-3.1-pro-preview`:
    text input <200K:   ~$2/1M
    audio input:        ~$2-3/1M est. (audio billed at ~text rate × 1-2)
    output:             ~$12/1M
~$0.0002 per ~3-s clip — ~25× cheaper than gpt-4o-audio-preview.

Structured output quirk (verified 2026-05-13): gemini-3.1-pro-preview
sometimes emits a free-text preamble ("Here is the JSON requested:") and
stops, even with `response_mime_type='application/json'` + `response_schema`.
We work around this by:
  1. Repeating the JSON-only constraint at the END of the system prompt
     and in the user message.
  2. Lenient parsing — extract the first `{...}` block via regex if the
     raw text isn't valid JSON.

Retries: full-jitter exponential backoff, 6 attempts, 60 s cap.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

DEFAULT_GEMINI_AUDIO_JUDGE_MODEL = "gemini-3.1-pro-preview"
# Cheap fast alternative for cost-sensitive runs.
GEMINI_FLASH_AUDIO_JUDGE_MODEL = "gemini-3-flash-preview"

# Rough 2026-05 pricing for the Pro-tier preview (USD per 1M tokens).
PRICE_PER_1M_INPUT_TEXT = 2.00
PRICE_PER_1M_OUTPUT = 12.00
# Audio is billed similarly to text on Gemini, sometimes at a small multiplier.
PRICE_PER_1M_INPUT_AUDIO = 3.00  # conservative est.

JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pronunciation_verdict": {
            "type": "string",
            "enum": ["bp_ok", "non_bp", "unclear"],
            "description": "Primary verdict on whether the audio sounds like standard Brazilian Portuguese.",
        },
        "drift": {
            "type": "string",
            "enum": ["EN", "EP", "ES", "FR", "other", "none"],
            "description": "If non_bp, the perceived accent / drift source. 'none' if bp_ok.",
        },
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "How strong the drift is acoustically.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Your certainty in the verdict.",
        },
        "evidence": {
            "type": "string",
            "description": "One concrete acoustic cue that drove the verdict.",
        },
    },
    "required": ["pronunciation_verdict", "drift", "severity", "confidence", "evidence"],
    "propertyOrdering": ["pronunciation_verdict", "drift", "severity",
                         "confidence", "evidence"],
}

JUDGE_SYSTEM_PROMPT = """You are judging whether a Brazilian Portuguese vocabulary recording is pronounced as standard Brazilian Portuguese.

Listen to the attached audio and decide whether the speaker pronounces the given word as:
- standard Brazilian Portuguese (`bp_ok`),
- some other accent / language (`non_bp`),
- unclear / borderline (`unclear`).

Focus on phonetic features only. Do not transcribe. Do not evaluate audio fidelity, voice quality, or recording artifacts.

Brazilian Portuguese phonetic features:
- Final -l → [w] (vocalized). NOT consonantal [l].
- Final -de / -te → [dʒi] / [tʃi] (palatalized). NOT [d]/[t] silent (which is European Portuguese).
- Initial r / double rr → [h] or [χ] (back fricative). NOT French uvular [ʁ].
- Coda s → [s] or [z]. NOT [ʃ] (which is European Portuguese).
- Final unstressed -e → [i] (clear "ee" vowel). NOT [ɨ] or silent (European Portuguese).
- Stress placement on cognates (animal, hospital, social, total) → final syllable. NOT initial (English).
- Silent initial h (hora, hospital, hotel) → no aspiration. NOT English aspirated [h].

Common drift directions:
- EN: English-leaning. Native-EN voices pronouncing PT cognates with English phonemes; final l consonantal; cognate stress on first syllable; aspirated initial h.
- EP: European Portuguese. Final -de/-te silent; coda s as [ʃ]; final unstressed -e silent; uvular r.
- ES: Spanish. Vowel quality differences; no nasalization on a, e, o; no [tʃ] palatalization on tio/tia.
- FR: French. Uvular r; nasal vowel in final position; stressed final syllable but with French tone.

Your primary decision is `bp_ok` vs `non_bp` vs `unclear`. The `drift` field guesses the source if non_bp; use `none` if bp_ok.

Severity captures how acoustically strong the deviation is — `low` is a subtle accent shift, `high` is obvious wrong-language pronunciation.

Confidence captures your own uncertainty — use `low` freely if the audio is borderline.

CRITICAL OUTPUT FORMAT: respond with ONE raw JSON object matching the
schema. No markdown code fences. No preamble like "Here is the JSON".
No explanation before or after. The very first character of your response
must be `{` and the very last character must be `}`."""


def _retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    # google-genai raises exceptions whose str includes status info.
    if "429" in msg or "rate limit" in msg or "quota" in msg:
        return True
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        return True
    if "timeout" in msg or "connection" in msg or "deadline" in msg:
        return True
    return False


def _backoff(attempt: int) -> float:
    base = 1.0
    cap = 60.0
    return random.uniform(0, min(cap, base * (2 ** attempt)))


@dataclass
class GeminiJudgeResult:
    sense_id: str
    pronunciation_verdict: str
    drift: str
    severity: str
    confidence: str
    evidence: str
    model: str
    latency_ms: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


class GeminiAudioJudgeClient:
    """Mirror of AudioJudgeClient that targets Gemini via google-genai SDK.

    Audio is base64 in/out via `types.Part.from_bytes(data=..., mime_type=...)`.
    Structured output via `response_mime_type='application/json'` + schema.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.model = model or DEFAULT_GEMINI_AUDIO_JUDGE_MODEL
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_cost_usd = 0.0

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "calls": self._calls,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_cost_usd": round(self._total_cost_usd, 4),
            }

    def judge(
        self,
        *,
        audio_bytes: bytes,
        audio_format: str = "mp3",
        pt: str,
        ipa_word_final: str,
        voice_id: str,
        sense_id: str,
        clip_type: str = "word",
        max_attempts: int = 6,
    ) -> GeminiJudgeResult:
        mime = f"audio/{audio_format}" if audio_format != "mp3" else "audio/mpeg"
        user_text = (
            f"Word: {pt}\n"
            f"Target IPA (Brazilian Portuguese): {ipa_word_final}\n"
            f"Clip type: {clip_type}\n\n"
            f"Respond with ONE raw JSON object only — no preamble, no markdown."
        )

        # The SDK uses ContentUnion. We pass: [audio Part, text Part].
        contents = [
            types.Part.from_bytes(data=audio_bytes, mime_type=mime),
            user_text,
        ]
        config = types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=JUDGE_RESPONSE_SCHEMA,
            temperature=0.0,
            # Gemini 3.1 Pro Preview is a thinking-mode-only model
            # (`thinking_budget=0` is rejected with HTTP 400). We give it
            # a modest 1024-token thinking budget for a phonetic-judgment
            # call, then a 5k output cap so the JSON response has plenty
            # of room after thinking-token consumption (max_output_tokens
            # includes BOTH thinking and emitted-response tokens).
            max_output_tokens=5000,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        )

        last_exc: Exception | None = None
        resp = None
        for attempt in range(max_attempts):
            try:
                t0 = time.time()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                latency_ms = int((time.time() - t0) * 1000)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _retryable(exc) or attempt == max_attempts - 1:
                    raise
                time.sleep(_backoff(attempt))
        if resp is None:
            assert last_exc is not None
            raise last_exc

        # Extract the JSON text. Gemini-3.1-pro-preview sometimes leaks a
        # preamble despite response_schema; recover by extracting the first
        # balanced `{…}` block.
        raw = (resp.text or "").strip()
        parsed: dict | None = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try fenced code blocks first.
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                # Find the first {...} block by counting braces.
                start = raw.find("{")
                if start >= 0:
                    depth = 0
                    for i, ch in enumerate(raw[start:], start=start):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                try:
                                    parsed = json.loads(raw[start:i+1])
                                except json.JSONDecodeError:
                                    parsed = None
                                break
        if parsed is None:
            raise RuntimeError(
                f"gemini audio judge for {sense_id} returned non-JSON: {raw[:200]!r}"
            )

        usage = resp.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        # We treat all input tokens at the audio rate as a conservative
        # over-estimate. Real Gemini billing splits audio vs text but the
        # SDK doesn't surface that cleanly here.
        cost = (
            input_tokens * PRICE_PER_1M_INPUT_AUDIO / 1_000_000
            + output_tokens * PRICE_PER_1M_OUTPUT / 1_000_000
        )

        result = GeminiJudgeResult(
            sense_id=sense_id,
            pronunciation_verdict=parsed["pronunciation_verdict"],
            drift=parsed["drift"],
            severity=parsed["severity"],
            confidence=parsed["confidence"],
            evidence=parsed["evidence"],
            model=self.model,
            latency_ms=latency_ms,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        with self._stats_lock:
            self._calls += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._total_cost_usd += cost

        if self.audit_path:
            audit_record = {
                "event": "judged",
                "judge": "gemini",
                "model": self.model,
                "sense_id": sense_id,
                "clip_type": clip_type,
                "voice_id": voice_id,
                "pt": pt,
                "ipa_word_final": ipa_word_final,
                "verdict": parsed["pronunciation_verdict"],
                "drift": parsed["drift"],
                "severity": parsed["severity"],
                "confidence": parsed["confidence"],
                "evidence": parsed["evidence"],
                "latency_ms": latency_ms,
                "cost_usd": round(cost, 6),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            }
            line = json.dumps(audit_record, ensure_ascii=False)
            with self._audit_lock:
                with self.audit_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

        return result
