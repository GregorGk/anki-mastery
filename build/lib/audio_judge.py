"""Audio-input dialect/accent judge — wrapper around OpenAI's
`gpt-4o-audio-preview` with structured Tool-Use output.

Listens to a Brazilian Portuguese vocabulary clip and judges whether the
speaker pronounces the word as standard Brazilian Portuguese, European
Portuguese, English, Spanish, French, or another accent. Primary
verdict is `bp_ok / non_bp / unclear`; the drift source guess is
secondary diagnostic information.

Usage:
    from build.lib.audio_judge import AudioJudgeClient

    client = AudioJudgeClient(audit_path="audit/08_1_audio_judge.jsonl")
    verdict = client.judge(
        audio_bytes=mp3_bytes,
        audio_format="mp3",
        pt="animal",
        ipa_word_final="ˌaniˈmaw",
        voice_id="4r3G9XKliGgVZLKMgjik",
        sense_id="0317.00.01",
    )
    # verdict: {"pronunciation_verdict": "bp_ok", "drift": "none", "severity": "low",
    #           "confidence": "high", "evidence": "final -l vocalized to [w]"}

Pricing (2026-05): `gpt-4o-audio-preview` at ~$100/1M input audio tokens
≈ $0.005 per ~3-second clip. Text input/output billed separately at
standard chat-completions rates. With prompt caching active for the
fixed system prompt (≥1024 tokens), the per-call cost is dominated by
audio.

Retries: full-jitter exponential backoff, 6 attempts, 60s cap, honors
`Retry-After` headers per project conventions.
"""
from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from build.lib import judge_prompts

DEFAULT_AUDIO_JUDGE_MODEL = "gpt-4o-audio-preview"

# Pricing 2026-05 (subject to model registry updates)
PRICE_PER_1M_AUDIO_TOKENS = 100.00
PRICE_PER_1M_TEXT_INPUT = 2.50
PRICE_PER_1M_OUTPUT = 10.00
# Cached input is ~50% of uncached
PRICE_PER_1M_CACHED_INPUT = 1.25

# A typical 2-3 second BP word clip is ~50-150 audio tokens.
TYPICAL_AUDIO_TOKENS = 100

JUDGE_TOOL_NAME = "judge_pronunciation"
JUDGE_TOOL_SCHEMA = {
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
            "description": "How strong the drift is acoustically. Low = subtle. High = obvious.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Your certainty in the verdict.",
        },
        "evidence": {
            "type": "string",
            "description": "One concrete acoustic cue that drove the verdict (e.g., 'final /l/ vocalized to [w] confirms BP', 'English vowel /æ/ on first syllable indicates EN drift').",
        },
    },
    "required": ["pronunciation_verdict", "drift", "severity", "confidence", "evidence"],
    "additionalProperties": False,
}

JUDGE_SYSTEM_PROMPT = """You are judging whether a Brazilian Portuguese vocabulary recording is pronounced as standard Brazilian Portuguese.

Listen to the attached audio and decide whether the speaker pronounces the given word as:
- standard Brazilian Portuguese (`bp_ok`),
- some other accent / language (`non_bp`),
- unclear / borderline (`unclear`).

Focus on phonetic features only. Do not transcribe. Do not evaluate audio fidelity, voice quality, or recording artifacts.

Brazilian Portuguese phonetic features:
- Final -l → [w] (vocalized; like Polish ł in łapa, English "wow"). NOT consonantal [l].
- Final -de / -te → [dʒi] / [tʃi] (palatalized; like Polish dżi/czi). NOT [d]/[t] silent (which is European Portuguese).
- Initial r / double rr → [h] or [χ] (back fricative; like English h or German Bach). NOT French uvular [ʁ].
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

Confidence captures your own uncertainty — use `low` freely if the audio is borderline; the calibration step will surface those for human review."""


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, APIError):
        return True
    return False


def _backoff(attempt: int) -> float:
    base = 1.0
    cap = 60.0
    return random.uniform(0, min(cap, base * (2 ** attempt)))


@dataclass
class JudgeResult:
    sense_id: str
    pronunciation_verdict: str  # bp_ok / non_bp / unclear
    drift: str  # EN / EP / ES / FR / other / none
    severity: str  # low / medium / high
    confidence: str  # low / medium / high
    evidence: str
    model: str
    latency_ms: int
    cost_usd: float
    audio_tokens: int
    cached_text_tokens: int
    uncached_text_tokens: int
    output_tokens: int


class AudioJudgeClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        audit_path: str | Path | None = None,
        prompt_version: str = "J1",
    ) -> None:
        self.model = model or DEFAULT_AUDIO_JUDGE_MODEL
        self.prompt_version = prompt_version
        self.system_prompt = judge_prompts.build_system_prompt(
            JUDGE_SYSTEM_PROMPT, prompt_version)
        self.prompt_hash = judge_prompts.prompt_hash(self.system_prompt)
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._calls = 0
        self._audio_tokens = 0
        self._cached_text_tokens = 0
        self._uncached_text_tokens = 0
        self._output_tokens = 0
        self._total_cost_usd = 0.0

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "calls": self._calls,
                "audio_tokens": self._audio_tokens,
                "cached_text_tokens": self._cached_text_tokens,
                "uncached_text_tokens": self._uncached_text_tokens,
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
    ) -> JudgeResult:
        """Single audio-judge call. Returns JudgeResult. Logs to audit JSONL.

        Raises after `max_attempts` retries.
        """
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        # J1 reproduces the production user message byte-for-byte.
        user_text = judge_prompts.user_text(
            pt=pt, ipa=ipa_word_final, clip_type=clip_type,
            version=self.prompt_version)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": audio_format},
                    },
                ],
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": JUDGE_TOOL_NAME,
                    "description": "Submit your verdict on the audio's pronunciation.",
                    "parameters": JUDGE_TOOL_SCHEMA,
                },
            }
        ]

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    modalities=["text"],
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": JUDGE_TOOL_NAME}},
                    max_completion_tokens=300,
                )
                latency_ms = int((time.time() - t0) * 1000)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _retryable(exc) or attempt == max_attempts - 1:
                    raise
                wait_s = _backoff(attempt)
                # Honor Retry-After if present
                if isinstance(exc, APIStatusError):
                    ra = getattr(exc.response, "headers", {}).get("Retry-After")
                    if ra:
                        try:
                            wait_s = max(wait_s, float(ra))
                        except (TypeError, ValueError):
                            pass
                time.sleep(wait_s)
        else:  # pragma: no cover
            assert last_exc is not None
            raise last_exc

        # Extract structured tool output
        choice = resp.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            raise RuntimeError(
                f"audio judge for {sense_id} returned no tool_calls. content={message.content!r}"
            )
        args_raw = tool_calls[0].function.arguments
        try:
            parsed = json.loads(args_raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"audio judge for {sense_id} returned malformed JSON: {args_raw!r}"
            ) from e

        # Usage accounting
        usage = resp.usage
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None) or {}
        audio_tokens = getattr(prompt_tokens_details, "audio_tokens", 0) or 0
        cached_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) or 0
        text_input_tokens = (usage.prompt_tokens or 0) - audio_tokens
        uncached_text_tokens = max(text_input_tokens - cached_tokens, 0)
        output_tokens = usage.completion_tokens or 0

        # Cost computation
        cost = (
            audio_tokens * PRICE_PER_1M_AUDIO_TOKENS / 1_000_000
            + uncached_text_tokens * PRICE_PER_1M_TEXT_INPUT / 1_000_000
            + cached_tokens * PRICE_PER_1M_CACHED_INPUT / 1_000_000
            + output_tokens * PRICE_PER_1M_OUTPUT / 1_000_000
        )

        result = JudgeResult(
            sense_id=sense_id,
            pronunciation_verdict=parsed["pronunciation_verdict"],
            drift=parsed["drift"],
            severity=parsed["severity"],
            confidence=parsed["confidence"],
            evidence=parsed["evidence"],
            model=self.model,
            latency_ms=latency_ms,
            cost_usd=cost,
            audio_tokens=audio_tokens,
            cached_text_tokens=cached_tokens,
            uncached_text_tokens=uncached_text_tokens,
            output_tokens=output_tokens,
        )

        with self._stats_lock:
            self._calls += 1
            self._audio_tokens += audio_tokens
            self._cached_text_tokens += cached_tokens
            self._uncached_text_tokens += uncached_text_tokens
            self._output_tokens += output_tokens
            self._total_cost_usd += cost

        # Audit log
        if self.audit_path:
            audit_record = {
                "event": "judged",
                "sense_id": sense_id,
                "clip_type": clip_type,
                "voice_id": voice_id,
                "pt": pt,
                "ipa_word_final": ipa_word_final,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash,
                "verdict": parsed["pronunciation_verdict"],
                "drift": parsed["drift"],
                "severity": parsed["severity"],
                "confidence": parsed["confidence"],
                "evidence": parsed["evidence"],
                "latency_ms": latency_ms,
                "cost_usd": round(cost, 6),
                "audio_tokens": audio_tokens,
                "cached_text_tokens": cached_tokens,
                "uncached_text_tokens": uncached_text_tokens,
                "output_tokens": output_tokens,
                "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            }
            line = json.dumps(audit_record, ensure_ascii=False)
            with self._audit_lock:
                with self.audit_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

        return result
