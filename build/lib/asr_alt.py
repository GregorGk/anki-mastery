"""Alternative ASR backends for A/B comparison.

Wraps four ASR models behind one common interface:
  - openai/whisper-1               (Whisper-V2 base; current default)
  - openai/gpt-4o-mini-transcribe  (GPT-4o-based, cheaper than whisper-1)
  - openai/gpt-4o-transcribe       (GPT-4o-based, full quality)
  - elevenlabs/scribe_v2           (ElevenLabs' own ASR)

Used by build/ab_asr_models.py to compare model agreement on the same
cached MP3 inputs without re-running ElevenLabs TTS.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from elevenlabs.client import ElevenLabs
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

# Per-minute prices (USD) — best-known as of 2026-05; used for cost reporting.
PRICE_PER_MINUTE = {
    "whisper-1": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
    "gpt-4o-transcribe": 0.006,
    "scribe_v2": 0.0067,
}

DEFAULT_MAX_ATTEMPTS = 3
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class TranscribeResponse:
    model: str
    transcript: str
    cost_usd: float
    attempts: int


def _backoff(attempt: int) -> float:
    return min(60.0, 2.0 ** (attempt - 1))


# --- OpenAI -------------------------------------------------------------- #


def _openai_response_format(model: str) -> str:
    """gpt-4o-transcribe / gpt-4o-mini-transcribe only support JSON."""
    if model == "whisper-1":
        return "text"
    return "json"


def _openai_extract_text(model: str, resp) -> str:
    if model == "whisper-1":
        # response_format=text returns plain string
        return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
    # response_format=json returns Transcription object with .text
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict):
        return resp.get("text", "")
    return str(resp)


def transcribe_openai(
    *,
    client: OpenAI,
    model: str,
    mp3_bytes: bytes,
    language: str = "pt",
    prompt: str | None = None,
    filename: str = "clip.mp3",
    duration_seconds: float = 0.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> TranscribeResponse:
    """Single OpenAI transcription call with retry."""
    kwargs: dict = {
        "model": model,
        "language": language,
        "temperature": 0.0,
        "response_format": _openai_response_format(model),
        "file": (filename, mp3_bytes, "audio/mpeg"),
    }
    if prompt:
        kwargs["prompt"] = prompt[:240]

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.audio.transcriptions.create(**kwargs)
            text = _openai_extract_text(model, resp).strip()
            cost = (duration_seconds / 60.0) * PRICE_PER_MINUTE[model]
            return TranscribeResponse(model=model, transcript=text, cost_usd=round(cost, 5), attempts=attempt)
        except APIStatusError as exc:
            if exc.status_code in RETRYABLE_STATUS and attempt < max_attempts:
                time.sleep(_backoff(attempt))
                last_exc = exc
                continue
            raise
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            if attempt < max_attempts:
                time.sleep(_backoff(attempt))
                last_exc = exc
                continue
            raise
    assert last_exc is not None
    raise last_exc


# --- ElevenLabs Scribe v2 ------------------------------------------------- #


def transcribe_elevenlabs(
    *,
    client: ElevenLabs,
    mp3_bytes: bytes,
    language_code: str = "por",  # ISO 639-3 for Portuguese
    duration_seconds: float = 0.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> TranscribeResponse:
    """ElevenLabs Scribe v2 transcription call with retry.

    Note: scribe_v2 does NOT support a biased-prompt parameter analogous
    to OpenAI's `prompt`. It auto-detects context from the audio + language
    code only.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            from io import BytesIO

            resp = client.speech_to_text.convert(
                file=BytesIO(mp3_bytes),
                model_id="scribe_v2",
                language_code=language_code,
                tag_audio_events=False,
                diarize=False,
            )
            # response is an object with .text and .language_code
            text = (getattr(resp, "text", "") or "").strip()
            cost = (duration_seconds / 60.0) * PRICE_PER_MINUTE["scribe_v2"]
            return TranscribeResponse(
                model="scribe_v2",
                transcript=text,
                cost_usd=round(cost, 5),
                attempts=attempt,
            )
        except Exception as exc:  # ElevenLabs-specific errors not split here
            if attempt < max_attempts:
                time.sleep(_backoff(attempt))
                last_exc = exc
                continue
            raise
    assert last_exc is not None
    raise last_exc


# --- Unified facade ------------------------------------------------------- #


def make_openai_client(api_key: str | None = None) -> OpenAI:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def make_elevenlabs_client(api_key: str | None = None) -> ElevenLabs:
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    return ElevenLabs(api_key=api_key, timeout=120.0)
