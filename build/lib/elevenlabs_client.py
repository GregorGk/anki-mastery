"""ElevenLabs TTS client with retry, idempotency, and locked quality settings.

Locked decisions (see docs/plan.md § "Audio quality settings"):

- model_id              = "eleven_multilingual_v2"
- output_format         = "pcm_44100"  (raw 16-bit signed-LE PCM, 44.1 kHz mono)
- voice_settings        = stability=0.65, similarity_boost=0.80, style=0.0,
                          use_speaker_boost=True
- seed                  = stable_hash(sense_id + clip_type + version)  (best-effort
                          determinism)
- apply_text_normalization = "auto"

Retry policy:
- Retryable: HTTP 429, 500, 502, 503, 504, 408, connection / timeout errors.
- Non-retryable: 400, 401, 403, 404, 422.
- Backoff: full jitter, base 1s, cap 60s, up to 6 attempts. Honors Retry-After
  if present.
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass

import httpx
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError as ElevenLabsApiError

DEFAULT_MODEL_ID = "eleven_flash_v2_5"  # Stage 9 migration: was eleven_multilingual_v2
DEFAULT_OUTPUT_FORMAT = "pcm_44100"
DEFAULT_LANGUAGE_CODE = "pt"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_BACKOFF_CAP_S = 60.0


class RateLimitExceeded(RuntimeError):
    """Raised when ElevenLabs returns 429 and `fail_fast_on_429` is set.

    Used to abort batch jobs (e.g. Stage 9 migration) immediately when
    the concurrent-request ceiling is hit — gives operator visibility
    instead of invisible retries and backoffs.
    """

# Locked voice settings.
LOCKED_VOICE_SETTINGS = VoiceSettings(
    stability=0.65,
    similarity_boost=0.80,
    style=0.0,
    use_speaker_boost=True,
)

RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_CODES = {400, 401, 403, 404, 422}


def stable_seed(sense_id: str, clip_type: str, version: int) -> int:
    """Deterministic seed for ElevenLabs.

    Best-effort determinism per docs ("Determinism is not guaranteed"). We pick
    a stable function of the inputs so a regen at the same version reproduces
    the same audio when ElevenLabs honors the seed.
    """
    payload = f"{sense_id}|{clip_type}|{version}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:8], 16) % 4_294_967_295


@dataclass
class ElevenLabsResult:
    audio_pcm: bytes
    sample_rate: int  # 44100
    sample_width: int  # 2 (16-bit)
    channels: int  # 1
    voice_id: str
    seed: int
    attempt: int  # 1-indexed; how many attempts it took
    char_count: int


class ElevenLabsClient:
    """Thin wrapper around the official ElevenLabs Python SDK with retry."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        language_code: str = DEFAULT_LANGUAGE_CODE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        voice_settings: VoiceSettings = LOCKED_VOICE_SETTINGS,
        rng: random.Random | None = None,
        pronunciation_dict_locators: list[dict] | None = None,
        fail_fast_on_429: bool = False,
    ) -> None:
        """Construct an ElevenLabs TTS client.

        pronunciation_dict_locators: optional list of dicts with shape
            [{"pronunciation_dictionary_id": "...", "version_id": "..."}, ...]
            Up to 3 per ElevenLabs API. When set, every TTS call attaches
            these locators so server-side alias rules apply before
            synthesis. Used in Stage 8 (alias dictionaries) and any
            future regen-flagged passes that should respell at TTS time.
        """
        api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set in environment")
        self.model_id = model_id
        self.output_format = output_format
        self.language_code = language_code
        self.max_attempts = max_attempts
        self.voice_settings = voice_settings
        self.pronunciation_dict_locators = pronunciation_dict_locators
        self.fail_fast_on_429 = fail_fast_on_429
        self._client = ElevenLabs(api_key=api_key, timeout=timeout_s)
        self._rng = rng or random.Random()
        # PCM constants for pcm_44100 (16-bit signed LE, mono).
        self._pcm_sample_rate = 44100
        self._pcm_sample_width = 2
        self._pcm_channels = 1

    # --- Public API -------------------------------------------------------- #

    def generate_pcm(
        self,
        *,
        text: str,
        voice_id: str,
        sense_id: str,
        clip_type: str,
        version: int,
    ) -> ElevenLabsResult:
        """Generate raw PCM bytes for one clip with retries.

        Caller should pipe the bytes into ffmpeg (loudness normalization +
        MP3 encode) — see build/lib/loudness.py.
        """
        seed = stable_seed(sense_id, clip_type, version)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                pcm_bytes = self._call_once(text=text, voice_id=voice_id, seed=seed)
                return ElevenLabsResult(
                    audio_pcm=pcm_bytes,
                    sample_rate=self._pcm_sample_rate,
                    sample_width=self._pcm_sample_width,
                    channels=self._pcm_channels,
                    voice_id=voice_id,
                    seed=seed,
                    attempt=attempt,
                    char_count=len(text),
                )
            except Exception as exc:  # noqa: BLE001 — re-classify below
                if not self._is_retryable(exc):
                    raise
                last_exc = exc
                if attempt < self.max_attempts:
                    wait_s = self._compute_wait(exc, attempt)
                    time.sleep(wait_s)
        # Exhausted retries
        assert last_exc is not None
        raise last_exc

    # --- Internals --------------------------------------------------------- #

    def _call_once(self, *, text: str, voice_id: str, seed: int) -> bytes:
        kwargs = dict(
            voice_id=voice_id,
            text=text,
            model_id=self.model_id,
            output_format=self.output_format,
            language_code=self.language_code,
            voice_settings=self.voice_settings,
            seed=seed,
            apply_text_normalization="auto",
        )
        if self.pronunciation_dict_locators:
            kwargs["pronunciation_dictionary_locators"] = self.pronunciation_dict_locators
        chunks = self._client.text_to_speech.convert(**kwargs)
        # convert() returns Iterator[bytes]; concatenate.
        return b"".join(chunks)

    def _is_retryable(self, exc: Exception) -> bool:
        # Fail-fast on 429 if requested — re-raises as RateLimitExceeded
        # so the caller (e.g. Stage 9 migration) can abort cleanly.
        if isinstance(exc, ElevenLabsApiError):
            status = getattr(exc, "status_code", None)
            if status == 429 and self.fail_fast_on_429:
                raise RateLimitExceeded(
                    f"429 from ElevenLabs; fail_fast_on_429 set. "
                    f"Headers: {getattr(exc, 'headers', None)}"
                ) from exc
        # Network / timeout flavors
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)):
            return True
        # ElevenLabs SDK error
        if isinstance(exc, ElevenLabsApiError):
            status = getattr(exc, "status_code", None)
            if status is None:
                return True
            if status in NON_RETRYABLE_HTTP_CODES:
                return False
            if status in RETRYABLE_HTTP_CODES:
                return True
            return False
        # Generic httpx/HTTP error fallback
        return False

    def _compute_wait(self, exc: Exception, attempt: int) -> float:
        # Honor Retry-After header if exposed by the exception's body.
        retry_after = self._extract_retry_after(exc)
        if retry_after is not None:
            return min(retry_after, DEFAULT_BACKOFF_CAP_S)
        # Full jitter exponential backoff
        ceiling = min(DEFAULT_BACKOFF_CAP_S, DEFAULT_BACKOFF_BASE_S * (2 ** (attempt - 1)))
        return self._rng.uniform(0, ceiling)

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        # ElevenLabs' ApiError exposes `headers` on some versions; defensive.
        headers = getattr(exc, "headers", None) or {}
        ra = headers.get("Retry-After") if hasattr(headers, "get") else None
        if not ra:
            return None
        try:
            return float(ra)
        except (TypeError, ValueError):
            return None
