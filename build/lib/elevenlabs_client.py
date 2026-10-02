"""ElevenLabs TTS client with retry, idempotency, and locked quality settings.

Locked decisions (see docs/plan.md § "Audio quality settings"):

- model_id              = passed explicitly by every production script
                          (multilingual_v2 → flash_v2_5 [Stage 9] → eleven_v3
                          [Stage 11] → eleven_v4 [Stage 19]). The module
                          default below is only a legacy fallback.
- output_format         = "pcm_44100"  (raw 16-bit signed-LE PCM, 44.1 kHz mono;
                          Pro tier)
- voice_settings        = stability=0.65, similarity_boost=0.80, style=0.0,
                          use_speaker_boost=True. eleven_v4 only exposes
                          stability + similarity (`/v1/models` reports
                          can_use_style / can_use_speaker_boost = False), so
                          `voice_settings_for_model` drops the other two.
- seed                  = stable_hash(sense_id + clip_type + version)  (best-effort
                          determinism); callers may pass an explicit `seed`.
- apply_text_normalization = "auto"

Retry policy:
- Retryable: HTTP 429, 500, 502, 503, 504, 408, connection / timeout errors.
- Non-retryable: 400, 401, 403, 404, 422.
- Out of credits → `QuotaExceeded` (never retried) so batch runs stop cleanly.
- Account-wide refusals → `TTSUnavailable` (QuotaExceeded's base class): 401
  (bad / revoked key), 402, 403, and 429s that outlast every attempt. They hit
  every clip alike, so a batch run must stop rather than log each clip failed.
  A 5xx that outlasts the retries is re-raised as is: it can be clip-specific.
- Backoff: full jitter, base 1s, cap 60s, up to 6 attempts. Honors Retry-After
  if present (header keys are lower-cased by the SDK).
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass, field

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


class TTSUnavailable(RuntimeError):
    """Raised when ElevenLabs refuses every call, not just this one: HTTP 401
    (bad or revoked key), 402, 403, or 429s that outlast every attempt.

    Never retried: the run must stop, flush its state, and resume once the key,
    plan or rate limit is fixed.
    """


class QuotaExceeded(TTSUnavailable):
    """Raised when the account is out of credits (`quota_exceeded`).

    Never retried: the run must stop, flush its state, and resume after the
    balance is topped up / the subscription renews.
    """

# Locked voice settings.
LOCKED_VOICE_SETTINGS = VoiceSettings(
    stability=0.65,
    similarity_boost=0.80,
    style=0.0,
    use_speaker_boost=True,
)

# Models that only accept stability + similarity_boost.
_STABILITY_SIMILARITY_ONLY_PREFIXES = ("eleven_v4",)

RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_CODES = {400, 401, 403, 404, 422}
# Account-wide refusals → TTSUnavailable (402 is in neither set above).
UNAVAILABLE_HTTP_CODES = {401, 402, 403}

# Response headers worth keeping per call (keys are lower-case in the SDK).
META_HEADERS = (
    "character-cost",
    "request-id",
    "current-concurrent-requests",
    "maximum-concurrent-requests",
)


def voice_settings_for_model(model_id: str, base: VoiceSettings) -> VoiceSettings:
    """Return the subset of `base` the given model actually accepts."""
    if any(model_id.startswith(p) for p in _STABILITY_SIMILARITY_ONLY_PREFIXES):
        return VoiceSettings(
            stability=base.stability,
            similarity_boost=base.similarity_boost,
        )
    return base


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
    # Filled by generate_pcm_meta() only.
    character_cost: int | None = None
    request_id: str = ""
    headers: dict = field(default_factory=dict)
    latency_ms: int = 0


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
        apply_text_normalization: str = "auto",
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
        self.voice_settings = voice_settings_for_model(model_id, voice_settings)
        self.pronunciation_dict_locators = pronunciation_dict_locators
        self.fail_fast_on_429 = fail_fast_on_429
        self.apply_text_normalization = apply_text_normalization
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
        seed: int | None = None,
        voice_settings: VoiceSettings | None = None,
    ) -> ElevenLabsResult:
        """Generate raw PCM bytes for one clip with retries.

        Caller should pipe the bytes into ffmpeg (loudness normalization +
        MP3 encode) — see build/lib/loudness.py.

        `seed` overrides the (sense_id, clip_type, version) default — Stage 19
        salts it per take so best-of-N candidates never share a seed.
        `voice_settings` overrides the client's settings for this call only.
        """
        return self._generate(
            text=text, voice_id=voice_id, sense_id=sense_id, clip_type=clip_type,
            version=version, seed=seed, voice_settings=voice_settings, meta=False,
        )

    def generate_pcm_meta(
        self,
        *,
        text: str,
        voice_id: str,
        sense_id: str,
        clip_type: str,
        version: int,
        seed: int | None = None,
        voice_settings: VoiceSettings | None = None,
    ) -> ElevenLabsResult:
        """Like generate_pcm, but via the raw-response API so the result also
        carries `character-cost`, `request-id` and the concurrency headers."""
        return self._generate(
            text=text, voice_id=voice_id, sense_id=sense_id, clip_type=clip_type,
            version=version, seed=seed, voice_settings=voice_settings, meta=True,
        )

    # --- Internals --------------------------------------------------------- #

    def _generate(
        self,
        *,
        text: str,
        voice_id: str,
        sense_id: str,
        clip_type: str,
        version: int,
        seed: int | None,
        voice_settings: VoiceSettings | None,
        meta: bool,
    ) -> ElevenLabsResult:
        if seed is None:
            seed = stable_seed(sense_id, clip_type, version)
        settings = (voice_settings_for_model(self.model_id, voice_settings)
                    if voice_settings is not None else self.voice_settings)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                t0 = time.monotonic()
                if meta:
                    pcm_bytes, headers = self._call_once_meta(
                        text=text, voice_id=voice_id, seed=seed, voice_settings=settings)
                else:
                    pcm_bytes = self._call_once(
                        text=text, voice_id=voice_id, seed=seed, voice_settings=settings)
                    headers = {}
                latency_ms = int((time.monotonic() - t0) * 1000)
                cost = headers.get("character-cost")
                return ElevenLabsResult(
                    audio_pcm=pcm_bytes,
                    sample_rate=self._pcm_sample_rate,
                    sample_width=self._pcm_sample_width,
                    channels=self._pcm_channels,
                    voice_id=voice_id,
                    seed=seed,
                    attempt=attempt,
                    char_count=len(text),
                    character_cost=int(cost) if cost not in (None, "") else None,
                    request_id=headers.get("request-id", ""),
                    headers=headers,
                    latency_ms=latency_ms,
                )
            except Exception as exc:  # noqa: BLE001 — re-classify below
                if _is_quota_exceeded(exc):
                    raise QuotaExceeded(
                        f"ElevenLabs quota exceeded: {getattr(exc, 'body', exc)}") from exc
                status = _status(exc)
                if status in UNAVAILABLE_HTTP_CODES:
                    raise TTSUnavailable(
                        f"ElevenLabs HTTP {status}: {getattr(exc, 'body', exc)}") from exc
                if not self._is_retryable(exc):
                    raise
                last_exc = exc
                if attempt < self.max_attempts:
                    wait_s = self._compute_wait(exc, attempt)
                    time.sleep(wait_s)
        # Exhausted retries
        assert last_exc is not None
        if _status(last_exc) == 429:
            raise TTSUnavailable(
                f"ElevenLabs HTTP 429 after {self.max_attempts} attempts: "
                f"{getattr(last_exc, 'body', last_exc)}") from last_exc
        raise last_exc

    def _request_kwargs(self, *, text: str, voice_id: str, seed: int,
                        voice_settings: VoiceSettings | None = None) -> dict:
        kwargs = dict(
            voice_id=voice_id,
            text=text,
            model_id=self.model_id,
            output_format=self.output_format,
            language_code=self.language_code,
            voice_settings=voice_settings or self.voice_settings,
            seed=seed,
            apply_text_normalization=self.apply_text_normalization,
        )
        if self.pronunciation_dict_locators:
            kwargs["pronunciation_dictionary_locators"] = self.pronunciation_dict_locators
        return kwargs

    def _call_once(self, *, text: str, voice_id: str, seed: int,
                   voice_settings: VoiceSettings | None = None) -> bytes:
        kwargs = self._request_kwargs(text=text, voice_id=voice_id, seed=seed,
                                      voice_settings=voice_settings)
        chunks = self._client.text_to_speech.convert(**kwargs)
        # convert() returns Iterator[bytes]; concatenate.
        return b"".join(chunks)

    def _call_once_meta(self, *, text: str, voice_id: str, seed: int,
                        voice_settings: VoiceSettings | None = None
                        ) -> tuple[bytes, dict]:
        kwargs = self._request_kwargs(text=text, voice_id=voice_id, seed=seed,
                                      voice_settings=voice_settings)
        voice = kwargs.pop("voice_id")
        with self._client.text_to_speech.with_raw_response.convert(voice, **kwargs) as resp:
            audio = b"".join(resp.data)
            all_headers = {k.lower(): v for k, v in resp.headers.items()}
        headers = {k: all_headers[k] for k in META_HEADERS if k in all_headers}
        return audio, headers

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
        # ElevenLabs' ApiError exposes `headers` as dict(httpx.Headers), whose
        # keys are lower-case — look the header up case-insensitively.
        headers = getattr(exc, "headers", None) or {}
        if not hasattr(headers, "items"):
            return None
        ra = next((v for k, v in headers.items()
                   if str(k).lower() == "retry-after"), None)
        if not ra:
            return None
        try:
            return float(ra)
        except (TypeError, ValueError):
            return None


def _status(exc: Exception) -> int | None:
    """HTTP status of an ElevenLabs SDK error (None for network errors)."""
    return getattr(exc, "status_code", None) if isinstance(exc, ElevenLabsApiError) else None


def _is_quota_exceeded(exc: Exception) -> bool:
    """True when ElevenLabs rejected the call for lack of credits."""
    if not isinstance(exc, ElevenLabsApiError):
        return False
    body = getattr(exc, "body", None)
    return "quota_exceeded" in str(body).lower()
