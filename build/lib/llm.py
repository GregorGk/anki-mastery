"""Anthropic LLM wrapper with Tool Use enforcement, prompt caching, and retry.

Used by every Stage that calls Anthropic. Prompt caching is mandatory for
high-volume stages (1.5, 2, 4, 5, 5.5) — the system prompt is cached so that
per-call cost drops ~5–6× because cached reads bill at 10% of base input rate.

Cache hit/miss telemetry is logged in audit JSONL via:
- `cache_creation_input_tokens` (first call seeds the cache)
- `cache_read_input_tokens` (subsequent calls within ~5 min TTL)

Validation expectation: cache-hit ratio should reach >95% within the first
~50 calls of a sustained stage. If it doesn't, the system prompt is being
mutated between calls (a bug) — cached reads will be 0.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from dotenv import load_dotenv

# Same override rationale as stage_1a.py — shell env may have an empty value.
load_dotenv(override=True)

# Default model — registry will replace this once config/models.yaml exists.
DEFAULT_GENERATOR_MODEL = "claude-sonnet-4-5"

# Premium tier — used selectively for high-risk, low-volume rows where
# Sonnet's quality ceiling matters (function-word polysemy, low-confidence
# pre-classifier outputs, top-100 examples, hard auditor cases). Per-call
# cost is ~5× Sonnet's, but at <100 calls the absolute spend is trivial
# (under $2). Override via AnthropicClient(premium_model=...) or by passing
# `tier="premium"` to call_tool.
DEFAULT_PREMIUM_MODEL = "claude-opus-4-5"

TIER_DEFAULT = "default"
TIER_PREMIUM = "premium"
VALID_TIERS = {TIER_DEFAULT, TIER_PREMIUM}

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 6
_BASE_BACKOFF_SEC = 1.0
_MAX_BACKOFF_SEC = 60.0

# Caching minimum: Anthropic requires the cached block to be >= 1024 tokens
# for Sonnet/Opus, >= 2048 for Haiku. We don't enforce that here — the API
# returns 0 cache_creation_input_tokens for sub-threshold blocks, which we
# detect via the cache_hit_ratio invariant.
_CACHE_CONTROL = {"type": "ephemeral"}


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS
    return False


def _backoff(attempt: int) -> float:
    """Full jitter exponential backoff."""
    cap = min(_MAX_BACKOFF_SEC, _BASE_BACKOFF_SEC * (2**attempt))
    return random.uniform(0, cap)


class AnthropicClient:
    """Synchronous Anthropic client with Tool-Use enforcement, prompt caching,
    and retries.

    Args:
        model: model ID (defaults to claude-sonnet-4-5)
        api_key: explicit API key (else reads ANTHROPIC_API_KEY env var)
        audit_path: append per-call provenance to this JSONL
        enable_caching: if True (default), wraps the system prompt with
            cache_control. Disable for tests or single-shot calls where
            caching adds no value.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        premium_model: str | None = None,
        api_key: str | None = None,
        audit_path: str | Path | None = None,
        enable_caching: bool = True,
    ) -> None:
        self.model = model or DEFAULT_GENERATOR_MODEL
        self.premium_model = premium_model or DEFAULT_PREMIUM_MODEL
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_caching = enable_caching
        # Audit-write lock so concurrent threads don't interleave JSONL lines.
        self._audit_lock = threading.Lock()
        # Telemetry for cache hit ratio (read across threads).
        self._stats_lock = threading.Lock()
        self._cache_creation_tokens = 0
        self._cache_read_tokens = 0
        self._uncached_input_tokens = 0
        self._calls = 0
        # Per-tier counters for cost auditing.
        self._calls_by_tier: dict[str, int] = {TIER_DEFAULT: 0, TIER_PREMIUM: 0}
        self._tokens_by_tier: dict[str, dict] = {
            TIER_DEFAULT: {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0},
            TIER_PREMIUM: {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0},
        }

    @property
    def cache_stats(self) -> dict:
        """Snapshot of cumulative cache telemetry."""
        with self._stats_lock:
            total_input = (
                self._cache_creation_tokens
                + self._cache_read_tokens
                + self._uncached_input_tokens
            )
            ratio = (
                self._cache_read_tokens / total_input if total_input > 0 else 0.0
            )
            return {
                "calls": self._calls,
                "cache_creation_tokens": self._cache_creation_tokens,
                "cache_read_tokens": self._cache_read_tokens,
                "uncached_input_tokens": self._uncached_input_tokens,
                "cache_hit_ratio": ratio,
                "calls_by_tier": dict(self._calls_by_tier),
                "tokens_by_tier": {
                    tier: dict(toks) for tier, toks in self._tokens_by_tier.items()
                },
            }

    def _model_for_tier(self, tier: str) -> str:
        """Resolve tier → model ID. Raises on invalid tier."""
        if tier == TIER_PREMIUM:
            return self.premium_model
        if tier == TIER_DEFAULT:
            return self.model
        raise ValueError(
            f"Invalid tier {tier!r}; must be one of {sorted(VALID_TIERS)}"
        )

    def _build_system_param(self, system: str) -> Any:
        """Wrap the system prompt with cache_control if caching is enabled."""
        if not self.enable_caching:
            return system
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": _CACHE_CONTROL,
            }
        ]

    def call_tool(
        self,
        *,
        system: str,
        user_message: str,
        tool_name: str,
        tool_input_schema: dict,
        tool_description: str = "",
        max_tokens: int = 1024,
        stage: str = "",
        provenance_key: str = "",
        tier: str = TIER_DEFAULT,
    ) -> dict:
        """Force tool invocation; return the structured tool input as a dict.

        Args:
            tier: 'default' (Sonnet) or 'premium' (Opus). Premium routes to a
                stronger model for high-risk, low-volume calls. Default
                tier is appropriate for >95% of pipeline calls.

        Raises if all retries fail or the model refuses to call the tool.
        """
        if tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier {tier!r}; must be one of {sorted(VALID_TIERS)}"
            )
        model_id = self._model_for_tier(tier)
        tools = [
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": tool_input_schema,
            }
        ]
        prompt_payload = {
            "model": model_id,
            "system": system,
            "user": user_message,
            "tool": tool_name,
            "schema": tool_input_schema,
        }
        prompt_hash = _hash_payload(prompt_payload)
        system_param = self._build_system_param(system)

        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self.client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    system=system_param,
                    tools=tools,
                    tool_choice={"type": "tool", "name": tool_name},
                    messages=[{"role": "user", "content": user_message}],
                )
            except Exception as exc:
                last_exc = exc
                if not _retryable(exc) or attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_backoff(attempt))
                continue

            # Find the tool_use block in the response
            tool_use_block = next(
                (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
                None,
            )
            if tool_use_block is None:
                # Model returned text instead of tool — retry once, else fail
                last_exc = RuntimeError(
                    f"Model did not call tool {tool_name!r}; got: "
                    f"{resp.content[:1] if resp.content else 'empty'}"
                )
                if attempt == _MAX_ATTEMPTS - 1:
                    raise last_exc
                time.sleep(_backoff(attempt))
                continue

            decision = dict(tool_use_block.input)
            response_hash = _hash_payload(decision)

            # Pull cache telemetry from the usage object.
            usage = resp.usage
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0

            with self._stats_lock:
                self._calls += 1
                self._cache_creation_tokens += cache_creation
                self._cache_read_tokens += cache_read
                self._uncached_input_tokens += input_tokens
                # Per-tier breakdown for cost auditing
                self._calls_by_tier[tier] += 1
                self._tokens_by_tier[tier]["input"] += input_tokens
                self._tokens_by_tier[tier]["output"] += output_tokens
                self._tokens_by_tier[tier]["cache_read"] += cache_read
                self._tokens_by_tier[tier]["cache_create"] += cache_creation

            if self.audit_path:
                audit_record = {
                    "stage": stage,
                    "provenance_key": provenance_key,
                    "model": model_id,
                    "tier": tier,
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "decision": decision,
                    "attempt": attempt,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                    "stop_reason": resp.stop_reason,
                }
                with self._audit_lock:
                    with self.audit_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

            return decision

        # Unreachable in practice
        raise last_exc or RuntimeError("LLM call failed without exception")
