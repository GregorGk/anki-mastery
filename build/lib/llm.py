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
from openai import (
    APIConnectionError as OpenAIConnectionError,
    APIStatusError as OpenAIStatusError,
    APITimeoutError as OpenAITimeoutError,
    OpenAI,
    RateLimitError as OpenAIRateLimitError,
)

# Same override rationale as stage_1a.py — shell env may have an empty value.
load_dotenv(override=True)

# OpenAI validator model — used as the cross-family validator at Stage 4
# (different family from the Anthropic generator). Mid-tier, cheap, fast.
# Override per-call or via constructor if you want to upgrade.
DEFAULT_OPENAI_VALIDATOR_MODEL = "gpt-4o-mini"

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

    # ------------------------------------------------------------------ #
    # Batch API
    # ------------------------------------------------------------------ #

    def submit_batch(
        self,
        requests: list[dict],
        *,
        system: str,
        tool_name: str,
        tool_input_schema: dict,
        tool_description: str = "",
        max_tokens: int = 1024,
        tier: str = TIER_DEFAULT,
    ) -> str:
        """Submit a batch of tool-use requests. Returns batch_id.

        Each request dict needs `custom_id` (unique str) and `user_message`.
        """
        if tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier {tier!r}; must be one of {sorted(VALID_TIERS)}"
            )
        model_id = self._model_for_tier(tier)
        system_param = self._build_system_param(system)
        tools = [
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": tool_input_schema,
            }
        ]

        batch_requests = []
        for req in requests:
            cid = req.get("custom_id")
            user_msg = req.get("user_message")
            if not cid or not user_msg:
                raise ValueError(
                    f"Batch request needs 'custom_id' and 'user_message': {req}"
                )
            batch_requests.append(
                {
                    "custom_id": cid,
                    "params": {
                        "model": model_id,
                        "max_tokens": max_tokens,
                        "system": system_param,
                        "tools": tools,
                        "tool_choice": {"type": "tool", "name": tool_name},
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                }
            )

        result = self.client.messages.batches.create(requests=batch_requests)
        return result.id

    def poll_batch(
        self,
        batch_id: str,
        *,
        timeout_s: int = 86400,
        poll_interval_s: int = 30,
        progress_callback=None,
    ) -> dict[str, dict]:
        """Block until the batch completes; return {custom_id: tool_input_dict}.

        Errored, expired, or canceled items are returned with a special
        sentinel key `_error` instead of the tool input dict.
        Results are NOT logged to audit JSONL here — callers decide whether
        to log per-row decisions after parsing.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            status = getattr(batch, "processing_status", None)
            if progress_callback:
                progress_callback(status, batch)
            if status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Batch {batch_id} did not finish within {timeout_s}s "
                    f"(last status: {status})"
                )
            time.sleep(poll_interval_s)

        # Stream results
        out: dict[str, dict] = {}
        for entry in self.client.messages.batches.results(batch_id):
            cid = entry.custom_id
            r = entry.result
            rtype = getattr(r, "type", None)
            if rtype == "succeeded":
                msg = r.message
                tool_block = next(
                    (b for b in msg.content if getattr(b, "type", None) == "tool_use"),
                    None,
                )
                if tool_block is None:
                    out[cid] = {"_error": "no_tool_use_block"}
                    continue
                # Telemetry
                usage = msg.usage
                cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
                cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                with self._stats_lock:
                    self._calls += 1
                    self._cache_creation_tokens += cache_creation
                    self._cache_read_tokens += cache_read
                    self._uncached_input_tokens += input_tokens
                # Note: tier tracking on batch is approximated by the submitter
                # (caller can pass tier explicitly via call_tool_batch).
                out[cid] = dict(tool_block.input)
            else:
                # errored / canceled / expired
                err = getattr(r, "error", None)
                err_type = getattr(err, "type", None) if err else rtype
                err_message = getattr(err, "message", None) if err else None
                out[cid] = {
                    "_error": err_type or "unknown",
                    "_error_message": err_message or "",
                    "_error_status": rtype,
                }
        return out

    def call_tool_batch(
        self,
        rows: list[dict],
        *,
        system: str,
        tool_name: str,
        tool_input_schema: dict,
        tool_description: str = "",
        max_tokens: int = 1024,
        tier: str = TIER_DEFAULT,
        stage: str = "",
        sync_threshold: int = 100,
        timeout_s: int = 86400,
        poll_interval_s: int = 30,
        progress_callback=None,
    ) -> dict[str, dict]:
        """High-level batch helper. Falls through to per-row sync if rows < sync_threshold.

        `rows` is a list of dicts each containing:
            - id: unique string (used as custom_id and provenance key)
            - user_message: the per-row prompt content

        Returns {id: decision_dict_or_error}. Errors are flagged with a `_error`
        key so callers can detect and route them.
        """
        if not rows:
            return {}

        # Small batches — go sync.
        if len(rows) < sync_threshold:
            results: dict[str, dict] = {}
            for row in rows:
                cid = row["id"]
                try:
                    results[cid] = self.call_tool(
                        system=system,
                        user_message=row["user_message"],
                        tool_name=tool_name,
                        tool_input_schema=tool_input_schema,
                        tool_description=tool_description,
                        max_tokens=max_tokens,
                        tier=tier,
                        stage=stage,
                        provenance_key=cid,
                    )
                except Exception as exc:
                    results[cid] = {
                        "_error": type(exc).__name__,
                        "_error_message": str(exc),
                    }
            return results

        # Large enough to use batch.
        batch_id = self.submit_batch(
            rows,
            system=system,
            tool_name=tool_name,
            tool_input_schema=tool_input_schema,
            tool_description=tool_description,
            max_tokens=max_tokens,
            tier=tier,
        )
        return self.poll_batch(
            batch_id,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            progress_callback=progress_callback,
        )


# ---------------------------------------------------------------------------- #
# OpenAI client (cross-family validator role)
# ---------------------------------------------------------------------------- #


def _openai_retryable(exc: Exception) -> bool:
    if isinstance(exc, (OpenAIRateLimitError, OpenAIConnectionError, OpenAITimeoutError)):
        return True
    if isinstance(exc, OpenAIStatusError):
        status = getattr(exc, "status_code", None)
        return status in _RETRYABLE_STATUS
    return False


class OpenAIClient:
    """Synchronous OpenAI client with Tool-Use enforcement and retry.

    Mirrors AnthropicClient's interface so Stage scripts can swap providers
    by changing the client instance. Used at Stage 4 as the cross-family
    validator: a different model family from the generator (Anthropic) means
    shared blind spots have to be bi-coincident, not single-model.

    OpenAI's automatic prompt caching kicks in for prompts ≥1024 tokens with
    no extra parameters — no `cache_control` block needed (unlike Anthropic).
    The first ~50 calls within a 5-min window establish the cache; subsequent
    calls bill cached input at 50% of base rate.

    Args:
        model: model ID (defaults to gpt-4o-mini)
        api_key: explicit API key (else reads OPENAI_API_KEY env var)
        audit_path: append per-call provenance to this JSONL
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.model = model or DEFAULT_OPENAI_VALIDATOR_MODEL
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._calls = 0
        self._cached_input_tokens = 0  # tokens read from cache
        self._uncached_input_tokens = 0
        self._output_tokens = 0

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            total_input = self._cached_input_tokens + self._uncached_input_tokens
            ratio = (
                self._cached_input_tokens / total_input if total_input > 0 else 0.0
            )
            return {
                "calls": self._calls,
                "cached_input_tokens": self._cached_input_tokens,
                "uncached_input_tokens": self._uncached_input_tokens,
                "output_tokens": self._output_tokens,
                "cache_hit_ratio": ratio,
            }

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
    ) -> dict:
        """Force tool invocation; return the structured tool input as a dict.

        OpenAI uses chat completions with function-calling tools.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": tool_input_schema,
                },
            }
        ]
        prompt_payload = {
            "model": self.model,
            "system": system,
            "user": user_message,
            "tool": tool_name,
            "schema": tool_input_schema,
        }
        prompt_hash = _hash_payload(prompt_payload)

        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    tools=tools,
                    tool_choice={
                        "type": "function",
                        "function": {"name": tool_name},
                    },
                )
            except Exception as exc:
                last_exc = exc
                if not _openai_retryable(exc) or attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_backoff(attempt))
                continue

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []
            if not tool_calls:
                last_exc = RuntimeError(
                    f"OpenAI did not call tool {tool_name!r}; got: "
                    f"{msg.content[:80] if msg.content else 'empty'}"
                )
                if attempt == _MAX_ATTEMPTS - 1:
                    raise last_exc
                time.sleep(_backoff(attempt))
                continue

            tc = tool_calls[0]
            try:
                decision = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                last_exc = RuntimeError(
                    f"OpenAI returned malformed tool arguments: {exc}; "
                    f"raw: {tc.function.arguments[:200]}"
                )
                if attempt == _MAX_ATTEMPTS - 1:
                    raise last_exc
                time.sleep(_backoff(attempt))
                continue
            response_hash = _hash_payload(decision)

            usage = resp.usage
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            # OpenAI exposes cached_tokens via prompt_tokens_details (newer API)
            ptd = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = getattr(ptd, "cached_tokens", 0) if ptd else 0
            uncached = max(input_tokens - cached_tokens, 0)

            with self._stats_lock:
                self._calls += 1
                self._cached_input_tokens += cached_tokens
                self._uncached_input_tokens += uncached
                self._output_tokens += output_tokens

            if self.audit_path:
                audit_record = {
                    "stage": stage,
                    "provenance_key": provenance_key,
                    "model": self.model,
                    "provider": "openai",
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "decision": decision,
                    "attempt": attempt,
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": resp.choices[0].finish_reason,
                }
                with self._audit_lock:
                    with self.audit_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

            return decision

        raise last_exc or RuntimeError("OpenAI call failed without exception")

    def call_tool_concurrent(
        self,
        rows: list[dict],
        *,
        system: str,
        tool_name: str,
        tool_input_schema: dict,
        tool_description: str = "",
        max_tokens: int = 1024,
        stage: str = "",
        concurrency: int = 6,
    ) -> dict[str, dict]:
        """Concurrent per-row sync calls using a thread pool.

        We don't implement OpenAI Batch API here — it requires file-upload
        + polling and is overkill for the validator pass at Stage 4 (where
        ~5,725 rows × Sonnet generator cost ~$30 already, so OpenAI batch
        50% saving on validator is small absolute spend).

        Each row dict: {"id": str, "user_message": str}.
        Returns {id: decision_dict_or_error}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not rows:
            return {}

        results: dict[str, dict] = {}

        def _one(row: dict) -> tuple[str, dict]:
            cid = row["id"]
            try:
                d = self.call_tool(
                    system=system,
                    user_message=row["user_message"],
                    tool_name=tool_name,
                    tool_input_schema=tool_input_schema,
                    tool_description=tool_description,
                    max_tokens=max_tokens,
                    stage=stage,
                    provenance_key=cid,
                )
                return cid, d
            except Exception as exc:
                return cid, {
                    "_error": type(exc).__name__,
                    "_error_message": str(exc),
                }

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_one, row) for row in rows]
            for fut in as_completed(futures):
                cid, decision = fut.result()
                results[cid] = decision

        return results
