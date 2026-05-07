"""Anthropic LLM wrapper with Tool Use enforcement and basic retry.

Stage 1a uses this only for idiom-expansion resolution (~12 calls). Later stages
will extend with batch APIs, semaphore-bounded concurrency, and provenance JSONL.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
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

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 6
_BASE_BACKOFF_SEC = 1.0
_MAX_BACKOFF_SEC = 60.0


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
    """Synchronous Anthropic client with Tool-Use enforcement and retries."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.model = model or DEFAULT_GENERATOR_MODEL
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

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

        Raises if all retries fail or the model refuses to call the tool.
        """
        tools = [
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": tool_input_schema,
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
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
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

            if self.audit_path:
                audit_record = {
                    "stage": stage,
                    "provenance_key": provenance_key,
                    "model": self.model,
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "decision": decision,
                    "attempt": attempt,
                    "input_tokens": getattr(resp.usage, "input_tokens", None),
                    "output_tokens": getattr(resp.usage, "output_tokens", None),
                    "stop_reason": resp.stop_reason,
                }
                with self.audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

            return decision

        # Unreachable in practice
        raise last_exc or RuntimeError("LLM call failed without exception")
