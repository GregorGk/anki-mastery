"""Tests for the model-tier routing in build/lib/llm.py.

These tests use a mock httpx transport so no real API calls are made.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.llm import (  # noqa: E402
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_PREMIUM_MODEL,
    TIER_DEFAULT,
    TIER_PREMIUM,
    AnthropicClient,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["answer"],
}


class _FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 100
        self.output_tokens = 20
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class _FakeToolUseBlock:
    def __init__(self, name: str, value: dict) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = value


class _FakeResponse:
    def __init__(self, decision: dict) -> None:
        self.content = [_FakeToolUseBlock("test_tool", decision)]
        self.usage = _FakeUsage()
        self.stop_reason = "end_turn"


class TestTierRouting:
    """Verify that tier param routes to the correct model."""

    def test_default_tier_uses_default_model(self):
        client = AnthropicClient(enable_caching=False)
        captured: dict[str, Any] = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeResponse({"answer": "a"})

        with patch.object(client.client.messages, "create", side_effect=fake_create):
            client.call_tool(
                system="x",
                user_message="y",
                tool_name="test_tool",
                tool_input_schema=SCHEMA,
            )

        assert captured["model"] == DEFAULT_GENERATOR_MODEL

    def test_premium_tier_uses_premium_model(self):
        client = AnthropicClient(enable_caching=False)
        captured: dict[str, Any] = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeResponse({"answer": "b"})

        with patch.object(client.client.messages, "create", side_effect=fake_create):
            client.call_tool(
                system="x",
                user_message="y",
                tool_name="test_tool",
                tool_input_schema=SCHEMA,
                tier=TIER_PREMIUM,
            )

        assert captured["model"] == DEFAULT_PREMIUM_MODEL

    def test_invalid_tier_raises(self):
        client = AnthropicClient(enable_caching=False)
        with pytest.raises(ValueError, match="Invalid tier"):
            client.call_tool(
                system="x",
                user_message="y",
                tool_name="test_tool",
                tool_input_schema=SCHEMA,
                tier="ultra",  # not a real tier
            )

    def test_custom_premium_model(self):
        client = AnthropicClient(
            enable_caching=False, premium_model="claude-opus-special"
        )
        captured: dict[str, Any] = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeResponse({"answer": "a"})

        with patch.object(client.client.messages, "create", side_effect=fake_create):
            client.call_tool(
                system="x",
                user_message="y",
                tool_name="test_tool",
                tool_input_schema=SCHEMA,
                tier=TIER_PREMIUM,
            )

        assert captured["model"] == "claude-opus-special"

    def test_per_tier_telemetry(self):
        """Mixed default + premium calls should be counted per-tier."""
        client = AnthropicClient(enable_caching=False)

        def fake_create(**kwargs):
            return _FakeResponse({"answer": "a"})

        with patch.object(client.client.messages, "create", side_effect=fake_create):
            for _ in range(3):
                client.call_tool(
                    system="x",
                    user_message="y",
                    tool_name="t",
                    tool_input_schema=SCHEMA,
                )
            for _ in range(2):
                client.call_tool(
                    system="x",
                    user_message="y",
                    tool_name="t",
                    tool_input_schema=SCHEMA,
                    tier=TIER_PREMIUM,
                )

        stats = client.cache_stats
        assert stats["calls"] == 5
        assert stats["calls_by_tier"][TIER_DEFAULT] == 3
        assert stats["calls_by_tier"][TIER_PREMIUM] == 2
        # Each call has 100 input + 20 output tokens
        assert stats["tokens_by_tier"][TIER_DEFAULT]["input"] == 300
        assert stats["tokens_by_tier"][TIER_PREMIUM]["input"] == 200
        assert stats["tokens_by_tier"][TIER_DEFAULT]["output"] == 60
        assert stats["tokens_by_tier"][TIER_PREMIUM]["output"] == 40

    def test_audit_records_tier_and_actual_model(self, tmp_path):
        """Audit JSONL must record both `tier` and the resolved `model`."""
        audit = tmp_path / "audit.jsonl"
        client = AnthropicClient(audit_path=audit, enable_caching=False)

        def fake_create(**kwargs):
            return _FakeResponse({"answer": "a"})

        with patch.object(client.client.messages, "create", side_effect=fake_create):
            client.call_tool(
                system="x",
                user_message="y",
                tool_name="t",
                tool_input_schema=SCHEMA,
                tier=TIER_DEFAULT,
            )
            client.call_tool(
                system="x",
                user_message="y",
                tool_name="t",
                tool_input_schema=SCHEMA,
                tier=TIER_PREMIUM,
            )

        records = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()]
        assert len(records) == 2
        assert records[0]["tier"] == TIER_DEFAULT
        assert records[0]["model"] == DEFAULT_GENERATOR_MODEL
        assert records[1]["tier"] == TIER_PREMIUM
        assert records[1]["model"] == DEFAULT_PREMIUM_MODEL
