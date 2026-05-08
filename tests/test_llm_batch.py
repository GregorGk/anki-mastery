"""Tests for Anthropic Batch API in build/lib/llm.py.

Mocked at the SDK level so no real API calls are made.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    "properties": {"answer": {"type": "string", "enum": ["a", "b"]}},
    "required": ["answer"],
}


# --- Fakes for batch SDK objects --------------------------------------------


class _FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 100
        self.output_tokens = 20
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class _FakeToolUseBlock:
    def __init__(self, value: dict) -> None:
        self.type = "tool_use"
        self.name = "test_tool"
        self.input = value


class _FakeMessage:
    def __init__(self, decision: dict) -> None:
        self.content = [_FakeToolUseBlock(decision)]
        self.usage = _FakeUsage()
        self.stop_reason = "end_turn"


class _FakeBatchResultItem:
    """Mirrors the Anthropic SDK BatchEntry shape we use."""

    def __init__(self, custom_id: str, status: str, decision: dict | None = None, error: str | None = None):
        self.custom_id = custom_id

        class _Result:
            def __init__(self) -> None:
                self.type = status
                if status == "succeeded":
                    self.message = _FakeMessage(decision or {})
                elif status == "errored":
                    err = MagicMock()
                    err.type = error or "api_error"
                    err.message = "fake error"
                    self.error = err

        self.result = _Result()


class _FakeBatch:
    def __init__(self, batch_id: str, processing_status: str = "ended") -> None:
        self.id = batch_id
        self.processing_status = processing_status


def _patch_messages_batches(client: AnthropicClient, *, results: list[_FakeBatchResultItem], statuses: list[str]):
    """Helper: install fakes for messages.batches.{create,retrieve,results}."""
    fake_create = MagicMock(return_value=_FakeBatch("batch_test_123"))
    state = {"index": 0}

    def fake_retrieve(batch_id: str):
        idx = min(state["index"], len(statuses) - 1)
        state["index"] += 1
        return _FakeBatch(batch_id, processing_status=statuses[idx])

    def fake_results(batch_id: str):
        for r in results:
            yield r

    client.client.messages.batches.create = fake_create
    client.client.messages.batches.retrieve = MagicMock(side_effect=fake_retrieve)
    client.client.messages.batches.results = MagicMock(side_effect=fake_results)
    return fake_create


# --- Tests --------------------------------------------------------------------


class TestSubmitBatch:
    def test_basic_submit(self):
        client = AnthropicClient(enable_caching=False)
        fake_create = _patch_messages_batches(
            client,
            results=[_FakeBatchResultItem("r1", "succeeded", {"answer": "a"})],
            statuses=["ended"],
        )

        batch_id = client.submit_batch(
            [
                {"custom_id": "r1", "user_message": "hello"},
                {"custom_id": "r2", "user_message": "world"},
            ],
            system="x",
            tool_name="test_tool",
            tool_input_schema=SCHEMA,
        )
        assert batch_id == "batch_test_123"

        # Verify request structure
        call_args = fake_create.call_args
        assert "requests" in call_args.kwargs
        reqs = call_args.kwargs["requests"]
        assert len(reqs) == 2
        assert reqs[0]["custom_id"] == "r1"
        assert reqs[1]["custom_id"] == "r2"
        assert reqs[0]["params"]["model"] == DEFAULT_GENERATOR_MODEL
        assert reqs[0]["params"]["tool_choice"] == {"type": "tool", "name": "test_tool"}

    def test_premium_tier_in_batch(self):
        client = AnthropicClient(enable_caching=False)
        fake_create = _patch_messages_batches(
            client,
            results=[],
            statuses=["ended"],
        )
        client.submit_batch(
            [{"custom_id": "r1", "user_message": "x"}],
            system="x",
            tool_name="t",
            tool_input_schema=SCHEMA,
            tier=TIER_PREMIUM,
        )
        reqs = fake_create.call_args.kwargs["requests"]
        assert reqs[0]["params"]["model"] == DEFAULT_PREMIUM_MODEL

    def test_invalid_tier(self):
        client = AnthropicClient(enable_caching=False)
        with pytest.raises(ValueError, match="Invalid tier"):
            client.submit_batch(
                [{"custom_id": "r1", "user_message": "x"}],
                system="x",
                tool_name="t",
                tool_input_schema=SCHEMA,
                tier="ultra",
            )

    def test_missing_custom_id_raises(self):
        client = AnthropicClient(enable_caching=False)
        with pytest.raises(ValueError, match="custom_id"):
            client.submit_batch(
                [{"user_message": "x"}],  # missing custom_id
                system="x",
                tool_name="t",
                tool_input_schema=SCHEMA,
            )


class TestPollBatch:
    def test_poll_to_completion(self):
        client = AnthropicClient(enable_caching=False)
        _patch_messages_batches(
            client,
            results=[
                _FakeBatchResultItem("r1", "succeeded", {"answer": "a"}),
                _FakeBatchResultItem("r2", "succeeded", {"answer": "b"}),
            ],
            statuses=["ended"],  # First poll returns ended
        )
        out = client.poll_batch("batch_test_123", poll_interval_s=0)
        assert out["r1"] == {"answer": "a"}
        assert out["r2"] == {"answer": "b"}

    def test_poll_with_intermediate_status(self):
        client = AnthropicClient(enable_caching=False)
        _patch_messages_batches(
            client,
            results=[_FakeBatchResultItem("r1", "succeeded", {"answer": "a"})],
            statuses=["in_progress", "in_progress", "ended"],
        )
        out = client.poll_batch("batch_test_123", poll_interval_s=0)
        assert out == {"r1": {"answer": "a"}}

    def test_errored_item_returns_error_dict(self):
        client = AnthropicClient(enable_caching=False)
        _patch_messages_batches(
            client,
            results=[
                _FakeBatchResultItem("r1", "succeeded", {"answer": "a"}),
                _FakeBatchResultItem("r2", "errored", error="some_api_error"),
            ],
            statuses=["ended"],
        )
        out = client.poll_batch("batch_test_123", poll_interval_s=0)
        assert out["r1"] == {"answer": "a"}
        assert "_error" in out["r2"]
        assert out["r2"]["_error"] == "some_api_error"

    def test_telemetry_updated_on_success(self):
        client = AnthropicClient(enable_caching=False)
        _patch_messages_batches(
            client,
            results=[
                _FakeBatchResultItem("r1", "succeeded", {"answer": "a"}),
                _FakeBatchResultItem("r2", "succeeded", {"answer": "b"}),
            ],
            statuses=["ended"],
        )
        client.poll_batch("batch_test_123", poll_interval_s=0)
        stats = client.cache_stats
        assert stats["calls"] == 2
        # Each fake message has 100 input + 20 output tokens
        assert stats["uncached_input_tokens"] == 200


class TestCallToolBatch:
    def test_small_batch_falls_through_to_sync(self):
        """When rows < sync_threshold, we use call_tool per row."""
        client = AnthropicClient(enable_caching=False)
        # Patch the sync path
        client.call_tool = MagicMock(return_value={"answer": "sync"})
        client.submit_batch = MagicMock()  # should NOT be called

        rows = [{"id": f"r{i}", "user_message": f"hi {i}"} for i in range(5)]
        out = client.call_tool_batch(
            rows,
            system="x",
            tool_name="t",
            tool_input_schema=SCHEMA,
            sync_threshold=100,
        )
        assert len(out) == 5
        assert all(v == {"answer": "sync"} for v in out.values())
        client.submit_batch.assert_not_called()
        assert client.call_tool.call_count == 5

    def test_large_batch_uses_batch_api(self):
        """When rows >= sync_threshold, we go through submit_batch + poll_batch."""
        client = AnthropicClient(enable_caching=False)
        # Patch the batch path
        client.submit_batch = MagicMock(return_value="batch_xyz")
        client.poll_batch = MagicMock(
            return_value={f"r{i}": {"answer": "batched"} for i in range(150)}
        )
        client.call_tool = MagicMock()  # should NOT be called

        rows = [{"id": f"r{i}", "user_message": f"hi {i}"} for i in range(150)]
        out = client.call_tool_batch(
            rows,
            system="x",
            tool_name="t",
            tool_input_schema=SCHEMA,
            sync_threshold=100,
        )
        assert len(out) == 150
        assert all(v == {"answer": "batched"} for v in out.values())
        client.call_tool.assert_not_called()
        client.submit_batch.assert_called_once()

    def test_empty_rows_returns_empty(self):
        client = AnthropicClient(enable_caching=False)
        out = client.call_tool_batch(
            [],
            system="x",
            tool_name="t",
            tool_input_schema=SCHEMA,
        )
        assert out == {}

    def test_sync_path_swallows_exceptions(self):
        client = AnthropicClient(enable_caching=False)

        def call_side_effect(**kwargs):
            cid = kwargs.get("provenance_key")
            if cid == "r1":
                return {"answer": "ok"}
            raise RuntimeError("boom")

        client.call_tool = MagicMock(side_effect=call_side_effect)

        rows = [
            {"id": "r1", "user_message": "x"},
            {"id": "r2", "user_message": "y"},
        ]
        out = client.call_tool_batch(
            rows,
            system="x",
            tool_name="t",
            tool_input_schema=SCHEMA,
            sync_threshold=100,
        )
        assert out["r1"] == {"answer": "ok"}
        assert "_error" in out["r2"]
        assert out["r2"]["_error"] == "RuntimeError"
