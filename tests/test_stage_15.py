"""Stage 1.5 tests — bp_status classification.

LLM is mocked so tests are fast and free. A separate end-to-end run is
exercised in the spot-check, not the test suite.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.stage_15 import (  # noqa: E402
    BP_STATUS_TOOL_SCHEMA,
    VALID_BP_STATUS,
    ClassifyResult,
    _load_manual_overrides,
    classify_one,
    run,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


# --- Fake LLM client --------------------------------------------------------


class FakeAnthropicClient:
    """Returns canned responses based on the headword. Used in tests so the
    suite doesn't burn API credits."""

    def __init__(self, *, responses: dict[str, dict] | None = None, default: dict | None = None):
        self.responses = responses or {}
        self.default = default or {
            "bp_status": "standard",
            "confidence": "high",
            "reason": "",
        }
        self.calls: list[dict] = []

    def call_tool(self, **kwargs):
        # Extract the headword from the user_message
        user_msg = kwargs.get("user_message", "")
        pt = ""
        for line in user_msg.splitlines():
            if line.startswith("pt:"):
                pt = line.split(":", 1)[1].strip()
                break
        self.calls.append({"pt": pt, **kwargs})
        return self.responses.get(pt, self.default)


# --- Schema sanity -----------------------------------------------------------


class TestSchema:
    def test_enum_complete(self):
        enum = BP_STATUS_TOOL_SCHEMA["properties"]["bp_status"]["enum"]
        assert set(enum) == VALID_BP_STATUS
        assert "ep_only" in enum

    def test_required_fields(self):
        assert "bp_status" in BP_STATUS_TOOL_SCHEMA["required"]
        assert "confidence" in BP_STATUS_TOOL_SCHEMA["required"]


# --- classify_one with a fake client ----------------------------------------


class TestClassifyOne:
    def test_standard(self):
        client = FakeAnthropicClient(default={"bp_status": "standard", "confidence": "high"})
        r = classify_one(
            {"pt": "casa", "en_all": "house"},
            client=client,
            system_prompt="(test)",
        )
        assert isinstance(r, ClassifyResult)
        assert r.bp_status == "standard"
        assert r.method == "llm"
        assert client.calls and client.calls[0]["pt"] == "casa"

    def test_nsfw(self):
        client = FakeAnthropicClient(
            responses={"rapariga": {"bp_status": "nsfw", "confidence": "high", "reason": "BP slur"}}
        )
        r = classify_one(
            {"pt": "rapariga", "en_all": "young girl / prostitute (BP)"},
            client=client,
            system_prompt="(test)",
        )
        assert r.bp_status == "nsfw"
        assert r.reason == "BP slur"

    def test_invalid_status_falls_back_to_standard(self):
        # Defensive code: if the model returns an out-of-enum value (shouldn't
        # happen with Tool Use but worth testing), we default to standard.
        client = FakeAnthropicClient(default={"bp_status": "garbage", "confidence": "high"})
        r = classify_one(
            {"pt": "x", "en_all": "y"},
            client=client,
            system_prompt="(test)",
        )
        assert r.bp_status == "standard"

    def test_llm_exception_falls_back(self):
        class ExplodingClient:
            def call_tool(self, **_):
                raise RuntimeError("boom")

        r = classify_one(
            {"pt": "x", "en_all": "y"},
            client=ExplodingClient(),
            system_prompt="(test)",
        )
        assert r.bp_status == "standard"
        assert r.method == "fallback_standard"
        assert "RuntimeError" in r.reason


# --- Manual overrides --------------------------------------------------------


class TestManualOverrides:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "_manual_bp_status.tsv"
        f.write_text(
            "source_line_number\tpt\tbp_status\tconfidence\treason\tapproved_by\tapproved_at\tnotes\n",
            encoding="utf-8",
        )
        overrides = _load_manual_overrides(f)
        assert overrides == {}

    def test_valid_override(self, tmp_path):
        f = tmp_path / "_manual_bp_status.tsv"
        write_tsv(
            f,
            [
                {
                    "source_line_number": "42",
                    "pt": "x",
                    "bp_status": "false_friend",
                    "confidence": "high",
                    "reason": "test",
                    "approved_by": "me",
                    "approved_at": "2026-05-08",
                    "notes": "",
                }
            ],
            fieldnames=[
                "source_line_number",
                "pt",
                "bp_status",
                "confidence",
                "reason",
                "approved_by",
                "approved_at",
                "notes",
            ],
        )
        overrides = _load_manual_overrides(f)
        assert 42 in overrides
        assert overrides[42]["bp_status"] == "false_friend"

    def test_invalid_status_skipped(self, tmp_path, capsys):
        f = tmp_path / "_manual_bp_status.tsv"
        write_tsv(
            f,
            [
                {
                    "source_line_number": "1",
                    "pt": "x",
                    "bp_status": "garbage",  # invalid
                    "confidence": "high",
                    "reason": "",
                    "approved_by": "",
                    "approved_at": "",
                    "notes": "",
                }
            ],
            fieldnames=[
                "source_line_number",
                "pt",
                "bp_status",
                "confidence",
                "reason",
                "approved_by",
                "approved_at",
                "notes",
            ],
        )
        overrides = _load_manual_overrides(f)
        assert overrides == {}


# --- End-to-end with a fake client ------------------------------------------


class TestEndToEnd:
    def test_run_with_fake_client(self, tmp_path):
        """Run the full Stage 1.5 pipeline against a small staged copy."""
        if not (DATA_DIR / "012-lexical_replaced.tsv").exists():
            pytest.skip("Stage 1c output missing")

        # Copy real inputs to tmp so the real ledger / outputs are not touched.
        for name in [
            "012-lexical_replaced.tsv",
            "_source_ledger.tsv",
            "_manual_bp_status.tsv",
        ]:
            src = DATA_DIR / name
            if src.exists():
                shutil.copy(src, tmp_path / name)

        # Build a fake client that flags `vosso` as ep_only and everything
        # else as standard.
        fake = FakeAnthropicClient(
            responses={
                "vosso": {
                    "bp_status": "ep_only",
                    "confidence": "high",
                    "reason": "mainly EP / archaic in BP",
                },
                "rapariga": {
                    "bp_status": "nsfw",
                    "confidence": "high",
                    "reason": "BP-specific slur",
                },
                "camisola": {
                    "bp_status": "false_friend",
                    "confidence": "high",
                    "reason": "BP nightgown vs EP sweater",
                },
            }
        )

        summary = run(
            input_path=tmp_path / "012-lexical_replaced.tsv",
            overrides_path=tmp_path / "_manual_bp_status.tsv",
            output_path=tmp_path / "015-bp_status.tsv",
            ep_drops_path=tmp_path / "_ep_drops.tsv",
            ledger_path=tmp_path / "_source_ledger.tsv",
            audit_path=tmp_path / "audit.jsonl",
            client=fake,  # type: ignore[arg-type]
            limit=200,  # don't classify all 4990 rows in the test
            concurrency=4,
            update_ledger=True,
        )

        # Sanity: most of 200 should be 'standard' under our fake.
        assert summary["bp_status_counts"]["standard"] >= 190
        # vosso (rank 3276) is past 200, so won't appear; that's OK
        # rapariga (2124) past 200; camisola (3842) past 200.
        # Within the first 200 rows, none should be ep_only/nsfw/false_friend.
        assert summary["bp_status_counts"]["ep_only"] == 0
        # Output should equal input (no ep_only filtered out)
        assert summary["output_rows"] == summary["input_rows"]

        # Verify the output schema
        out_rows = read_tsv(tmp_path / "015-bp_status.tsv")
        assert all("bp_status" in r and r["bp_status"] in VALID_BP_STATUS for r in out_rows)
        assert all("bp_status_method" in r for r in out_rows)

    def test_ep_only_filtered_to_drops(self, tmp_path):
        """When the LLM returns ep_only, the row is excluded from output and
        appended to _ep_drops.tsv."""
        # Build a tiny synthetic input
        input_path = tmp_path / "input.tsv"
        write_tsv(
            input_path,
            [
                {
                    "source_line_number": "1",
                    "rank": "1",
                    "expansion_index": "0",
                    "source_pt": "fake_ep_word",
                    "pt": "fake_ep_word",
                    "pt_type": "single_word",
                    "gender": "",
                    "pos": "",
                    "en_all": "obscure EP item",
                    "annotation": "",
                    "normalization_action": "none",
                    "bp_replacement": "",
                    "merge_target_rank": "",
                    "merge_target_pt": "",
                    "merged_from_source_lines": "",
                    "source_variants": "",
                    "source_line": "fake_ep_word = obscure EP item",
                },
                {
                    "source_line_number": "2",
                    "rank": "2",
                    "expansion_index": "0",
                    "source_pt": "casa",
                    "pt": "casa",
                    "pt_type": "single_word",
                    "gender": "",
                    "pos": "",
                    "en_all": "house",
                    "annotation": "",
                    "normalization_action": "none",
                    "bp_replacement": "",
                    "merge_target_rank": "",
                    "merge_target_pt": "",
                    "merged_from_source_lines": "",
                    "source_variants": "",
                    "source_line": "casa = house",
                },
            ],
            fieldnames=[
                "source_line_number",
                "rank",
                "expansion_index",
                "source_pt",
                "pt",
                "pt_type",
                "gender",
                "pos",
                "en_all",
                "annotation",
                "normalization_action",
                "bp_replacement",
                "merge_target_rank",
                "merge_target_pt",
                "merged_from_source_lines",
                "source_variants",
                "source_line",
            ],
        )
        overrides = tmp_path / "_manual_bp_status.tsv"
        overrides.write_text(
            "source_line_number\tpt\tbp_status\tconfidence\treason\tapproved_by\tapproved_at\tnotes\n",
            encoding="utf-8",
        )
        # Empty ledger
        ledger = tmp_path / "_source_ledger.tsv"
        write_tsv(
            ledger,
            [
                {
                    "source_line_number": "1",
                    "rank": "1",
                    "source_raw": "fake_ep_word = obscure EP item",
                    "source_pt": "fake_ep_word",
                    "source_en_all": "obscure EP item",
                    "action": "keep",
                    "normalization_action": "none",
                    "bp_replacement": "",
                    "merge_target_rank": "",
                    "merge_target_pt": "",
                    "output_sense_ids": "",
                    "drop_reason": "",
                    "manual_review_status": "not_required",
                    "stage_decided": "1a",
                    "notes": "",
                },
                {
                    "source_line_number": "2",
                    "rank": "2",
                    "source_raw": "casa = house",
                    "source_pt": "casa",
                    "source_en_all": "house",
                    "action": "keep",
                    "normalization_action": "none",
                    "bp_replacement": "",
                    "merge_target_rank": "",
                    "merge_target_pt": "",
                    "output_sense_ids": "",
                    "drop_reason": "",
                    "manual_review_status": "not_required",
                    "stage_decided": "1a",
                    "notes": "",
                },
            ],
            fieldnames=[
                "source_line_number",
                "rank",
                "source_raw",
                "source_pt",
                "source_en_all",
                "action",
                "normalization_action",
                "bp_replacement",
                "merge_target_rank",
                "merge_target_pt",
                "output_sense_ids",
                "drop_reason",
                "manual_review_status",
                "stage_decided",
                "notes",
            ],
        )

        fake = FakeAnthropicClient(
            responses={
                "fake_ep_word": {
                    "bp_status": "ep_only",
                    "confidence": "high",
                    "reason": "no BP currency",
                },
                "casa": {"bp_status": "standard", "confidence": "high"},
            }
        )

        summary = run(
            input_path=input_path,
            overrides_path=overrides,
            output_path=tmp_path / "015-bp_status.tsv",
            ep_drops_path=tmp_path / "_ep_drops.tsv",
            ledger_path=ledger,
            audit_path=tmp_path / "audit.jsonl",
            client=fake,  # type: ignore[arg-type]
            update_ledger=True,
        )

        assert summary["input_rows"] == 2
        assert summary["output_rows"] == 1  # casa kept, fake_ep_word dropped
        assert summary["ep_drops"] == 1
        assert summary["bp_status_counts"]["ep_only"] == 1

        # Ledger row 1 should now have action=drop_ep_only
        ledger_after = read_tsv(ledger)
        row_1 = next(r for r in ledger_after if r["source_line_number"] == "1")
        assert row_1["action"] == "drop_ep_only"
        assert row_1["drop_reason"]
        assert row_1["stage_decided"] == "1.5"

        # _ep_drops.tsv should have the dropped row with drop_stage=1.5
        drops = read_tsv(tmp_path / "_ep_drops.tsv")
        assert len(drops) == 1
        assert drops[0]["pt"] == "fake_ep_word"
        assert drops[0]["drop_stage"] == "1.5"
