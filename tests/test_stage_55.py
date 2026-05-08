"""Stage 5.5 tests — single-pass auditor.

LLM mocked; ProgressTracker, override loading, and end-to-end flow tested directly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.stage_55 import (  # noqa: E402
    AUDIT_TOOL_SCHEMA,
    AuditResult,
    ProgressTracker,
    _build_user_message,
    _load_existing_audit,
    _load_overrides,
    _load_prior_fixes,
    _serialize_audit_row,
    run,
)


# --- ProgressTracker --------------------------------------------------------


class TestProgressTracker:
    def test_started_completed_counts(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        t = ProgressTracker(path, total=3)
        t.started("s1")
        t.completed("s1", verdict="pass", defect_count=0)
        t.started("s2")
        t.completed("s2", verdict="regenerate", defect_count=2)
        snap = t.snapshot()
        assert snap["done"] == 2
        assert snap["in_flight"] == 0
        assert snap["verdicts"]["pass"] == 1
        assert snap["verdicts"]["regenerate"] == 1

    def test_in_flight_tracking(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        t = ProgressTracker(path, total=2)
        t.started("s1")
        t.started("s2")
        snap = t.snapshot()
        assert snap["in_flight"] == 2
        t.completed("s1", verdict="pass", defect_count=0)
        snap = t.snapshot()
        assert snap["in_flight"] == 1

    def test_errored(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        t = ProgressTracker(path, total=1)
        t.started("s1")
        t.errored("s1", error_type="APITimeoutError", error_msg="timeout")
        snap = t.snapshot()
        assert snap["errored"] == 1
        assert snap["in_flight"] == 0

    def test_jsonl_emitted(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        t = ProgressTracker(path, total=1)
        t.started("s1")
        t.completed("s1", verdict="pass", defect_count=0)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "started"
        assert json.loads(lines[1])["event"] == "completed"

    def test_stuck_check_fast(self, tmp_path):
        """Stuck-check should not flag a fresh in-flight call."""
        path = tmp_path / "progress.jsonl"
        t = ProgressTracker(path, total=1)
        t.started("s1")
        stuck = t.stuck_check(threshold_sec=10)
        assert stuck == []

    def test_stuck_check_threshold_zero(self, tmp_path):
        """With threshold=0, anything in-flight is flagged."""
        path = tmp_path / "progress.jsonl"
        t = ProgressTracker(path, total=1)
        t.started("s1")
        time.sleep(0.01)
        stuck = t.stuck_check(threshold_sec=0)
        assert len(stuck) == 1
        assert stuck[0][0] == "s1"

    def test_format_summary(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        t = ProgressTracker(path, total=10)
        t.started("s1")
        t.completed("s1", verdict="pass", defect_count=0)
        s = t.format_summary()
        assert "1/10" in s or "1," in s
        assert "P/R/H: 1/0/0" in s


# --- prompt formatting -----------------------------------------------------


class TestUserMessage:
    def test_basic(self):
        row = {
            "pt": "casa",
            "pt_display": "a casa",
            "pt_type": "single_word",
            "gender": "a",
            "pos": "noun",
            "tags": "#noun",
            "bp_status": "standard",
            "en_primary": "house",
            "en_all": "house / home",
            "example_pt": "Minha casa é grande.",
            "example_en": "My house is big.",
            "target_word_used": "casa",
            "ipa_word_final": "kˈazɐ",
            "ipa_example_final": "mˈĩɲɐ kˈazɐ ɛ ɡɾˈɐ̃dʒi",
        }
        msg = _build_user_message(row)
        assert "casa" in msg
        assert "house" in msg
        assert "kˈazɐ" in msg

    def test_includes_prior_fix(self):
        row = {"pt": "x", "en_primary": "x"}
        prior = {
            "failure_axes": "is_bp",
            "failure_reason": "EP word `comboio`",
        }
        msg = _build_user_message(row, prior_fix=prior)
        assert "prior_validator_status" in msg
        assert "is_bp" in msg
        assert "EP word" in msg


# --- override loading ------------------------------------------------------


class TestOverrides:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "manual.tsv"
        write_tsv(
            p,
            [],
            fieldnames=["sense_id", "verdict_override", "reason", "notes"],
        )
        assert _load_overrides(p) == {}

    def test_valid_override(self, tmp_path):
        p = tmp_path / "manual.tsv"
        write_tsv(
            p,
            [
                {
                    "sense_id": "0001.00.01",
                    "verdict_override": "pass",
                    "reason": "ok",
                    "notes": "",
                }
            ],
            fieldnames=["sense_id", "verdict_override", "reason", "notes"],
        )
        ov = _load_overrides(p)
        assert ov["0001.00.01"]["verdict"] == "pass"

    def test_invalid_verdict_skipped(self, tmp_path):
        p = tmp_path / "manual.tsv"
        write_tsv(
            p,
            [
                {
                    "sense_id": "0001.00.01",
                    "verdict_override": "wat",
                    "reason": "",
                    "notes": "",
                }
            ],
            fieldnames=["sense_id", "verdict_override", "reason", "notes"],
        )
        ov = _load_overrides(p)
        assert ov == {}


# --- existing-audit resume support -----------------------------------------


class TestExistingAudit:
    def test_load_partial(self, tmp_path):
        p = tmp_path / "055-audit.tsv"
        write_tsv(
            p,
            [
                {
                    "sense_id": "0001.00.01",
                    "verdict": "pass",
                    "defect_count": "0",
                    "defects_json": "[]",
                    "reason": "clean",
                    "confidence": "high",
                    "regen_attempts": "0",
                    "final_status": "pass_first",
                    "audit_method": "llm",
                }
            ],
            fieldnames=[
                "sense_id",
                "verdict",
                "defect_count",
                "defects_json",
                "reason",
                "confidence",
                "regen_attempts",
                "final_status",
                "audit_method",
            ],
        )
        existing = _load_existing_audit(p)
        assert "0001.00.01" in existing
        assert existing["0001.00.01"]["verdict"] == "pass"


# --- end-to-end with mocked LLM --------------------------------------------


class _MockAnthropicClient:
    def __init__(self, responses: dict[str, dict]):
        self._responses = responses
        self.cache_stats = {"calls": 0, "cache_hit_ratio": 0.0}

    def call_tool(self, *, system, user_message, tool_name, **kwargs):
        for needle, resp in self._responses.items():
            if needle in user_message:
                return dict(resp)
        return {
            "defects": [],
            "verdict": "pass",
            "reason": "mock fallback",
            "confidence": "high",
        }


class TestEndToEnd:
    def _write_input(self, path: Path, rows: list[dict]) -> None:
        # Stage 5 schema (the columns Stage 5.5 reads)
        fieldnames = [
            "sense_id",
            "rank",
            "expansion_index",
            "sense_index",
            "pt",
            "pt_display",
            "pt_type",
            "gender",
            "pos",
            "en_primary",
            "en_all",
            "tags",
            "bp_status",
            "example_pt",
            "example_en",
            "target_word_used",
            "ipa_word_machine",
            "ipa_word_final",
            "ipa_example_machine",
            "ipa_example_final",
            "ipa_source",
            "ipa_confidence",
            "ipa_notes",
        ]
        write_tsv(path, rows, fieldnames=fieldnames)

    def _row(self, sid: str, pt: str, ex_pt: str) -> dict:
        return {
            "sense_id": sid,
            "rank": "1",
            "expansion_index": "0",
            "sense_index": "1",
            "pt": pt,
            "pt_display": pt,
            "pt_type": "single_word",
            "gender": "",
            "pos": "noun",
            "en_primary": "x",
            "en_all": "x",
            "tags": "",
            "bp_status": "standard",
            "example_pt": ex_pt,
            "example_en": "x",
            "target_word_used": pt,
            "ipa_word_machine": "x",
            "ipa_word_final": "x",
            "ipa_example_machine": "x",
            "ipa_example_final": "x",
            "ipa_source": "corrected",
            "ipa_confidence": "high",
            "ipa_notes": "",
        }

    def test_pass_verdict(self, tmp_path):
        input_path = tmp_path / "05-ipa.tsv"
        self._write_input(input_path, [self._row("0001.00.01", "casa", "Minha casa.")])

        mock = _MockAnthropicClient(
            {
                "casa": {
                    "defects": [],
                    "verdict": "pass",
                    "reason": "clean row",
                    "confidence": "high",
                }
            }
        )

        summary = run(
            input_path=input_path,
            prior_fixes_path=tmp_path / "no_fixes.tsv",
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "055-audit.tsv",
            human_review_path=tmp_path / "_jury.tsv",
            audit_path=tmp_path / "audit.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            concurrency=1,
            anthropic_client=mock,
        )

        assert summary["audited_this_run"] == 1
        assert summary["final_status_counts"]["pass_first"] == 1
        out = read_tsv(tmp_path / "055-audit.tsv")
        assert out[0]["verdict"] == "pass"
        assert out[0]["final_status"] == "pass_first"
        assert out[0]["defect_count"] == "0"

    def test_regenerate_routes_to_human_review_when_skipped(self, tmp_path):
        """skip_regeneration=True (default) treats verdict=regenerate as human_review."""
        input_path = tmp_path / "05-ipa.tsv"
        self._write_input(input_path, [self._row("0001.00.01", "casa", "Bad example.")])

        mock = _MockAnthropicClient(
            {
                "casa": {
                    "defects": [
                        {
                            "axis": "example_uses_intended_sense",
                            "severity": "high",
                            "description": "wrong sense",
                        }
                    ],
                    "verdict": "regenerate",
                    "reason": "wrong sense used",
                    "confidence": "high",
                }
            }
        )

        summary = run(
            input_path=input_path,
            prior_fixes_path=tmp_path / "no_fixes.tsv",
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "055-audit.tsv",
            human_review_path=tmp_path / "_jury.tsv",
            audit_path=tmp_path / "audit.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            concurrency=1,
            skip_regeneration=True,
            anthropic_client=mock,
        )

        out = read_tsv(tmp_path / "055-audit.tsv")
        assert out[0]["verdict"] == "regenerate"
        assert out[0]["final_status"] == "human_review"  # routed since skip_regeneration
        # Human review queue populated
        hr = read_tsv(tmp_path / "_jury.tsv")
        assert len(hr) == 1
        assert hr[0]["sense_id"] == "0001.00.01"

    def test_human_review_routed(self, tmp_path):
        input_path = tmp_path / "05-ipa.tsv"
        self._write_input(input_path, [self._row("0002.00.01", "x", "y")])

        mock = _MockAnthropicClient(
            {
                "y": {
                    "defects": [
                        {
                            "axis": "sense_consistency",
                            "severity": "high",
                            "description": "stage 2 error",
                        }
                    ],
                    "verdict": "human_review",
                    "reason": "upstream sense issue",
                    "confidence": "medium",
                }
            }
        )

        summary = run(
            input_path=input_path,
            prior_fixes_path=tmp_path / "no_fixes.tsv",
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "055-audit.tsv",
            human_review_path=tmp_path / "_jury.tsv",
            audit_path=tmp_path / "audit.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            concurrency=1,
            anthropic_client=mock,
        )

        out = read_tsv(tmp_path / "055-audit.tsv")
        assert out[0]["final_status"] == "human_review"
        hr = read_tsv(tmp_path / "_jury.tsv")
        assert len(hr) == 1

    def test_manual_override_skips_llm(self, tmp_path):
        input_path = tmp_path / "05-ipa.tsv"
        self._write_input(input_path, [self._row("0001.00.01", "casa", "Bad.")])

        overrides = tmp_path / "manual.tsv"
        write_tsv(
            overrides,
            [
                {
                    "sense_id": "0001.00.01",
                    "verdict_override": "pass",
                    "reason": "manual decision",
                    "notes": "",
                }
            ],
            fieldnames=["sense_id", "verdict_override", "reason", "notes"],
        )

        class _Raising:
            cache_stats = {"calls": 0, "cache_hit_ratio": 0}
            def call_tool(self, **kwargs):
                raise RuntimeError("LLM should not be called")

        summary = run(
            input_path=input_path,
            prior_fixes_path=tmp_path / "no_fixes.tsv",
            overrides_path=overrides,
            output_path=tmp_path / "055-audit.tsv",
            human_review_path=tmp_path / "_jury.tsv",
            audit_path=tmp_path / "audit.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            concurrency=1,
            anthropic_client=_Raising(),
        )

        assert summary["manual_overrides"] == 1
        out = read_tsv(tmp_path / "055-audit.tsv")
        assert out[0]["verdict"] == "pass"
        assert out[0]["audit_method"] == "manual_override"

    def test_resume_skips_existing(self, tmp_path):
        input_path = tmp_path / "05-ipa.tsv"
        self._write_input(
            input_path,
            [
                self._row("0001.00.01", "casa", "x"),
                self._row("0002.00.01", "rua", "y"),
            ],
        )
        # Pre-populate output with sid 0001.00.01
        write_tsv(
            tmp_path / "055-audit.tsv",
            [
                {
                    "sense_id": "0001.00.01",
                    "verdict": "pass",
                    "defect_count": "0",
                    "defects_json": "[]",
                    "reason": "previously audited",
                    "confidence": "high",
                    "regen_attempts": "0",
                    "final_status": "pass_first",
                    "audit_method": "llm",
                }
            ],
            fieldnames=[
                "sense_id",
                "verdict",
                "defect_count",
                "defects_json",
                "reason",
                "confidence",
                "regen_attempts",
                "final_status",
                "audit_method",
            ],
        )

        # Mock returns pass for everything
        mock = _MockAnthropicClient({})
        summary = run(
            input_path=input_path,
            prior_fixes_path=tmp_path / "no_fixes.tsv",
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "055-audit.tsv",
            human_review_path=tmp_path / "_jury.tsv",
            audit_path=tmp_path / "audit.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            concurrency=1,
            anthropic_client=mock,
        )

        # Only sid 0002 audited this run
        assert summary["audited_this_run"] == 1
        assert summary["resumed_skipped"] == 1
        out = read_tsv(tmp_path / "055-audit.tsv")
        assert len(out) == 2


# --- schema -----------------------------------------------------------------


class TestSchema:
    def test_required(self):
        assert set(AUDIT_TOOL_SCHEMA["required"]) == {
            "defects",
            "verdict",
            "reason",
            "confidence",
        }

    def test_verdict_enum(self):
        enum = AUDIT_TOOL_SCHEMA["properties"]["verdict"]["enum"]
        assert set(enum) == {"pass", "regenerate", "human_review"}

    def test_defect_axis_enum_includes_ipa(self):
        axis_enum = AUDIT_TOOL_SCHEMA["properties"]["defects"]["items"]["properties"]["axis"]["enum"]
        assert "ipa_plausibility" in axis_enum
        assert "sensitive_policy" in axis_enum
