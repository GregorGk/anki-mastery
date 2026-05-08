"""Stage 1.8 tests — duplicate detection."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.stage_18 import (  # noqa: E402
    DUP_TYPE_AMBIGUOUS,
    DUP_TYPE_EXACT,
    DUP_TYPE_SAME_HEADWORD_SENSES,
    categorize_cluster,
    run,
)

DATA_DIR = REPO_ROOT / "data"


# --- categorize_cluster unit tests ------------------------------------------


class TestCategorize:
    def test_single_row(self):
        assert categorize_cluster([{"en_all": "house"}]) == "single"

    def test_exact_duplicate(self):
        rows = [
            {"en_all": "house"},
            {"en_all": "house"},
        ]
        assert categorize_cluster(rows) == DUP_TYPE_EXACT

    def test_same_headword_different_senses_high_overlap(self):
        # Two glosses sharing a content word → similar
        rows = [
            {"en_all": "to register"},
            {"en_all": "to register / record"},
        ]
        assert categorize_cluster(rows) == DUP_TYPE_SAME_HEADWORD_SENSES

    def test_ambiguous_no_overlap(self):
        # Genuinely different
        rows = [
            {"en_all": "apple"},
            {"en_all": "banana"},
        ]
        assert categorize_cluster(rows) == DUP_TYPE_AMBIGUOUS

    def test_ambiguous_partial_overlap(self):
        # 3 rows, two pairs overlap, one doesn't → mixed = ambiguous
        rows = [
            {"en_all": "house / home"},
            {"en_all": "house"},
            {"en_all": "completely unrelated thing"},
        ]
        assert categorize_cluster(rows) == DUP_TYPE_AMBIGUOUS


# --- end-to-end tests --------------------------------------------------------


def _make_input_row(line: int, pt: str, en_all: str, **extras) -> dict:
    """Build a minimal row matching the 015-bp_status.tsv schema."""
    base = {
        "source_line_number": str(line),
        "rank": str(line),
        "expansion_index": "0",
        "source_pt": pt,
        "pt": pt,
        "pt_type": "single_word",
        "gender": "",
        "pos": "",
        "en_all": en_all,
        "annotation": "",
        "normalization_action": "none",
        "bp_replacement": "",
        "merge_target_rank": "",
        "merge_target_pt": "",
        "merged_from_source_lines": "",
        "source_variants": "",
        "bp_status": "standard",
        "bp_status_confidence": "high",
        "bp_status_method": "llm",
        "bp_status_reason": "",
        "source_line": f"{pt} = {en_all}",
    }
    base.update(extras)
    return base


class TestEndToEnd:
    def test_no_duplicates(self, tmp_path):
        rows = [
            _make_input_row(1, "casa", "house"),
            _make_input_row(2, "comer", "to eat"),
            _make_input_row(3, "bom", "good"),
        ]
        in_path = tmp_path / "in.tsv"
        write_tsv(in_path, rows, fieldnames=list(rows[0].keys()))

        summary = run(
            input_path=in_path,
            output_path=tmp_path / "out.tsv",
            duplicates_path=tmp_path / "dups.tsv",
        )

        assert summary["input_rows"] == 3
        assert summary["output_rows"] == 3
        assert summary["duplicate_clusters"] == 0
        assert summary["exact_duplicates"] == 0

    def test_exact_duplicate_drops_one(self, tmp_path):
        rows = [
            _make_input_row(1, "x", "house"),
            _make_input_row(2, "x", "house"),  # exact duplicate
            _make_input_row(3, "y", "tree"),
        ]
        in_path = tmp_path / "in.tsv"
        write_tsv(in_path, rows, fieldnames=list(rows[0].keys()))

        summary = run(
            input_path=in_path,
            output_path=tmp_path / "out.tsv",
            duplicates_path=tmp_path / "dups.tsv",
        )

        assert summary["input_rows"] == 3
        assert summary["output_rows"] == 2  # one dropped
        assert summary["exact_duplicates"] == 1
        assert summary["rows_dropped_exact_duplicate"] == 1

        out_rows = read_tsv(tmp_path / "out.tsv")
        # The surviving x row should be tagged kept_exact_duplicate
        x_rows = [r for r in out_rows if r["pt"] == "x"]
        assert len(x_rows) == 1
        assert x_rows[0]["dedupe_action"] == "kept_exact_duplicate"

    def test_same_headword_passes_through(self, tmp_path):
        # Both rows share content word 'register' → similar → pass-through
        rows = [
            _make_input_row(1, "registrar", "to register"),
            _make_input_row(2, "registrar", "to register / record"),
        ]
        in_path = tmp_path / "in.tsv"
        write_tsv(in_path, rows, fieldnames=list(rows[0].keys()))

        summary = run(
            input_path=in_path,
            output_path=tmp_path / "out.tsv",
            duplicates_path=tmp_path / "dups.tsv",
        )

        assert summary["output_rows"] == 2  # both kept
        assert summary["same_headword_clusters"] == 1

        out_rows = read_tsv(tmp_path / "out.tsv")
        assert all(r["dedupe_action"] == "pass_through_for_sense_split" for r in out_rows)

    def test_ambiguous_flagged_manual_review(self, tmp_path):
        rows = [
            _make_input_row(1, "x", "apple"),
            _make_input_row(2, "x", "banana"),  # unrelated gloss
        ]
        in_path = tmp_path / "in.tsv"
        write_tsv(in_path, rows, fieldnames=list(rows[0].keys()))

        summary = run(
            input_path=in_path,
            output_path=tmp_path / "out.tsv",
            duplicates_path=tmp_path / "dups.tsv",
        )

        assert summary["output_rows"] == 2  # both kept; manual review fires
        assert summary["ambiguous_clusters"] == 1
        assert summary["rows_flagged_manual_review"] == 2

        dups = read_tsv(tmp_path / "dups.tsv")
        assert len(dups) == 1
        assert dups[0]["duplicate_type"] == "ambiguous_collision"
        assert dups[0]["suggested_action"] == "manual_review"


@pytest.mark.skipif(
    not (DATA_DIR / "015-bp_status.tsv").exists(),
    reason="Stage 1.5 output not present yet",
)
class TestRealCorpus:
    def test_no_duplicates_on_real_corpus(self, tmp_path):
        """After Stages 1a–1.5, every pt should be unique on this corpus."""
        from build.stage_18 import run

        summary = run(
            input_path=DATA_DIR / "015-bp_status.tsv",
            output_path=tmp_path / "out.tsv",
            duplicates_path=tmp_path / "dups.tsv",
        )
        # We expect 0 duplicate clusters on the 4985-line BP corpus.
        # If this changes (e.g., new entries added), this test should be
        # updated and the duplicates investigated.
        assert summary["duplicate_clusters"] == 0, (
            f"unexpected duplicates: {summary}"
        )
