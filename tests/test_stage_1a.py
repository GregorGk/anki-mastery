"""Stage 1a tests against the golden set.

The golden set is at tests/golden_set.tsv with 57 hand-curated rows from
data/source.txt. Each row carries a `category` and `notes` describing what
the parser must do with it.

These tests exercise the parser with --no-llm (deterministic heuristic) so
they are fast, free, and reproducible. A separate optional test runs with the
LLM if ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.parse import (  # noqa: E402
    detect_idiom_candidate,
    detect_markers,
    detect_pt_type,
    extract_parentheticals,
    is_false_positive,
    parse_line,
    split_on_first_equals,
)
from build.lib.tsv import read_tsv  # noqa: E402

GOLDEN_SET = REPO_ROOT / "tests" / "golden_set.tsv"
SOURCE_FILE = REPO_ROOT / "data" / "source.txt"


@pytest.fixture(scope="module")
def golden_rows() -> list[dict]:
    return read_tsv(GOLDEN_SET)


@pytest.fixture(scope="module")
def source_lines() -> dict[int, str]:
    """Return {line_number: line_text} for every line in source.txt."""
    if not SOURCE_FILE.exists():
        pytest.skip("data/source.txt not present")
    out: dict[int, str] = {}
    with SOURCE_FILE.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            out[n] = line.rstrip("\n")
    return out


# --- Pure-function unit tests -----------------------------------------------


class TestSplitOnFirstEquals:
    def test_simple(self):
        assert split_on_first_equals("o = the / it") == ("o", "the / it")

    def test_embedded_eq_eu(self):
        # source.txt line 31
        result = split_on_first_equals("eu = I (OBJ = me)")
        assert result == ("eu", "I (OBJ = me)")

    def test_embedded_eq_medida(self):
        # source.txt line 314
        result = split_on_first_equals(
            "medida = measure (a m. que = to the degree / extent that)"
        )
        assert result == (
            "medida",
            "measure (a m. que = to the degree / extent that)",
        )

    def test_embedded_eq_vos(self):
        # source.txt line 3372
        result = split_on_first_equals("vós = you (PL) (OBJ = vos)")
        assert result == ("vós", "you (PL) (OBJ = vos)")

    def test_no_equals(self):
        assert split_on_first_equals("just a string") is None

    def test_empty(self):
        assert split_on_first_equals("") is None
        assert split_on_first_equals("   ") is None

    def test_trailing_newline(self):
        assert split_on_first_equals("o = the\n") == ("o", "the")


class TestExtractParentheticals:
    def test_single(self):
        assert extract_parentheticals("foo (bar)") == ["bar"]

    def test_multiple(self):
        assert extract_parentheticals("a (one) b (two)") == ["one", "two"]

    def test_nested_kept_as_one(self):
        # We track depth and only emit at depth-0 close.
        assert extract_parentheticals("a (outer (inner) more)") == ["outer (inner) more"]

    def test_unbalanced(self):
        # Unbalanced parens shouldn't crash; just emit nothing or partial.
        result = extract_parentheticals("a (unclosed")
        assert isinstance(result, list)


class TestFalsePositives:
    def test_m_gerais(self):
        assert is_false_positive("M. Gerais: state in B")

    def test_obj(self):
        assert is_false_positive("OBJ = me")
        assert is_false_positive("OBJ = vos")

    def test_eg(self):
        assert is_false_positive("e.g. some example")

    def test_real_idiom_not_false_positive(self):
        assert not is_false_positive("em d.")
        assert not is_false_positive("a m. que = to the degree")


class TestIdiomCandidate:
    def test_simple(self):
        assert detect_idiom_candidate("em d.", "diante") == "em d."

    def test_with_que(self):
        assert detect_idiom_candidate("a m. que = to the degree", "medida") == (
            "a m. que = to the degree"
        )

    def test_two_alternatives(self):
        assert detect_idiom_candidate("em / ao r.", "redor") == "em / ao r."

    def test_initial_mismatch(self):
        # `(em s.)` against `vigor` would not be a candidate
        assert detect_idiom_candidate("em s.", "vigor") is None

    def test_false_positive_skipped(self):
        assert detect_idiom_candidate("M. Gerais: state in B", "mina") is None
        assert detect_idiom_candidate("OBJ = me", "eu") is None


class TestPtType:
    def test_single_word(self):
        assert detect_pt_type("casa") == "single_word"

    def test_hyphenated(self):
        assert detect_pt_type("primeiro-ministro") == "hyphenated_compound"
        assert detect_pt_type("segunda-feira") == "hyphenated_compound"

    def test_space_compound(self):
        assert detect_pt_type("em diante") == "space_compound"


class TestMarkers:
    def test_bp_only(self):
        m = detect_markers("train (BP)")
        assert m["has_bp_marker"] is True
        assert m["has_ep_marker"] is False

    def test_ep_only(self):
        m = detect_markers("nightgown (BP) / sweater (EP)")
        assert m["has_bp_marker"] is True
        assert m["has_ep_marker"] is True

    def test_mainly_ep(self):
        m = detect_markers("your (PL / mainly EP)")
        assert m["has_mainly_ep_marker"] is True

    def test_reflexive_se(self):
        m = detect_markers("to specialize in (+se)")
        assert m["has_reflexive_se"] is True

    def test_gender_split_m_f(self):
        m = detect_markers("capital (M investment / F city)")
        assert m["has_gender_split"] is True

    def test_gender_split_per_sense(self):
        # cabra = goat (F) / guy (M)
        m = detect_markers("goat (F) / guy (M)")
        assert m["has_gender_split"] is True

    def test_mina_not_gender_split(self):
        # M. Gerais must not trigger gender split
        m = detect_markers("mine (M. Gerais: state in B)")
        assert m["has_gender_split"] is False


# --- Golden-set behavioural tests -------------------------------------------


class TestGoldenSet:
    def test_golden_loads(self, golden_rows):
        assert len(golden_rows) >= 50, "Golden set should have ~57 rows"

    def test_all_golden_rows_match_source(self, golden_rows, source_lines):
        """Every golden row's source_raw must equal source.txt at that line number."""
        for row in golden_rows:
            ln = int(row["source_line_number"])
            assert (
                source_lines[ln] == row["source_raw"]
            ), f"Line {ln} mismatch: golden={row['source_raw']!r}, source={source_lines[ln]!r}"

    def test_embedded_eq_rows_parse(self, golden_rows):
        """The 3 embedded-= rows must parse without losing the RHS '='."""
        embedded = [r for r in golden_rows if r["category"] == "embedded_eq"]
        assert len(embedded) >= 2, "Expect at least 2 embedded_eq rows (eu, vós)"
        for r in embedded:
            ln = int(r["source_line_number"])
            parsed = parse_line(ln, r["source_raw"])
            assert parsed is not None
            assert "=" in parsed.en_all  # the RHS still contains '='
            assert "=" not in parsed.pt  # the headword does not

    def test_embedded_eq_idiom_medida(self, golden_rows):
        """Line 314 (medida) is BOTH embedded-= AND idiom candidate."""
        row = next(r for r in golden_rows if r["category"] == "embedded_eq_idiom")
        parsed = parse_line(int(row["source_line_number"]), row["source_raw"])
        assert parsed is not None
        assert parsed.pt == "medida"
        assert "a m. que = to the degree / extent that" in parsed.en_all
        assert any("a m. que" in c for c in parsed.idiom_candidates)

    def test_idiom_candidates_detected(self, golden_rows):
        """Every idiom_expansion row must produce ≥1 idiom candidate."""
        idiom_rows = [r for r in golden_rows if r["category"] == "idiom_expansion"]
        assert len(idiom_rows) >= 8
        for r in idiom_rows:
            parsed = parse_line(int(r["source_line_number"]), r["source_raw"])
            assert parsed is not None, f"Line {r['source_line_number']} failed to parse"
            assert (
                parsed.idiom_candidates
            ), f"Line {r['source_line_number']} ({parsed.pt}): no idiom candidate detected"

    def test_mina_false_positive(self, golden_rows):
        """`mina (M. Gerais: state in B)` must NOT be flagged as idiom or gender split."""
        row = next(r for r in golden_rows if r["category"] == "mina_false_positive")
        parsed = parse_line(int(row["source_line_number"]), row["source_raw"])
        assert parsed is not None
        assert parsed.pt == "mina"
        assert (
            not parsed.idiom_candidates
        ), f"mina falsely got idiom candidates: {parsed.idiom_candidates}"
        assert not parsed.has_gender_split, "mina falsely got gender split flag"

    def test_forced_gender_splits(self, golden_rows):
        """All forced_gender_split rows have has_gender_split=True."""
        gender_rows = [r for r in golden_rows if r["category"] == "forced_gender_split"]
        assert len(gender_rows) == 8
        for r in gender_rows:
            parsed = parse_line(int(r["source_line_number"]), r["source_raw"])
            assert parsed is not None
            assert (
                parsed.has_gender_split
            ), f"{parsed.pt}: gender split not detected in {parsed.en_all!r}"

    def test_bp_markers(self, golden_rows):
        """All bp_marker / false_friend rows have has_bp_marker=True."""
        bp_rows = [
            r
            for r in golden_rows
            if r["category"] in {"bp_marker", "bp_marker_polysemy", "false_friend", "false_friend_nsfw"}
        ]
        assert len(bp_rows) >= 5
        for r in bp_rows:
            parsed = parse_line(int(r["source_line_number"]), r["source_raw"])
            assert parsed is not None
            assert parsed.has_bp_marker, f"{parsed.pt}: (BP) marker not detected"

    def test_mainly_ep_marker(self, golden_rows):
        """`vosso (PL / mainly EP)` triggers has_mainly_ep_marker."""
        rows = [r for r in golden_rows if r["category"] == "mainly_ep_marker"]
        assert len(rows) >= 1
        for r in rows:
            parsed = parse_line(int(r["source_line_number"]), r["source_raw"])
            assert parsed is not None
            assert parsed.has_mainly_ep_marker

    def test_reflexive_se(self, golden_rows):
        rows = [r for r in golden_rows if r["category"] == "reflexive"]
        assert len(rows) >= 2
        for r in rows:
            parsed = parse_line(int(r["source_line_number"]), r["source_raw"])
            assert parsed is not None
            assert parsed.has_reflexive_se, f"{parsed.pt}: (+se) not detected"

    def test_pt_type_classification(self, golden_rows):
        """Hyphenated headwords classify as hyphenated_compound."""
        for r in golden_rows:
            if r["category"] in {"hyphenated_compound", "hyphenated_weekday", "hyphenated_deprecated"}:
                parsed = parse_line(int(r["source_line_number"]), r["source_raw"])
                assert parsed is not None
                assert parsed.pt_type == "hyphenated_compound", (
                    f"{parsed.pt}: expected hyphenated_compound, got {parsed.pt_type}"
                )


# --- End-to-end run on golden set (no LLM) ----------------------------------


class TestRunOnGoldenSubset:
    """Run the full Stage 1a pipeline on a temp source containing only golden lines."""

    def test_run_no_llm_subset(self, tmp_path, golden_rows):
        """Build a source.txt subset from the golden set and run Stage 1a."""
        # Sort by source_line_number so line numbers are sane.
        sorted_rows = sorted(golden_rows, key=lambda r: int(r["source_line_number"]))
        # Build a synthetic source.txt where each line preserves the right rank
        # by padding with empty strings up to its line number.
        max_ln = max(int(r["source_line_number"]) for r in sorted_rows)
        lines = [""] * max_ln
        for r in sorted_rows:
            lines[int(r["source_line_number"]) - 1] = r["source_raw"]

        synthetic = tmp_path / "source.txt"
        synthetic.write_text("\n".join(lines) + "\n", encoding="utf-8")

        from build.stage_1a import run as run_1a  # noqa: E402

        ledger = tmp_path / "_source_ledger.tsv"
        sense_map = tmp_path / "_sense_source_map.tsv"
        idioms = tmp_path / "_idioms_expanded.tsv"
        flags = tmp_path / "_flags.tsv"
        audit = tmp_path / "01a_parse.jsonl"

        summary = run_1a(
            source_path=synthetic,
            ledger_out=ledger,
            sense_map_out=sense_map,
            idioms_out=idioms,
            flags_out=flags,
            audit_out=audit,
            use_llm=False,
        )
        # Sanity checks on the summary
        assert summary["source_lines_processed"] == max_ln
        assert summary["ledger_original_rows"] == max_ln
        assert summary["idiom_candidates_resolved"] >= 8  # at least our golden idioms
        assert summary["flags_logged"] >= 5  # BP/EP/mainly-EP flags
        assert summary["parse_failures"] == 0

        # Validate ledger content
        ledger_rows = read_tsv(ledger)
        assert len(ledger_rows) >= max_ln
        # Every kept row's source_pt has no '='
        for row in ledger_rows:
            if row["action"] not in {"manual_review", "drop_ep_only"}:
                assert "=" not in row["source_pt"], row

        # Validate flags content
        flag_rows = read_tsv(flags)
        assert any(f["flag_type"] == "bp_marker" for f in flag_rows)
        assert any(f["flag_type"] == "ep_marker" for f in flag_rows)

        # Validate idiom expansion happened (heuristic)
        idiom_rows = read_tsv(idioms)
        assert all(r["method"] in {"heuristic", "llm"} for r in idiom_rows)
