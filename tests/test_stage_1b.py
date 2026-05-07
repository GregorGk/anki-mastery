"""Stage 1b tests against the golden set."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.bp_rules import (  # noqa: E402
    apply_hyphen_rule,
    apply_spelling_rule,
    load_hyphen_rules,
    load_spelling_map,
    normalize_pt,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="module")
def spelling_map():
    return load_spelling_map(DATA_DIR / "_ep_spelling_map.tsv")


@pytest.fixture(scope="module")
def hyphen_rules():
    return load_hyphen_rules(DATA_DIR / "_hyphen_rules.tsv")


# --- Spelling rules ----------------------------------------------------------


class TestSpellingRules:
    def test_load_map(self, spelling_map):
        assert "facto" in spelling_map
        assert spelling_map["facto"].bp_form == "fato"
        assert spelling_map["facto"].rule_type == "ct_drop"

    def test_apply_facto(self, spelling_map):
        assert apply_spelling_rule("facto", spelling_map) == ("fato", "ct_drop")

    def test_apply_contactar(self, spelling_map):
        assert apply_spelling_rule("contactar", spelling_map) == ("contatar", "ct_drop")

    def test_apply_economico(self, spelling_map):
        assert apply_spelling_rule("económico", spelling_map) == ("econômico", "accent")

    def test_no_rule(self, spelling_map):
        # `casa` is BP-native, no rule applies
        assert apply_spelling_rule("casa", spelling_map) == ("casa", None)

    def test_already_bp(self, spelling_map):
        # `fato` is already the BP form; should not double-normalize
        assert apply_spelling_rule("fato", spelling_map) == ("fato", None)


# --- Hyphen rules ------------------------------------------------------------


class TestHyphenRules:
    def test_no_hyphen(self, hyphen_rules):
        assert apply_hyphen_rule("casa", hyphen_rules) == ("casa", None)

    def test_dehyphenate_mao_de_obra(self, hyphen_rules):
        result, rule = apply_hyphen_rule("mão-de-obra", hyphen_rules)
        assert result == "mão de obra"
        assert rule == "acordo_1990_de_obra"

    def test_dehyphenate_dia_a_dia(self, hyphen_rules):
        result, rule = apply_hyphen_rule("dia-a-dia", hyphen_rules)
        assert result == "dia a dia"
        assert rule == "acordo_1990_repeated"

    def test_preserve_weekday(self, hyphen_rules):
        result, rule = apply_hyphen_rule("segunda-feira", hyphen_rules)
        assert result == "segunda-feira"
        assert rule and rule.startswith("preserve:weekday")

    def test_preserve_bem_estar(self, hyphen_rules):
        result, rule = apply_hyphen_rule("bem-estar", hyphen_rules)
        assert result == "bem-estar"
        assert rule and rule.startswith("preserve:")

    def test_preserve_primeiro_ministro(self, hyphen_rules):
        result, rule = apply_hyphen_rule("primeiro-ministro", hyphen_rules)
        assert result == "primeiro-ministro"
        assert rule and rule.startswith("preserve:")

    def test_preserve_geographic(self, hyphen_rules):
        result, rule = apply_hyphen_rule("norte-americano", hyphen_rules)
        assert result == "norte-americano"
        assert rule and rule.startswith("preserve:")

    def test_unknown_hyphen_kept(self, hyphen_rules):
        # No rule matches; conservative default is to keep the hyphen.
        result, rule = apply_hyphen_rule("foo-bar-unknown", hyphen_rules)
        assert result == "foo-bar-unknown"
        assert rule is None


# --- normalize_pt composition ------------------------------------------------


class TestNormalizeComposition:
    def test_spelling_then_hyphen_separate(self, spelling_map, hyphen_rules):
        # facto has no hyphen
        pt, actions = normalize_pt("facto", spelling_map, hyphen_rules)
        assert pt == "fato"
        assert actions == ["spelling"]

    def test_hyphen_only(self, spelling_map, hyphen_rules):
        pt, actions = normalize_pt("dia-a-dia", spelling_map, hyphen_rules)
        assert pt == "dia a dia"
        assert actions == ["hyphen"]

    def test_no_change(self, spelling_map, hyphen_rules):
        pt, actions = normalize_pt("casa", spelling_map, hyphen_rules)
        assert pt == "casa"
        assert actions == []

    def test_preserve_does_not_record_action(self, spelling_map, hyphen_rules):
        # `bem-estar` matches the `preserve` regex but the OUTPUT is unchanged,
        # so we don't add 'hyphen' to the actions list.
        pt, actions = normalize_pt("bem-estar", spelling_map, hyphen_rules)
        assert pt == "bem-estar"
        assert actions == []


# --- End-to-end on actual ledger --------------------------------------------


class TestEndToEnd:
    def test_run_on_real_ledger(self, tmp_path):
        """Run Stage 1b against the committed ledger and verify outputs."""
        from build.stage_1b import run as run_1b

        ledger_path = DATA_DIR / "_source_ledger.tsv"
        if not ledger_path.exists():
            pytest.skip("Ledger not present; run Stage 1a first")

        # Use a tmp ledger to avoid mutating the real one during tests.
        import shutil

        tmp_ledger = tmp_path / "_source_ledger.tsv"
        shutil.copy(ledger_path, tmp_ledger)
        out_path = tmp_path / "01-normalized.tsv"

        summary = run_1b(
            ledger_path=tmp_ledger,
            spelling_map_path=DATA_DIR / "_ep_spelling_map.tsv",
            hyphen_rules_path=DATA_DIR / "_hyphen_rules.tsv",
            output_path=out_path,
            update_ledger=True,
        )

        # Sanity: 4985 originals + 13 idiom expansions = 4998 active rows
        assert summary["active_rows_normalized"] == 4998
        # Spelling: at minimum `contactar` (line 4321 in golden), and a handful more
        assert summary["spelling_rules_applied"] >= 1
        # Hyphen: at minimum `mão-de-obra` and `dia-a-dia`
        assert summary["hyphen_rules_applied"] >= 2
        assert summary["output_rows"] == 4998

        # Validate the output rows
        rows = read_tsv(out_path)
        assert len(rows) == 4998

        # Check specific known cases from the golden set
        by_ln: dict[int, list[dict]] = {}
        for r in rows:
            by_ln.setdefault(int(r["source_line_number"]), []).append(r)

        # 4321: contactar -> contatar
        line_4321 = next(r for r in by_ln[4321] if int(r["expansion_index"]) == 0)
        assert line_4321["pt"] == "contatar"
        assert "spelling" in line_4321["normalization_action"]

        # 4667: mão-de-obra -> mão de obra
        line_4667 = next(r for r in by_ln[4667] if int(r["expansion_index"]) == 0)
        assert line_4667["pt"] == "mão de obra"
        assert "hyphen" in line_4667["normalization_action"]
        assert line_4667["pt_type"] == "space_compound"

        # 4379: dia-a-dia -> dia a dia
        line_4379 = next(r for r in by_ln[4379] if int(r["expansion_index"]) == 0)
        assert line_4379["pt"] == "dia a dia"
        assert "hyphen" in line_4379["normalization_action"]

        # 3760: segunda-feira -> preserved
        line_3760 = next(r for r in by_ln[3760] if int(r["expansion_index"]) == 0)
        assert line_3760["pt"] == "segunda-feira"
        # 'hyphen' should NOT be in normalization_action because preserve doesn't change output
        assert "hyphen" not in line_3760["normalization_action"]
        assert line_3760["pt_type"] == "hyphenated_compound"

        # 3832: bem-estar -> preserved
        line_3832 = next(r for r in by_ln[3832] if int(r["expansion_index"]) == 0)
        assert line_3832["pt"] == "bem-estar"
        assert line_3832["pt_type"] == "hyphenated_compound"

        # 2377: primeiro-ministro -> preserved
        line_2377 = next(r for r in by_ln[2377] if int(r["expansion_index"]) == 0)
        assert line_2377["pt"] == "primeiro-ministro"
        assert line_2377["pt_type"] == "hyphenated_compound"

        # 314: medida (original) and idiom expansion (à medida que)
        rows_314 = by_ln[314]
        original = next(r for r in rows_314 if int(r["expansion_index"]) == 0)
        assert original["pt"] == "medida"
        expansion = next(r for r in rows_314 if int(r["expansion_index"]) == 1)
        assert expansion["pt"] == "à medida que"
        # idiom_expansion should be in normalization_action for the expansion row
        assert "idiom_expansion" in expansion["normalization_action"]

        # 2224: redor produces TWO idiom expansions (em redor, ao redor)
        rows_2224 = by_ln[2224]
        exp_indices = sorted(int(r["expansion_index"]) for r in rows_2224)
        assert exp_indices == [0, 1, 2], f"expected expansion_index 0,1,2 for redor, got {exp_indices}"

    def test_invariant_no_eq_in_pt(self):
        """After Stage 1b, no row in 01-normalized.tsv has '=' in pt."""
        out = DATA_DIR / "01-normalized.tsv"
        if not out.exists():
            pytest.skip("01-normalized.tsv not generated yet")
        for row in read_tsv(out):
            assert "=" not in row["pt"], row

    def test_invariant_rank_non_decreasing(self):
        """Ranks in 01-normalized.tsv are non-decreasing."""
        out = DATA_DIR / "01-normalized.tsv"
        if not out.exists():
            pytest.skip("01-normalized.tsv not generated yet")
        last = -1
        for row in read_tsv(out):
            r = int(row["rank"])
            assert r >= last, f"rank {r} after {last}"
            last = r

    def test_invariant_unique_rank_expansion_index(self):
        """`(rank, expansion_index)` is unique."""
        out = DATA_DIR / "01-normalized.tsv"
        if not out.exists():
            pytest.skip("01-normalized.tsv not generated yet")
        seen: set[tuple[int, int]] = set()
        for row in read_tsv(out):
            key = (int(row["rank"]), int(row["expansion_index"]))
            assert key not in seen, f"duplicate {key}"
            seen.add(key)
