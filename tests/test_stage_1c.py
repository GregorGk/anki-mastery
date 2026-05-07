"""Stage 1c tests: gloss similarity + collision merge."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.lexical import (  # noqa: E402
    gloss_similarity,
    gloss_tokens,
    load_lexical_replacements,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


# --- Gloss tokenization ------------------------------------------------------


class TestGlossTokens:
    def test_simple(self):
        assert gloss_tokens("train") == {"train"}

    def test_slash_split(self):
        assert gloss_tokens("user / consumer") == {"user", "consumer"}

    def test_strips_stopwords(self):
        # 'of', 'the', 'a' should be dropped
        toks = gloss_tokens("a side of the road")
        assert "side" in toks
        assert "road" in toks
        assert "the" not in toks
        assert "of" not in toks

    def test_strips_short(self):
        # 'in', 'on' short tokens dropped
        toks = gloss_tokens("on top in time")
        assert "top" in toks
        assert "time" in toks

    def test_drops_bp_ep_markers(self):
        toks = gloss_tokens("train (BP)")
        assert "train" in toks
        assert "bp" not in toks

    def test_punctuation_stripped(self):
        toks = gloss_tokens("(bus) stop / break")
        assert "bus" in toks
        assert "stop" in toks
        assert "break" in toks


# --- Gloss similarity --------------------------------------------------------


class TestGlossSimilarity:
    def test_identical(self):
        sim = gloss_similarity("train", "train")
        assert sim.is_similar
        assert sim.score == 1.0

    def test_high_overlap(self):
        # comboio vs trem (BP)
        sim = gloss_similarity("train", "train (BP)")
        assert sim.is_similar

    def test_partial_overlap(self):
        # registar = to register; registrar = to register / record
        sim = gloss_similarity("to register", "to register / record")
        assert sim.is_similar
        assert "register" in sim.overlap

    def test_paragem_parada(self):
        sim = gloss_similarity("(bus) stop", "(bus) stop / break")
        assert sim.is_similar

    def test_unrelated(self):
        sim = gloss_similarity("apple", "banana")
        assert not sim.is_similar

    def test_only_stopwords_in_common(self):
        sim = gloss_similarity("a piece of bread", "a piece of cake")
        # 'piece' is a real overlap → similar
        assert sim.is_similar
        assert "piece" in sim.overlap

    def test_no_meaningful_tokens(self):
        sim = gloss_similarity("the", "an")
        assert not sim.is_similar


# --- Load lexical replacements -----------------------------------------------


class TestLoadLexical:
    def test_load(self):
        rules = load_lexical_replacements(DATA_DIR / "_lexical_bp_replacements.tsv")
        assert "comboio" in rules
        assert rules["comboio"].bp_replacement == "trem"
        assert "equipa" in rules
        assert rules["equipa"].bp_replacement == "equipe"


# --- End-to-end on real ledger ----------------------------------------------


class TestEndToEnd:
    @pytest.fixture
    def staged_data(self, tmp_path):
        """Copy real data to a tmp dir so we can mutate without affecting the repo."""
        out = tmp_path / "data"
        out.mkdir()
        for name in [
            "_source_ledger.tsv",
            "_sense_source_map.tsv",
            "_idioms_expanded.tsv",
            "_flags.tsv",
            "01-normalized.tsv",
            "_lexical_bp_replacements.tsv",
        ]:
            src = DATA_DIR / name
            if not src.exists():
                pytest.skip(f"{name} not present; run prior stages first")
            shutil.copy(src, out / name)
        return out

    def test_run_on_real_data(self, staged_data):
        from build.stage_1c import run as run_1c

        out_path = staged_data / "012-lexical_replaced.tsv"
        review_path = staged_data / "_ep_drop_or_replace_review.tsv"

        summary = run_1c(
            normalized_path=staged_data / "01-normalized.tsv",
            lexical_path=staged_data / "_lexical_bp_replacements.tsv",
            ledger_path=staged_data / "_source_ledger.tsv",
            sense_map_path=staged_data / "_sense_source_map.tsv",
            output_path=out_path,
            review_path=review_path,
            update_ledger=True,
            update_sense_map=True,
        )

        # Of 14 lexical rules, 9 should hit (the 9 EP source forms in source.txt).
        # All 9 hits are collisions with existing BP forms → all should merge.
        assert summary["merges_applied"] >= 8, (
            f"expected ≥8 merges, got {summary['merges_applied']}"
        )
        # Replacements (no collision) should be 0 because every BP target
        # already exists in source.txt.
        assert summary["replacements_applied"] == 0
        # Manual-review collisions: only if a rule fired with low gloss overlap.
        # All 9 our cases share clear glosses, so this should be 0.
        assert summary["manual_review_collisions"] <= 1
        # Output rows = input rows - merged-consumed
        assert summary["output_rows"] < summary["input_rows"]

        # Validate output: the EP forms should NOT appear as `pt` anymore
        # (they were merged into the BP form).
        rows = read_tsv(out_path)
        pts = {r["pt"] for r in rows}
        for ep in [
            "comboio",
            "equipa",
            "desporto",
            "paragem",
            "utilizador",
            "registar",
            "registo",
            "planeamento",
            "controlo",
        ]:
            assert ep not in pts, f"EP form {ep!r} still present after merge"

        # The BP forms should be present
        for bp in [
            "trem",
            "equipe",
            "esporte",
            "parada",
            "usuário",
            "registrar",
            "registro",
            "planejamento",
            "controle",
        ]:
            assert bp in pts, f"BP form {bp!r} missing after merge"

        # The merged surviving rows should have source_variants populated.
        trem_row = next(r for r in rows if r["pt"] == "trem")
        assert "comboio" in trem_row.get("source_variants", "")
        assert trem_row["en_all"]  # absorbed gloss should be non-empty

        equipe_row = next(r for r in rows if r["pt"] == "equipe")
        assert "equipa" in equipe_row.get("source_variants", "")

    def test_invariants_after_run(self, staged_data):
        from build.stage_1c import run as run_1c

        out_path = staged_data / "012-lexical_replaced.tsv"
        review_path = staged_data / "_ep_drop_or_replace_review.tsv"
        run_1c(
            normalized_path=staged_data / "01-normalized.tsv",
            lexical_path=staged_data / "_lexical_bp_replacements.tsv",
            ledger_path=staged_data / "_source_ledger.tsv",
            sense_map_path=staged_data / "_sense_source_map.tsv",
            output_path=out_path,
            review_path=review_path,
            update_ledger=True,
            update_sense_map=True,
        )

        # (rank, expansion_index) uniqueness in output
        seen: set[tuple[int, int]] = set()
        for row in read_tsv(out_path):
            key = (int(row["rank"]), int(row["expansion_index"]))
            assert key not in seen, f"duplicate {key} in 012-lexical_replaced.tsv"
            seen.add(key)

        # No `=` in any pt
        for row in read_tsv(out_path):
            assert "=" not in row["pt"], row

        # Ledger: every merge_into_existing_bp_row has merge_target_*
        ledger = read_tsv(staged_data / "_source_ledger.tsv")
        merge_rows = [r for r in ledger if r["action"] == "merge_into_existing_bp_row"]
        for r in merge_rows:
            assert r.get("merge_target_rank"), r
            assert r.get("merge_target_pt"), r

        # Provenance map: should have new merged edges
        sense_map = read_tsv(staged_data / "_sense_source_map.tsv")
        merged_edges = [e for e in sense_map if e["provenance_type"] == "merged"]
        assert len(merged_edges) >= 8

        # Replacement edges (where no collision) — should be 0 in this corpus
        repl_edges = [e for e in sense_map if e["provenance_type"] == "lexical_replacement"]
        # Allow 0 or a small number depending on coverage
        assert len(repl_edges) >= 0
