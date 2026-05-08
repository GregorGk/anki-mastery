"""Stage 2 tests — sense split.

LLM is mocked so tests are fast and free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.sense_split import (  # noqa: E402
    CAT_FORCED_GENDER_SPLIT,
    CAT_FUNCTION_WORD_POLYSEMY,
    CAT_IDIOM_EXPANSION,
    CAT_LEXICAL_POLYSEMY,
    CAT_SINGLE_SENSE_NO_SPLIT,
    PREMIUM_FUNCTION_WORDS,
    categorize,
    has_forced_gender_split,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.stage_2 import (  # noqa: E402
    SENSE_SPLIT_TOOL_SCHEMA,
    SplitResult,
    _make_sense_id,
    run,
)

DATA_DIR = REPO_ROOT / "data"


# --- categorize unit tests --------------------------------------------------


class TestCategorize:
    def test_no_slash_single_sense(self):
        cat = categorize({"pt": "casa", "en_all": "house", "expansion_index": 0})
        assert cat.category == CAT_SINGLE_SENSE_NO_SPLIT
        assert not cat.needs_llm

    def test_idiom_expansion(self):
        cat = categorize({"pt": "à medida que", "en_all": "measure (a m. que = ...)", "expansion_index": 1})
        assert cat.category == CAT_IDIOM_EXPANSION
        assert cat.needs_llm
        assert cat.llm_tier == "default"

    def test_forced_gender_split_inline(self):
        cat = categorize({"pt": "capital", "en_all": "capital (M investment / F city)", "expansion_index": 0})
        assert cat.category == CAT_FORCED_GENDER_SPLIT
        assert cat.needs_llm
        assert cat.llm_tier == "premium"

    def test_forced_gender_split_per_sense(self):
        cat = categorize({"pt": "cabra", "en_all": "(F) goat / (M) coward", "expansion_index": 0})
        assert cat.category == CAT_FORCED_GENDER_SPLIT
        assert cat.llm_tier == "premium"

    def test_function_word(self):
        for fw in ["o", "de", "que", "se", "para"]:
            cat = categorize({"pt": fw, "en_all": "the / it", "expansion_index": 0})
            assert cat.category == CAT_FUNCTION_WORD_POLYSEMY
            assert cat.llm_tier == "premium"

    def test_function_word_no_slash_still_premium(self):
        # `se` could appear with no slash but still polysemous; we route to LLM
        # because it's in the premium list.
        cat = categorize({"pt": "se", "en_all": "if", "expansion_index": 0})
        # No slash -> single_sense (deterministic short-circuit takes priority)
        # ...wait, the order in categorize is:
        # 1) expansion_index > 0 -> idiom
        # 2) forced gender split -> premium
        # 3) function word -> premium  (THIS FIRES BEFORE no-slash check)
        # 4) no slash -> single sense
        # So "se" with no slash goes to function_word_polysemy.
        assert cat.category == CAT_FUNCTION_WORD_POLYSEMY

    def test_lexical_polysemy_default(self):
        cat = categorize({"pt": "ponto", "en_all": "point / dot / period", "expansion_index": 0})
        assert cat.category == CAT_LEXICAL_POLYSEMY
        assert cat.llm_tier == "default"

    def test_premium_word_list_has_basics(self):
        for must_include in ["o", "de", "que", "se", "ser", "ter", "estar"]:
            assert must_include in PREMIUM_FUNCTION_WORDS


# --- has_forced_gender_split unit tests -------------------------------------


class TestForcedGenderSplit:
    def test_inline_pattern_m_first(self):
        assert has_forced_gender_split("capital (M investment / F city)")

    def test_inline_pattern_f_first(self):
        assert has_forced_gender_split("polícia (F police force / M policeman)")

    def test_per_sense_pattern(self):
        assert has_forced_gender_split("goat (F) / guy (M)")
        assert has_forced_gender_split("cure (F) / priest (M)")

    def test_no_gender_split(self):
        assert not has_forced_gender_split("house / home")
        assert not has_forced_gender_split("point / dot / period")

    def test_mina_false_positive_not_triggered(self):
        # The Stage 1a parser already handles M. Gerais with its own guard,
        # but the Stage 2 detector should also avoid false positives.
        assert not has_forced_gender_split("mine (M. Gerais: state in B)")


# --- sense_id format --------------------------------------------------------


class TestSenseId:
    def test_format(self):
        assert _make_sense_id(1, 0, 1) == "0001.00.01"
        assert _make_sense_id(314, 1, 1) == "0314.01.01"
        assert _make_sense_id(9999, 99, 99) == "9999.99.99"

    def test_zero_padding(self):
        assert _make_sense_id(5, 0, 5) == "0005.00.05"


# --- Tool Use schema sanity -------------------------------------------------


class TestToolSchema:
    def test_required_fields(self):
        assert "is_polysemy" in SENSE_SPLIT_TOOL_SCHEMA["required"]
        assert "senses" in SENSE_SPLIT_TOOL_SCHEMA["required"]
        assert "confidence" in SENSE_SPLIT_TOOL_SCHEMA["required"]

    def test_senses_min_one(self):
        assert SENSE_SPLIT_TOOL_SCHEMA["properties"]["senses"]["minItems"] == 1

    def test_confidence_enum(self):
        enum = SENSE_SPLIT_TOOL_SCHEMA["properties"]["confidence"]["enum"]
        assert set(enum) == {"high", "medium", "low"}


# --- Fake LLM client and end-to-end -----------------------------------------


class FakeAnthropicClient:
    """Returns canned responses keyed by `pt`. Used in tests."""

    def __init__(self, *, responses: dict[str, dict] | None = None, default: dict | None = None):
        self.responses = responses or {}
        self.default = default or {
            "is_polysemy": False,
            "senses": [{"en_primary": "default-en", "gender": "", "annotation": ""}],
            "confidence": "high",
        }
        self.calls: list[dict] = []
        self.cache_stats = {"calls": 0, "cache_hit_ratio": 0.0, "calls_by_tier": {}}

    def call_tool(self, **kwargs):
        user_msg = kwargs.get("user_message", "")
        pt = ""
        for line in user_msg.splitlines():
            if line.startswith("pt:"):
                pt = line.split(":", 1)[1].strip()
                break
        self.calls.append({"pt": pt, **kwargs})
        return self.responses.get(pt, self.default)


class TestEndToEnd:
    def test_run_with_fake_client(self, tmp_path):
        # Tiny synthetic input
        rows = [
            {
                "source_line_number": "1",
                "rank": "1",
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
                "bp_status": "standard",
                "bp_status_confidence": "high",
                "bp_status_method": "llm",
                "bp_status_reason": "",
                "source_line": "casa = house",
                "dedupe_action": "",
            },
            {
                "source_line_number": "2",
                "rank": "2",
                "expansion_index": "0",
                "source_pt": "ponto",
                "pt": "ponto",
                "pt_type": "single_word",
                "gender": "",
                "pos": "",
                "en_all": "point / dot / period",
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
                "source_line": "ponto = point / dot / period",
                "dedupe_action": "",
            },
            {
                "source_line_number": "3",
                "rank": "3",
                "expansion_index": "0",
                "source_pt": "capital",
                "pt": "capital",
                "pt_type": "single_word",
                "gender": "",
                "pos": "",
                "en_all": "capital (M investment / F city)",
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
                "source_line": "capital = capital (M investment / F city)",
                "dedupe_action": "",
            },
        ]
        in_path = tmp_path / "in.tsv"
        write_tsv(in_path, rows, fieldnames=list(rows[0].keys()))

        # Empty overrides + minimal ledger + minimal sense_map
        overrides = tmp_path / "_manual_sense_splits.tsv"
        overrides.write_text(
            "source_line_number\tsource_pt\texpansion_index\tsense_index\ten_primary\tgender\tpt_display\tannotation_json\tnotes\n",
            encoding="utf-8",
        )
        ledger = tmp_path / "_source_ledger.tsv"
        write_tsv(
            ledger,
            [
                {
                    "source_line_number": str(r["source_line_number"]),
                    "rank": str(r["rank"]),
                    "source_raw": r["source_line"],
                    "source_pt": r["source_pt"],
                    "source_en_all": r["en_all"],
                    "action": "keep",
                    "normalization_action": "none",
                    "bp_replacement": "",
                    "merge_target_rank": "",
                    "merge_target_pt": "",
                    "output_sense_ids": "",
                    "drop_reason": "",
                    "manual_review_status": "not_required",
                    "stage_decided": "1c",
                    "notes": "",
                }
                for r in rows
            ],
            fieldnames=[
                "source_line_number", "rank", "source_raw", "source_pt", "source_en_all",
                "action", "normalization_action", "bp_replacement", "merge_target_rank",
                "merge_target_pt", "output_sense_ids", "drop_reason", "manual_review_status",
                "stage_decided", "notes",
            ],
        )
        sense_map = tmp_path / "_sense_source_map.tsv"
        sense_map.write_text(
            "sense_id\tsource_line_number\tprovenance_type\tnotes\n",
            encoding="utf-8",
        )

        fake = FakeAnthropicClient(
            responses={
                "ponto": {
                    "is_polysemy": True,
                    "senses": [
                        {"en_primary": "point", "gender": "", "annotation": ""},
                        {"en_primary": "dot", "gender": "", "annotation": ""},
                        {"en_primary": "period", "gender": "", "annotation": ""},
                    ],
                    "confidence": "high",
                },
                "capital": {
                    "is_polysemy": True,
                    "senses": [
                        {"en_primary": "investment", "gender": "o", "annotation": "(M)"},
                        {"en_primary": "city", "gender": "a", "annotation": "(F)"},
                    ],
                    "confidence": "high",
                },
            }
        )

        summary = run(
            input_path=in_path,
            overrides_path=overrides,
            ledger_path=ledger,
            sense_map_path=sense_map,
            output_path=tmp_path / "02-senses.tsv",
            audit_path=tmp_path / "audit.jsonl",
            client=fake,  # type: ignore[arg-type]
            update_ledger=True,
            update_sense_map=True,
            concurrency=2,
        )

        # casa: 1 sense (deterministic, no slash)
        # ponto: 3 senses (LLM)
        # capital: 2 senses (forced gender split, LLM premium)
        assert summary["output_senses"] == 6
        assert summary["polysemy_input_rows"] == 2
        assert summary["deterministic_decisions"] == 1
        assert summary["llm_calls"] == 2

        out_rows = read_tsv(tmp_path / "02-senses.tsv")
        assert len(out_rows) == 6

        # casa
        casa_rows = [r for r in out_rows if r["pt"] == "casa"]
        assert len(casa_rows) == 1
        assert casa_rows[0]["sense_id"] == "0001.00.01"
        assert casa_rows[0]["en_primary"] == "house"
        assert casa_rows[0]["split_method"] == "deterministic"

        # ponto: 3 senses
        ponto_rows = [r for r in out_rows if r["pt"] == "ponto"]
        assert len(ponto_rows) == 3
        assert {r["sense_id"] for r in ponto_rows} == {
            "0002.00.01",
            "0002.00.02",
            "0002.00.03",
        }
        assert {r["en_primary"] for r in ponto_rows} == {"point", "dot", "period"}

        # capital: 2 senses with gender
        cap_rows = [r for r in out_rows if r["pt"] == "capital"]
        assert len(cap_rows) == 2
        cap_sorted = sorted(cap_rows, key=lambda r: int(r["sense_index"]))
        assert cap_sorted[0]["sense_id"] == "0003.00.01"
        assert cap_sorted[0]["gender"] == "o"
        assert cap_sorted[0]["en_primary"] == "investment"
        assert cap_sorted[1]["sense_id"] == "0003.00.02"
        assert cap_sorted[1]["gender"] == "a"
        assert cap_sorted[1]["en_primary"] == "city"

        # Ledger output_sense_ids populated
        ledger_after = read_tsv(ledger)
        assert ledger_after[0]["output_sense_ids"] == "0001.00.01"
        assert ledger_after[1]["output_sense_ids"] == "0002.00.01,0002.00.02,0002.00.03"
        assert ledger_after[2]["output_sense_ids"] == "0003.00.01,0003.00.02"

        # sense_source_map has 6 edges (one per sense)
        edges = read_tsv(sense_map)
        assert len(edges) == 6
        assert all(e["provenance_type"] == "original" for e in edges)

        # Verify capital was routed to premium tier
        cap_call = next(c for c in fake.calls if c["pt"] == "capital")
        assert cap_call["tier"] == "premium"
        ponto_call = next(c for c in fake.calls if c["pt"] == "ponto")
        assert ponto_call["tier"] == "default"

    def test_manual_override_wins(self, tmp_path):
        rows = [
            {
                "source_line_number": "5",
                "rank": "5",
                "expansion_index": "0",
                "source_pt": "ambiguous",
                "pt": "ambiguous",
                "pt_type": "single_word",
                "gender": "",
                "pos": "",
                "en_all": "thing / item",
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
                "source_line": "ambiguous = thing / item",
                "dedupe_action": "",
            },
        ]
        in_path = tmp_path / "in.tsv"
        write_tsv(in_path, rows, fieldnames=list(rows[0].keys()))

        # Manual override forces a single sense with custom en_primary
        overrides = tmp_path / "_manual_sense_splits.tsv"
        write_tsv(
            overrides,
            [
                {
                    "source_line_number": "5",
                    "source_pt": "ambiguous",
                    "expansion_index": "0",
                    "sense_index": "1",
                    "en_primary": "manually-set-thing",
                    "gender": "",
                    "pt_display": "",
                    "annotation_json": "",
                    "notes": "",
                }
            ],
            fieldnames=[
                "source_line_number", "source_pt", "expansion_index", "sense_index",
                "en_primary", "gender", "pt_display", "annotation_json", "notes",
            ],
        )

        ledger = tmp_path / "_source_ledger.tsv"
        write_tsv(
            ledger,
            [
                {
                    "source_line_number": "5",
                    "rank": "5",
                    "source_raw": "ambiguous = thing / item",
                    "source_pt": "ambiguous",
                    "source_en_all": "thing / item",
                    "action": "keep",
                    "normalization_action": "none",
                    "bp_replacement": "",
                    "merge_target_rank": "",
                    "merge_target_pt": "",
                    "output_sense_ids": "",
                    "drop_reason": "",
                    "manual_review_status": "not_required",
                    "stage_decided": "1c",
                    "notes": "",
                }
            ],
            fieldnames=[
                "source_line_number", "rank", "source_raw", "source_pt", "source_en_all",
                "action", "normalization_action", "bp_replacement", "merge_target_rank",
                "merge_target_pt", "output_sense_ids", "drop_reason", "manual_review_status",
                "stage_decided", "notes",
            ],
        )
        sense_map = tmp_path / "_sense_source_map.tsv"
        sense_map.write_text(
            "sense_id\tsource_line_number\tprovenance_type\tnotes\n",
            encoding="utf-8",
        )

        # LLM would normally be called for "thing / item" but override should win
        fake = FakeAnthropicClient(
            default={
                "is_polysemy": True,
                "senses": [
                    {"en_primary": "LLM-WOULD-PICK-THIS", "gender": "", "annotation": ""}
                ],
                "confidence": "high",
            }
        )

        summary = run(
            input_path=in_path,
            overrides_path=overrides,
            ledger_path=ledger,
            sense_map_path=sense_map,
            output_path=tmp_path / "out.tsv",
            audit_path=tmp_path / "audit.jsonl",
            client=fake,  # type: ignore[arg-type]
            concurrency=1,
        )

        assert summary["manual_overrides_used"] == 1
        assert summary["llm_calls"] == 0  # override prevented LLM call

        out = read_tsv(tmp_path / "out.tsv")
        assert len(out) == 1
        assert out[0]["en_primary"] == "manually-set-thing"
        assert out[0]["split_method"] == "manual_override"
