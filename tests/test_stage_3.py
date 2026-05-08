"""Stage 3 tests — enrichment.

LLM mocked; deterministic shortcuts and tag derivation tested directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.enrich import (  # noqa: E402
    FUNCTION_WORD_POS,
    NUMERALS,
    compose_pt_display,
    derive_tags,
    deterministic_gender,
    deterministic_pos,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.stage_3 import (  # noqa: E402
    ENRICH_TOOL_SCHEMA,
    EnrichResult,
    _load_manual_overrides,
    run,
)

DATA_DIR = REPO_ROOT / "data"


# --- deterministic_pos -------------------------------------------------------


class TestDeterministicPos:
    def test_idiom_expansion(self):
        row = {
            "pt": "à medida que",
            "en_primary": "to the degree that",
            "expansion_index": "1",
            "split_category": "idiom_expansion",
        }
        assert deterministic_pos(row) == "idiom"

    def test_function_word_article(self):
        row = {"pt": "o", "en_primary": "the", "expansion_index": "0"}
        assert deterministic_pos(row) == "art"

    def test_function_word_preposition(self):
        row = {"pt": "de", "en_primary": "of", "expansion_index": "0"}
        assert deterministic_pos(row) == "prep"

    def test_function_word_pronoun(self):
        row = {"pt": "lhe", "en_primary": "him", "expansion_index": "0"}
        assert deterministic_pos(row) == "pron"

    def test_numeral(self):
        row = {"pt": "três", "en_primary": "three", "expansion_index": "0"}
        assert deterministic_pos(row) == "num"

    def test_to_pattern_verb(self):
        row = {"pt": "comer", "en_primary": "to eat", "expansion_index": "0"}
        assert deterministic_pos(row) == "verb"

    def test_reflexive_verb(self):
        row = {
            "pt": "lavar-se",
            "en_primary": "wash oneself",
            "annotation": "reflexive",
            "expansion_index": "0",
        }
        assert deterministic_pos(row) == "verb"

    def test_unknown_returns_empty(self):
        # Plain noun with no shortcut applicable
        row = {"pt": "casa", "en_primary": "house", "expansion_index": "0"}
        assert deterministic_pos(row) == ""


# --- deterministic_gender ----------------------------------------------------


class TestDeterministicGender:
    def test_carries_through_set_gender(self):
        row = {"gender": "o"}
        assert deterministic_gender(row) == "o"

    def test_feminine(self):
        row = {"gender": "a"}
        assert deterministic_gender(row) == "a"

    def test_epicene(self):
        row = {"gender": "o/a"}
        assert deterministic_gender(row) == "o/a"

    def test_empty_gender(self):
        row = {"gender": ""}
        assert deterministic_gender(row) == ""

    def test_invalid_gender_treated_as_unset(self):
        # Garbage in gender column is treated as "no shortcut"
        row = {"gender": "x"}
        assert deterministic_gender(row) == ""


# --- compose_pt_display ------------------------------------------------------


class TestPtDisplay:
    def test_masculine_noun(self):
        assert compose_pt_display("caminho", "o", "single_word") == "o caminho"

    def test_feminine_noun(self):
        assert compose_pt_display("casa", "a", "single_word") == "a casa"

    def test_epicene(self):
        assert compose_pt_display("estudante", "o/a", "single_word") == "o/a estudante"

    def test_idiom_no_article(self):
        assert compose_pt_display("à medida que", "", "idiom") == "à medida que"

    def test_space_compound_no_article(self):
        assert compose_pt_display("em diante", "", "space_compound") == "em diante"

    def test_no_gender_no_article(self):
        # Verbs and adjectives stay bare
        assert compose_pt_display("comer", "", "single_word") == "comer"


# --- derive_tags -------------------------------------------------------------


class TestDeriveTags:
    def test_top500(self):
        row = {"rank": "100", "pt_type": "single_word"}
        tags = derive_tags(row, gender="o", pos="noun", cognate_en=False)
        assert "#top500" in tags
        assert "#noun" in tags
        # No cognate, no morphology beyond pt_type implications
        assert "#cognate-en" not in tags

    def test_top1000(self):
        row = {"rank": "750", "pt_type": "single_word"}
        tags = derive_tags(row, gender="", pos="verb", cognate_en=True)
        assert "#top1000" in tags
        assert "#verb" in tags
        assert "#cognate-en" in tags

    def test_top5000(self):
        row = {"rank": "4500", "pt_type": "single_word"}
        tags = derive_tags(row, gender="", pos="adj", cognate_en=False)
        assert "#top5000" in tags
        assert "#adj" in tags

    def test_idiom_morphology(self):
        row = {"rank": "300", "pt_type": "idiom", "expansion_index": "1"}
        tags = derive_tags(row, gender="", pos="idiom", cognate_en=False)
        assert "#idiom" in tags

    def test_hyphenated(self):
        row = {"rank": "2000", "pt_type": "hyphenated_compound"}
        tags = derive_tags(row, gender="", pos="noun", cognate_en=False)
        assert "#hyphenated" in tags

    def test_compound(self):
        row = {"rank": "2000", "pt_type": "space_compound"}
        tags = derive_tags(row, gender="", pos="", cognate_en=False)
        assert "#compound" in tags

    def test_reflexive_tag(self):
        row = {"rank": "100", "pt_type": "single_word", "annotation": "reflexive"}
        tags = derive_tags(row, gender="", pos="verb", cognate_en=False)
        assert "#reflexive" in tags

    def test_gendered_meaning_tag(self):
        row = {
            "rank": "400",
            "pt_type": "single_word",
            "split_category": "forced_gender_split",
        }
        tags = derive_tags(row, gender="o", pos="noun", cognate_en=False)
        assert "#gendered-meaning" in tags

    def test_function_word_tag(self):
        row = {
            "rank": "5",
            "pt_type": "single_word",
            "split_category": "function_word_polysemy",
        }
        tags = derive_tags(row, gender="", pos="conj", cognate_en=False)
        assert "#function-word" in tags

    def test_bp_status_uncommon(self):
        row = {"rank": "3000", "pt_type": "single_word", "bp_status": "uncommon"}
        tags = derive_tags(row, gender="", pos="adv", cognate_en=False)
        assert "#bp-rare" in tags

    def test_bp_status_nsfw(self):
        row = {"rank": "2000", "pt_type": "single_word", "bp_status": "nsfw"}
        tags = derive_tags(row, gender="a", pos="noun", cognate_en=False)
        assert "#nsfw" in tags

    def test_bp_status_false_friend(self):
        row = {"rank": "2000", "pt_type": "single_word", "bp_status": "false_friend"}
        tags = derive_tags(row, gender="a", pos="noun", cognate_en=False)
        assert "#false-friend" in tags

    def test_dedupe(self):
        # If pos="pron" and split_category=function_word_polysemy, both might
        # try to add #pronoun via different paths. Ensure no duplicates.
        row = {
            "rank": "10",
            "pt_type": "single_word",
            "split_category": "function_word_polysemy",
        }
        tags = derive_tags(row, gender="", pos="pron", cognate_en=False)
        assert tags.count("#pronoun") <= 1
        assert tags.count("#function-word") == 1


# --- Tool Use schema sanity --------------------------------------------------


class TestSchema:
    def test_required(self):
        for f in ["gender", "pos", "is_cognate_en", "confidence"]:
            assert f in ENRICH_TOOL_SCHEMA["required"]

    def test_gender_enum(self):
        assert set(ENRICH_TOOL_SCHEMA["properties"]["gender"]["enum"]) == {
            "o",
            "a",
            "o/a",
            "",
        }

    def test_pos_enum_includes_blank(self):
        assert "" in ENRICH_TOOL_SCHEMA["properties"]["pos"]["enum"]


# --- Manual overrides --------------------------------------------------------


class TestManualOverrides:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "_manual_gender.tsv"
        f.write_text(
            "sense_id\tpt\tgender\tpos\tis_cognate_en\tnotes\n",
            encoding="utf-8",
        )
        assert _load_manual_overrides(f) == {}

    def test_valid_override(self, tmp_path):
        f = tmp_path / "_manual_gender.tsv"
        write_tsv(
            f,
            [
                {
                    "sense_id": "0001.00.01",
                    "pt": "o",
                    "gender": "",
                    "pos": "art",
                    "is_cognate_en": "false",
                    "notes": "test",
                }
            ],
            fieldnames=["sense_id", "pt", "gender", "pos", "is_cognate_en", "notes"],
        )
        overrides = _load_manual_overrides(f)
        assert "0001.00.01" in overrides
        assert overrides["0001.00.01"]["pos"] == "art"
        assert overrides["0001.00.01"]["is_cognate_en"] is False

    def test_invalid_gender_skipped(self, tmp_path):
        f = tmp_path / "_manual_gender.tsv"
        write_tsv(
            f,
            [
                {
                    "sense_id": "0099.00.01",
                    "pt": "x",
                    "gender": "garbage",
                    "pos": "",
                    "is_cognate_en": "",
                    "notes": "",
                }
            ],
            fieldnames=["sense_id", "pt", "gender", "pos", "is_cognate_en", "notes"],
        )
        assert _load_manual_overrides(f) == {}


# --- End-to-end with mocked LLM ---------------------------------------------


class FakeAnthropicClient:
    """Returns canned responses keyed by sense_id."""

    def __init__(self, *, responses: dict[str, dict] | None = None, default: dict | None = None):
        self.responses = responses or {}
        self.default = default or {
            "gender": "",
            "pos": "noun",
            "is_cognate_en": False,
            "confidence": "high",
        }
        self.calls: list[dict] = []
        self.cache_stats = {"calls": 0, "cache_hit_ratio": 0.0}

    def call_tool(self, **kwargs):
        sid = kwargs.get("provenance_key", "")
        self.calls.append({"sense_id": sid, **kwargs})
        return self.responses.get(sid, self.default)


class TestEndToEnd:
    def test_run_with_fake_client(self, tmp_path):
        input_rows = [
            # Casa: simple noun
            {
                "sense_id": "0050.00.01",
                "source_line_number": "50",
                "rank": "50",
                "expansion_index": "0",
                "sense_index": "1",
                "source_pt": "casa",
                "pt": "casa",
                "pt_type": "single_word",
                "gender": "",
                "pos": "",
                "en_primary": "house",
                "en_all": "house",
                "annotation": "",
                "normalization_action": "none",
                "bp_status": "standard",
                "bp_status_confidence": "high",
                "split_category": "single_sense_no_split",
                "split_confidence": "high",
                "split_method": "deterministic",
                "split_reason": "",
                "manual_review_required": "no",
                "source_line": "casa = house",
            },
            # Comer: verb (deterministic shortcut "to eat")
            {
                "sense_id": "0100.00.01",
                "source_line_number": "100",
                "rank": "100",
                "expansion_index": "0",
                "sense_index": "1",
                "source_pt": "comer",
                "pt": "comer",
                "pt_type": "single_word",
                "gender": "",
                "pos": "",
                "en_primary": "to eat",
                "en_all": "to eat",
                "annotation": "",
                "normalization_action": "none",
                "bp_status": "standard",
                "bp_status_confidence": "high",
                "split_category": "single_sense_no_split",
                "split_confidence": "high",
                "split_method": "deterministic",
                "split_reason": "",
                "manual_review_required": "no",
                "source_line": "comer = to eat",
            },
            # Capital (M sense): forced gender split row, gender already set
            {
                "sense_id": "0408.00.01",
                "source_line_number": "408",
                "rank": "408",
                "expansion_index": "0",
                "sense_index": "1",
                "source_pt": "capital",
                "pt": "capital",
                "pt_type": "single_word",
                "gender": "o",  # already set
                "pos": "",
                "en_primary": "investment",
                "en_all": "capital (M investment / F city)",
                "annotation": "(M)",
                "normalization_action": "none",
                "bp_status": "standard",
                "bp_status_confidence": "high",
                "split_category": "forced_gender_split",
                "split_confidence": "high",
                "split_method": "llm",
                "split_reason": "",
                "manual_review_required": "no",
                "source_line": "capital = capital (M investment / F city)",
            },
        ]
        in_path = tmp_path / "in.tsv"
        write_tsv(in_path, input_rows, fieldnames=list(input_rows[0].keys()))

        overrides = tmp_path / "_manual_gender.tsv"
        overrides.write_text(
            "sense_id\tpt\tgender\tpos\tis_cognate_en\tnotes\n", encoding="utf-8"
        )

        fake = FakeAnthropicClient(
            responses={
                "0050.00.01": {
                    "gender": "a",
                    "pos": "noun",
                    "is_cognate_en": False,
                    "confidence": "high",
                },
                "0100.00.01": {
                    "gender": "",
                    "pos": "verb",
                    "is_cognate_en": False,
                    "confidence": "high",
                },
                "0408.00.01": {
                    "gender": "o",
                    "pos": "noun",
                    "is_cognate_en": True,  # capital is cognate
                    "confidence": "high",
                },
            }
        )

        summary = run(
            input_path=in_path,
            overrides_path=overrides,
            output_path=tmp_path / "03-enriched.tsv",
            audit_path=tmp_path / "audit.jsonl",
            client=fake,  # type: ignore[arg-type]
            concurrency=2,
        )

        assert summary["input_rows"] == 3
        assert summary["output_rows"] == 3
        assert summary["llm_calls"] == 3
        assert summary["fallback_calls"] == 0
        assert summary["cognate_count"] == 1

        out = read_tsv(tmp_path / "03-enriched.tsv")
        assert len(out) == 3

        # Casa: a casa, noun
        casa = next(r for r in out if r["pt"] == "casa")
        assert casa["gender"] == "a"
        assert casa["pos"] == "noun"
        assert casa["pt_display"] == "a casa"
        assert "#noun" in casa["tags"]
        assert "#top500" in casa["tags"]

        # Comer: bare display, verb tag
        comer = next(r for r in out if r["pt"] == "comer")
        assert comer["gender"] == ""
        assert comer["pos"] == "verb"
        assert comer["pt_display"] == "comer"
        assert "#verb" in comer["tags"]

        # Capital (M): o capital, noun, cognate
        cap = next(r for r in out if r["pt"] == "capital")
        assert cap["gender"] == "o"
        assert cap["pos"] == "noun"
        assert cap["pt_display"] == "o capital"
        assert "#cognate-en" in cap["tags"]
        assert "#gendered-meaning" in cap["tags"]
        assert cap["is_cognate_en"] == "true"

    def test_manual_override_skips_llm(self, tmp_path):
        rows = [
            {
                "sense_id": "9999.00.01",
                "source_line_number": "9999",
                "rank": "9999",
                "expansion_index": "0",
                "sense_index": "1",
                "source_pt": "test",
                "pt": "test",
                "pt_type": "single_word",
                "gender": "",
                "pos": "",
                "en_primary": "test gloss",
                "en_all": "test gloss",
                "annotation": "",
                "normalization_action": "none",
                "bp_status": "standard",
                "bp_status_confidence": "high",
                "split_category": "single_sense_no_split",
                "split_confidence": "high",
                "split_method": "deterministic",
                "split_reason": "",
                "manual_review_required": "no",
                "source_line": "test = test gloss",
            }
        ]
        in_path = tmp_path / "in.tsv"
        write_tsv(in_path, rows, fieldnames=list(rows[0].keys()))

        overrides = tmp_path / "_manual_gender.tsv"
        write_tsv(
            overrides,
            [
                {
                    "sense_id": "9999.00.01",
                    "pt": "test",
                    "gender": "a",
                    "pos": "noun",
                    "is_cognate_en": "true",
                    "notes": "manual",
                }
            ],
            fieldnames=["sense_id", "pt", "gender", "pos", "is_cognate_en", "notes"],
        )

        fake = FakeAnthropicClient(
            default={
                "gender": "WRONG",
                "pos": "WRONG",
                "is_cognate_en": False,
                "confidence": "low",
            }
        )

        summary = run(
            input_path=in_path,
            overrides_path=overrides,
            output_path=tmp_path / "out.tsv",
            audit_path=tmp_path / "audit.jsonl",
            client=fake,  # type: ignore[arg-type]
        )

        assert summary["manual_overrides_used"] == 1
        assert summary["llm_calls"] == 0
        assert len(fake.calls) == 0  # LLM never called

        out = read_tsv(tmp_path / "out.tsv")
        assert out[0]["gender"] == "a"
        assert out[0]["pos"] == "noun"
        assert out[0]["enrich_method"] == "manual_override"
        assert out[0]["pt_display"] == "a test"
