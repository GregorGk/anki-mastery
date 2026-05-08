"""Stage 4 tests — example sentence generation + cross-family validation.

LLMs are mocked. Deterministic checks (token_in_sentence, word_count) and
the example_policy classifier are tested directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.example_gen import (  # noqa: E402
    build_generator_user_message,
    build_validator_user_message,
    example_policy_for_row,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.lib.validate import (  # noqa: E402
    example_within_word_limit,
    token_in_sentence,
    tokenize,
    word_count,
)
from build.stage_4 import (  # noqa: E402
    GENERATOR_TOOL_SCHEMA,
    VALIDATOR_TOOL_SCHEMA,
    GenerationResult,
    ValidationResult,
    _check_deterministic,
    _load_manual_overrides,
    run,
)


# --- token_in_sentence ------------------------------------------------------


class TestTokenInSentence:
    def test_simple_match(self):
        assert token_in_sentence("casa", "Minha casa é grande.")

    def test_substring_not_match(self):
        # 'casa' should NOT match inside 'casamento'
        assert not token_in_sentence("casa", "Casamento bonito.")

    def test_por_not_in_porque(self):
        # The classic substring trap
        assert not token_in_sentence("por", "Eu não sei porque.")

    def test_por_in_por(self):
        assert token_in_sentence("por", "Por que você foi?")

    def test_hyphenated_compound(self):
        assert token_in_sentence(
            "primeiro-ministro", "O primeiro-ministro chegou ontem."
        )

    def test_enclitic_pronoun(self):
        # `dizer-lhe` is one token (hyphen preserved)
        assert token_in_sentence("dizer-lhe", "Vou dizer-lhe agora.")

    def test_case_insensitive(self):
        assert token_in_sentence("CASA", "minha casa")
        assert token_in_sentence("casa", "Minha CASA")

    def test_punctuation_stripped(self):
        assert token_in_sentence("amigo", "Olá, amigo!")
        assert token_in_sentence("amigo", "Você é meu amigo?")

    def test_apostrophe_kept(self):
        # n'água is one token
        assert token_in_sentence("n'água", "Caiu n'água.")

    def test_empty_target(self):
        assert not token_in_sentence("", "Some sentence.")

    def test_conjugated_form_explicit(self):
        # For headword `ir`, target_word_used should be `vou` if that's the form used
        assert token_in_sentence("vou", "Vou ao supermercado.")
        assert not token_in_sentence("ir", "Vou ao supermercado.")  # ir not in sentence

    def test_multiword_idiom(self):
        assert token_in_sentence(
            "à medida que", "À medida que cresço, aprendo mais."
        )
        assert token_in_sentence(
            "em vigor", "A lei está em vigor desde janeiro."
        )
        assert token_in_sentence(
            "no entanto", "Estava cansado; no entanto, fui trabalhar."
        )
        assert token_in_sentence(
            "por cento", "Trinta por cento dos alunos faltaram."
        )

    def test_multiword_partial_no_match(self):
        # 'em vigor' shouldn't match 'em vigência'
        assert not token_in_sentence("em vigor", "Em vigência total agora.")

    def test_multiword_non_contiguous(self):
        # 'à medida que' must be contiguous; 'à mesa que' shouldn't match
        assert not token_in_sentence(
            "à medida que", "À mesa que falamos sobre vida."
        )


class TestWordCount:
    def test_basic(self):
        assert word_count("Minha casa é grande.") == 4

    def test_within_limit(self):
        assert example_within_word_limit("uma duas três", limit=15)
        assert not example_within_word_limit(" ".join(["a"] * 16), limit=15)

    def test_at_limit(self):
        assert example_within_word_limit(" ".join(["a"] * 15), limit=15)


class TestTokenize:
    def test_lowercase(self):
        assert tokenize("Olá Mundo") == ["olá", "mundo"]

    def test_strips_punct(self):
        assert tokenize("Olá, mundo!") == ["olá", "mundo"]

    def test_keeps_hyphen(self):
        assert "primeiro-ministro" in tokenize("O primeiro-ministro falou.")


# --- example_policy ---------------------------------------------------------


class TestExamplePolicy:
    def test_safe_row_empty_policy(self):
        row = {"pt": "casa", "bp_status": "standard"}
        assert example_policy_for_row(row) == ""

    def test_nsfw_sexual(self):
        row = {"pt": "rapariga", "bp_status": "nsfw"}
        policy = example_policy_for_row(row)
        assert "clinical" in policy.lower() or "neutral" in policy.lower()

    def test_nsfw_violence(self):
        row = {"pt": "matar", "bp_status": "nsfw"}
        policy = example_policy_for_row(row)
        # matar -> violence policy via keyword screen + NSFW status
        assert policy
        assert "violence" in policy.lower() or "graphic" in policy.lower()

    def test_false_friend(self):
        row = {"pt": "camisola", "bp_status": "false_friend"}
        policy = example_policy_for_row(row)
        assert "BP" in policy or "context" in policy.lower()

    def test_weapon_keyword(self):
        row = {"pt": "faca", "bp_status": "standard"}
        policy = example_policy_for_row(row)
        # 'bala' is in flags but other weapons get keyword screen
        assert "violence" in policy.lower() or "graphic" in policy.lower()

    def test_drug_keyword(self):
        row = {"pt": "droga", "bp_status": "standard"}
        policy = example_policy_for_row(row)
        assert "clinical" in policy.lower()

    def test_slur_keyword(self):
        row = {"pt": "mulato", "bp_status": "standard"}
        policy = example_policy_for_row(row)
        # Even without nsfw status, keyword screen catches slurs
        assert "scholarly" in policy.lower() or "neutrally" in policy.lower()


# --- prompt formatting ------------------------------------------------------


class TestBuildUserMessage:
    def test_generator_includes_pt(self):
        row = {
            "pt": "casa",
            "pt_display": "a casa",
            "pos": "noun",
            "gender": "a",
            "en_primary": "house",
            "en_all": "house / home",
            "pt_type": "single_word",
            "bp_status": "standard",
            "tags": "#top500 #noun #single-word",
        }
        msg = build_generator_user_message(row)
        assert "casa" in msg
        assert "house" in msg
        assert "example_policy" not in msg  # safe row

    def test_generator_includes_policy_for_sensitive(self):
        row = {
            "pt": "rapariga",
            "pt_display": "a rapariga",
            "pos": "noun",
            "gender": "a",
            "en_primary": "young girl",
            "en_all": "young girl / prostitute (BP)",
            "pt_type": "single_word",
            "bp_status": "nsfw",
            "tags": "#nsfw",
        }
        msg = build_generator_user_message(row)
        assert "example_policy" in msg

    def test_validator_includes_example(self):
        row = {
            "pt": "casa",
            "en_primary": "house",
            "en_all": "house / home",
            "pos": "noun",
            "tags": "#top500",
        }
        msg = build_validator_user_message(
            row,
            example_pt="Minha casa é grande.",
            example_en="My house is big.",
            target_word_used="casa",
        )
        assert "Minha casa é grande." in msg
        assert "My house is big." in msg
        assert "casa" in msg


# --- deterministic checks ---------------------------------------------------


class TestDeterministicCheck:
    def test_pass(self):
        token_match, wc_ok = _check_deterministic(
            "casa", "Minha casa é grande."
        )
        assert token_match
        assert wc_ok

    def test_token_match_fail(self):
        token_match, _ = _check_deterministic("casa", "Sem nada aqui.")
        assert not token_match

    def test_word_count_fail(self):
        long = " ".join(["uma"] * 20)
        _, wc_ok = _check_deterministic("uma", long)
        assert not wc_ok


# --- manual overrides -------------------------------------------------------


class TestManualOverrides:
    def test_load_empty(self, tmp_path):
        p = tmp_path / "empty.tsv"
        write_tsv(
            p,
            [],
            fieldnames=["sense_id", "example_pt", "example_en", "target_word_used"],
        )
        assert _load_manual_overrides(p) == {}

    def test_load_basic(self, tmp_path):
        p = tmp_path / "manual.tsv"
        write_tsv(
            p,
            [
                {
                    "sense_id": "0001.00.01",
                    "example_pt": "Custom PT.",
                    "example_en": "Custom EN.",
                    "target_word_used": "custom",
                }
            ],
            fieldnames=["sense_id", "example_pt", "example_en", "target_word_used"],
        )
        ov = _load_manual_overrides(p)
        assert "0001.00.01" in ov
        assert ov["0001.00.01"]["example_pt"] == "Custom PT."

    def test_load_missing_file(self, tmp_path):
        assert _load_manual_overrides(tmp_path / "nope.tsv") == {}


# --- end-to-end with mocked LLMs -------------------------------------------


class _MockAnthropicClient:
    """Mock that returns canned tool responses keyed on sense_id."""

    def __init__(self, responses: dict[str, dict]):
        self._responses = responses
        self.cache_stats = {"calls": 0, "cache_hit_ratio": 0.0}

    def call_tool(self, *, system, user_message, tool_name, **kwargs):
        # Pick the response by extracting `pt:` line from user_message
        for line in user_message.splitlines():
            if line.startswith("pt:"):
                pt = line.split(":", 1)[1].strip()
                # Match against any registered sense_id whose pt fits
                for sid, resp in self._responses.items():
                    if resp.get("_pt") == pt:
                        return {k: v for k, v in resp.items() if not k.startswith("_")}
        # Fallback: first response
        if self._responses:
            first = next(iter(self._responses.values()))
            return {k: v for k, v in first.items() if not k.startswith("_")}
        return {"example_pt": "", "example_en": "", "target_word_used": ""}


class _MockOpenAIClient:
    """Mock that always returns pass."""

    def __init__(self):
        self.stats = {"calls": 0, "cache_hit_ratio": 0.0}

    def call_tool(self, **kwargs):
        return {
            "uses_intended_sense": True,
            "is_bp": True,
            "is_natural": True,
            "translation_matches": True,
            "fails_neutral_example_policy": False,
            "validation_status": "pass",
            "validation_reason": "",
        }


class TestEndToEnd:
    def _write_input_tsv(self, path: Path, rows: list[dict]) -> None:
        # Match Stage 3 schema
        fieldnames = [
            "sense_id",
            "source_line_number",
            "rank",
            "expansion_index",
            "sense_index",
            "source_pt",
            "pt",
            "pt_display",
            "pt_type",
            "gender",
            "pos",
            "en_primary",
            "en_all",
            "annotation",
            "normalization_action",
            "bp_status",
            "bp_status_confidence",
            "split_category",
            "split_confidence",
            "split_method",
            "is_cognate_en",
            "family_root",
            "enrich_method",
            "enrich_confidence",
            "enrich_reason",
            "tags",
            "source_line",
        ]
        write_tsv(path, rows, fieldnames=fieldnames)

    def test_run_with_mocks(self, tmp_path):
        input_path = tmp_path / "03-enriched.tsv"
        self._write_input_tsv(
            input_path,
            [
                {
                    "sense_id": "0309.00.01",
                    "source_line_number": "309",
                    "rank": "309",
                    "expansion_index": "0",
                    "sense_index": "1",
                    "pt": "amigo",
                    "pt_display": "o amigo",
                    "pt_type": "single_word",
                    "gender": "o",
                    "pos": "noun",
                    "en_primary": "friend",
                    "en_all": "friend",
                    "tags": "#top500 #noun #single-word",
                    "bp_status": "standard",
                    "source_line": "amigo = friend",
                }
            ],
        )

        anthropic_mock = _MockAnthropicClient(
            {
                "0309.00.01": {
                    "_pt": "amigo",
                    "example_pt": "Meu amigo mora em São Paulo.",
                    "example_en": "My friend lives in São Paulo.",
                    "target_word_used": "amigo",
                }
            }
        )
        openai_mock = _MockOpenAIClient()

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "04-examples.tsv",
            fixes_path=tmp_path / "_fixes.tsv",
            generate_audit=tmp_path / "gen.jsonl",
            validate_audit=tmp_path / "val.jsonl",
            concurrency=2,
            anthropic_client=anthropic_mock,
            openai_client=openai_mock,
        )

        assert summary["input_rows"] == 1
        assert summary["output_rows"] == 1
        assert summary["validator_pass"] == 1
        assert summary["fixes_rows"] == 0
        assert summary["deterministic_token_match_fail"] == 0

        out_rows = read_tsv(tmp_path / "04-examples.tsv")
        assert out_rows[0]["example_pt"] == "Meu amigo mora em São Paulo."
        assert out_rows[0]["example_validation_status"] == "pass"
        assert out_rows[0]["example_token_match"] == "pass"

    def test_manual_override_skips_llm(self, tmp_path):
        input_path = tmp_path / "03-enriched.tsv"
        self._write_input_tsv(
            input_path,
            [
                {
                    "sense_id": "0001.00.01",
                    "source_line_number": "1",
                    "rank": "1",
                    "expansion_index": "0",
                    "sense_index": "1",
                    "pt": "o",
                    "pt_display": "o",
                    "pt_type": "single_word",
                    "gender": "",
                    "pos": "art",
                    "en_primary": "the",
                    "en_all": "the",
                    "tags": "#top500 #art",
                    "bp_status": "standard",
                    "source_line": "o = the",
                }
            ],
        )
        overrides_path = tmp_path / "manual.tsv"
        write_tsv(
            overrides_path,
            [
                {
                    "sense_id": "0001.00.01",
                    "example_pt": "O livro está na mesa.",
                    "example_en": "The book is on the table.",
                    "target_word_used": "o",
                }
            ],
            fieldnames=["sense_id", "example_pt", "example_en", "target_word_used"],
        )

        # Use a mock that would FAIL if called (no responses)
        anthropic_mock = _MockAnthropicClient({})
        openai_mock = _MockOpenAIClient()

        summary = run(
            input_path=input_path,
            overrides_path=overrides_path,
            output_path=tmp_path / "04-examples.tsv",
            fixes_path=tmp_path / "_fixes.tsv",
            generate_audit=tmp_path / "gen.jsonl",
            validate_audit=tmp_path / "val.jsonl",
            concurrency=2,
            anthropic_client=anthropic_mock,
            openai_client=openai_mock,
        )

        assert summary["manual_overrides"] == 1
        assert summary["generated_via_llm"] == 0
        out_rows = read_tsv(tmp_path / "04-examples.tsv")
        assert out_rows[0]["example_pt"] == "O livro está na mesa."
        assert out_rows[0]["example_method"] == "manual_override"

    def test_deterministic_token_fail_routes_to_fixes(self, tmp_path):
        input_path = tmp_path / "03-enriched.tsv"
        self._write_input_tsv(
            input_path,
            [
                {
                    "sense_id": "0309.00.01",
                    "source_line_number": "309",
                    "rank": "309",
                    "expansion_index": "0",
                    "sense_index": "1",
                    "pt": "amigo",
                    "pt_display": "o amigo",
                    "pt_type": "single_word",
                    "gender": "o",
                    "pos": "noun",
                    "en_primary": "friend",
                    "en_all": "friend",
                    "tags": "#top500",
                    "bp_status": "standard",
                    "source_line": "amigo = friend",
                }
            ],
        )

        # Generator hallucinates: target_word_used not in example_pt
        anthropic_mock = _MockAnthropicClient(
            {
                "0309.00.01": {
                    "_pt": "amigo",
                    "example_pt": "Eu gosto de música.",  # 'amigo' missing
                    "example_en": "I like music.",
                    "target_word_used": "amigo",
                }
            }
        )
        openai_mock = _MockOpenAIClient()

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "04-examples.tsv",
            fixes_path=tmp_path / "_fixes.tsv",
            generate_audit=tmp_path / "gen.jsonl",
            validate_audit=tmp_path / "val.jsonl",
            concurrency=2,
            anthropic_client=anthropic_mock,
            openai_client=openai_mock,
        )

        assert summary["deterministic_token_match_fail"] == 1
        assert summary["fixes_rows"] == 1
        fixes = read_tsv(tmp_path / "_fixes.tsv")
        assert fixes[0]["sense_id"] == "0309.00.01"
        assert "token_match" in fixes[0]["failure_axes"]


# --- schema ------------------------------------------------------------------


class TestSchemas:
    def test_generator_schema_required_fields(self):
        assert set(GENERATOR_TOOL_SCHEMA["required"]) == {
            "example_pt",
            "example_en",
            "target_word_used",
        }

    def test_validator_schema_status_enum(self):
        statuses = VALIDATOR_TOOL_SCHEMA["properties"]["validation_status"]["enum"]
        assert set(statuses) == {"pass", "fail", "borderline"}
