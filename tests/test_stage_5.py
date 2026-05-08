"""Stage 5 tests — IPA generation (eSpeak baseline + LLM correction).

LLM is mocked. eSpeak is exercised via lib/ipa.py with a guard for
environments where espeak-ng is missing (tests skip in that case).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.ipa import (  # noqa: E402
    EspeakNotFound,
    token_count,
    transcribe_tokens,
    transcribe_word,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.stage_5 import (  # noqa: E402
    CORRECT_IPA_TOOL_SCHEMA,
    IPAResult,
    _build_user_message,
    _load_overrides,
    run,
)

ESPEAK_AVAILABLE = shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None
needs_espeak = pytest.mark.skipif(
    not ESPEAK_AVAILABLE, reason="espeak-ng not installed"
)


# --- eSpeak wrapper ---------------------------------------------------------


@needs_espeak
class TestTranscribeWord:
    def test_simple_word(self):
        ipa = transcribe_word("casa")
        assert ipa
        # Stress mark expected
        assert "ˈ" in ipa

    def test_empty_returns_empty(self):
        assert transcribe_word("") == ""
        assert transcribe_word("   ") == ""

    def test_idiom_phrase(self):
        ipa = transcribe_word("à medida que")
        assert ipa
        # eSpeak gives a single transcription for the whole phrase
        assert "ˈ" in ipa


@needs_espeak
class TestTranscribeTokens:
    def test_isolated_form_token_count_matches(self):
        sentence = "Minha casa é grande."
        ipa = transcribe_tokens(sentence)
        # 4 tokens: minha / casa / é / grande
        assert token_count(ipa) == 4

    def test_hyphenated_compound_one_token(self):
        sentence = "O primeiro-ministro chegou."
        # Tokenize keeps hyphens, so 4 tokens: o / primeiro-ministro / chegou
        ipa = transcribe_tokens(sentence)
        assert token_count(ipa) == 3

    def test_empty_sentence(self):
        assert transcribe_tokens("") == ""


def test_token_count_helper():
    assert token_count("a b c") == 3
    assert token_count("") == 0
    assert token_count("   ") == 0


def test_espeak_not_found_message(monkeypatch):
    """When espeak is genuinely missing, EspeakNotFound is raised with helpful text."""
    monkeypatch.setattr("build.lib.ipa.ESPEAK_BIN", None)
    with pytest.raises(EspeakNotFound, match="brew install"):
        transcribe_word("casa")


# --- prompt formatting ------------------------------------------------------


class TestUserMessage:
    def test_includes_required_fields(self):
        row = {"pt": "casa", "example_pt": "Minha casa é grande."}
        msg = _build_user_message(row, "kˈazæ", "mˌiɲæ kˈazæ ɛ ɡrˈɐ̃ŋdʒy")
        assert "casa" in msg
        assert "Minha casa é grande." in msg
        assert "kˈazæ" in msg


# --- manual overrides -------------------------------------------------------


class TestManualOverrides:
    def test_load_empty(self, tmp_path):
        p = tmp_path / "manual.tsv"
        write_tsv(
            p,
            [],
            fieldnames=[
                "sense_id",
                "ipa_word_override",
                "ipa_example_override",
                "notes",
            ],
        )
        assert _load_overrides(p) == {}

    def test_load_word_only(self, tmp_path):
        p = tmp_path / "manual.tsv"
        write_tsv(
            p,
            [
                {
                    "sense_id": "0001.00.01",
                    "ipa_word_override": "kˈazɐ",
                    "ipa_example_override": "",
                    "notes": "test",
                }
            ],
            fieldnames=[
                "sense_id",
                "ipa_word_override",
                "ipa_example_override",
                "notes",
            ],
        )
        ov = _load_overrides(p)
        assert "0001.00.01" in ov
        assert ov["0001.00.01"]["ipa_word"] == "kˈazɐ"


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
            "ipa_word_final": "ipa-fallback",
            "ipa_example_final": "ipa-ex-fallback",
            "confidence": "low",
            "notes": "mock fallback",
        }


@needs_espeak
class TestEndToEnd:
    def _write_input(self, path: Path, rows: list[dict]) -> None:
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
            "example_method",
            "example_token_match",
            "example_word_count_ok",
            "example_validation_status",
            "example_validation_reason",
            "example_policy",
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
            "example_method": "llm",
            "example_token_match": "pass",
            "example_word_count_ok": "pass",
            "example_validation_status": "pass",
            "example_validation_reason": "",
            "example_policy": "",
        }

    def test_run_with_mock_llm(self, tmp_path):
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [self._row("0001.00.01", "casa", "Minha casa é grande.")],
        )

        mock = _MockAnthropicClient(
            {
                "casa": {
                    "ipa_word_final": "kˈazɐ",
                    "ipa_example_final": "mˈĩɲɐ kˈazɐ ɛ ɡɾˈɐ̃dʒi",
                    "confidence": "high",
                    "notes": "fixed final -a, palatalization",
                }
            }
        )

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "05-ipa.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            anthropic_client=mock,
        )

        assert summary["input_rows"] == 1
        assert summary["output_rows"] == 1
        assert summary["ipa_source_counts"]["corrected"] == 1
        out = read_tsv(tmp_path / "05-ipa.tsv")
        assert out[0]["ipa_word_machine"]  # eSpeak ran
        assert out[0]["ipa_word_final"] == "kˈazɐ"
        assert out[0]["ipa_example_final"] == "mˈĩɲɐ kˈazɐ ɛ ɡɾˈɐ̃dʒi"
        assert out[0]["ipa_confidence"] == "high"

    def test_skip_correction_uses_machine_baseline(self, tmp_path):
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [self._row("0001.00.01", "casa", "Minha casa.")],
        )

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "05-ipa.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            skip_correction=True,
        )

        assert summary["ipa_source_counts"]["machine"] == 1
        out = read_tsv(tmp_path / "05-ipa.tsv")
        # final == machine when skip_correction=True
        assert out[0]["ipa_word_final"] == out[0]["ipa_word_machine"]
        assert out[0]["ipa_example_final"] == out[0]["ipa_example_machine"]

    def test_manual_override_skips_llm(self, tmp_path):
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [self._row("0001.00.01", "casa", "Minha casa.")],
        )
        overrides_path = tmp_path / "manual.tsv"
        write_tsv(
            overrides_path,
            [
                {
                    "sense_id": "0001.00.01",
                    "ipa_word_override": "kˈazɐ",
                    "ipa_example_override": "mˈĩɲɐ kˈazɐ",
                    "notes": "test",
                }
            ],
            fieldnames=[
                "sense_id",
                "ipa_word_override",
                "ipa_example_override",
                "notes",
            ],
        )

        class _Raising:
            cache_stats = {"calls": 0, "cache_hit_ratio": 0}
            def call_tool(self, **kwargs):
                raise RuntimeError("LLM should not be called")

        summary = run(
            input_path=input_path,
            overrides_path=overrides_path,
            output_path=tmp_path / "05-ipa.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            anthropic_client=_Raising(),
        )

        assert summary["manual_overrides"] == 1
        out = read_tsv(tmp_path / "05-ipa.tsv")
        assert out[0]["ipa_word_final"] == "kˈazɐ"
        assert out[0]["ipa_source"] == "manual_override"

    def test_token_mismatch_warning(self, tmp_path):
        """When LLM returns ipa_example_final with wrong token count, the
        summary surfaces it (but the row still writes — soft invariant)."""
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [self._row("0001.00.01", "casa", "Minha casa é grande.")],  # 4 tokens
        )

        mock = _MockAnthropicClient(
            {
                "casa": {
                    "ipa_word_final": "kˈazɐ",
                    "ipa_example_final": "wrong",  # 1 token, expected 4
                    "confidence": "low",
                    "notes": "mistake",
                }
            }
        )

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            output_path=tmp_path / "05-ipa.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            anthropic_client=mock,
        )

        assert summary["token_mismatch_count"] == 1


# --- schema -----------------------------------------------------------------


class TestSchema:
    def test_required_fields(self):
        assert set(CORRECT_IPA_TOOL_SCHEMA["required"]) == {
            "ipa_word_final",
            "ipa_example_final",
            "confidence",
        }

    def test_confidence_enum(self):
        enum = CORRECT_IPA_TOOL_SCHEMA["properties"]["confidence"]["enum"]
        assert set(enum) == {"high", "medium", "low"}
