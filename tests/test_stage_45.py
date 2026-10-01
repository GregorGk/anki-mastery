"""Stage 4.5 tests — speaker gender classifier + voice assignment.

LLM is mocked. Phase 2a/2b deterministic logic is tested directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.lib.voices import (  # noqa: E402
    SEED_FEMALE_POOL,
    SEED_MALE_POOL,
    SEED_NEUTRAL_BALANCE,
    Voice,
    assign_voice_ids,
    load_voices,
    resolve_neutrals,
    split_by_gender,
    usage_counts,
    verify_balanced_within_pool,
)
from build.stage_45 import (  # noqa: E402
    CLASSIFIER_TOOL_SCHEMA,
    ClassificationResult,
    _build_user_message,
    _load_overrides,
    run,
)

CONFIG_DIR = REPO_ROOT / "config"


# --- voice pool loader ------------------------------------------------------


class TestVoicePool:
    """Active round-robin pool is 2F + 5M = 7 voices: the 2026-05 voice swap
    remediation left 2F + 6M (3 voices flagged in pilot review dropped; see
    docs/plan.md § Stage 6 pilot remediation), then Stage 11 retired the BP
    male voice "Lair" (status `retired_bp_only`, kept in voices.tsv only for
    its English pair). `load_voices` returns `status == "active"` rows, so the
    escape-hatch voice Dani (`active_escape_hatch`) is excluded too."""

    def test_committed_pool_has_correct_counts(self):
        voices = load_voices(CONFIG_DIR / "voices.tsv")
        females = [v for v in voices if v.gender == "female"]
        males = [v for v in voices if v.gender == "male"]
        assert len(females) == 2, f"expected 2 female voices, got {len(females)}"
        assert len(males) == 5, f"expected 5 male voices, got {len(males)}"

    def test_pool_indices_are_unique_within_gender(self):
        voices = load_voices(CONFIG_DIR / "voices.tsv")
        for gender in ("female", "male"):
            indices = [v.pool_index for v in voices if v.gender == gender]
            assert len(set(indices)) == len(indices), (
                f"duplicate pool_index in {gender} voices"
            )

    def test_split_by_gender(self):
        voices = load_voices(CONFIG_DIR / "voices.tsv")
        female_voices, male_voices = split_by_gender(voices)
        assert len(female_voices) == 2
        assert len(male_voices) == 5
        # All voice_ids unique
        assert len(set(female_voices + male_voices)) == 7


# --- Phase 2a: resolve_neutrals ---------------------------------------------


class TestResolveNeutrals:
    def test_passthrough_explicit(self):
        gender = {"a": "male", "b": "female"}
        out = resolve_neutrals(gender, seed=42)
        assert out == {"a": "male", "b": "female"}

    def test_neutral_split_balanced_even(self):
        gender = {f"sid{i}": "neutral" for i in range(10)}
        out = resolve_neutrals(gender, seed=44)
        female = sum(1 for g in out.values() if g == "female")
        male = sum(1 for g in out.values() if g == "male")
        assert female == 5
        assert male == 5

    def test_neutral_split_balanced_odd(self):
        gender = {f"sid{i}": "neutral" for i in range(11)}
        out = resolve_neutrals(gender, seed=44)
        female = sum(1 for g in out.values() if g == "female")
        male = sum(1 for g in out.values() if g == "male")
        # Off by at most 1
        assert abs(female - male) <= 1
        assert female + male == 11

    def test_deterministic(self):
        gender = {f"sid{i:04d}": "neutral" for i in range(100)}
        out1 = resolve_neutrals(gender, seed=44)
        out2 = resolve_neutrals(gender, seed=44)
        assert out1 == out2

    def test_different_seeds_diverge(self):
        gender = {f"sid{i:04d}": "neutral" for i in range(100)}
        out44 = resolve_neutrals(gender, seed=44)
        out99 = resolve_neutrals(gender, seed=99)
        # Some assignments differ
        diffs = sum(1 for sid in out44 if out44[sid] != out99[sid])
        assert diffs > 0

    def test_mixed_explicit_and_neutral(self):
        gender = {
            "explicit_M": "male",
            "explicit_F": "female",
            "n1": "neutral",
            "n2": "neutral",
            "n3": "neutral",
            "n4": "neutral",
        }
        out = resolve_neutrals(gender, seed=44)
        assert out["explicit_M"] == "male"
        assert out["explicit_F"] == "female"
        # 4 neutrals split 2/2
        neutral_assignments = [
            out[sid] for sid in ("n1", "n2", "n3", "n4")
        ]
        assert neutral_assignments.count("female") == 2
        assert neutral_assignments.count("male") == 2

    def test_invalid_gender_raises(self):
        with pytest.raises(ValueError):
            resolve_neutrals({"a": "wat"}, seed=44)


# --- Phase 2b: assign_voice_ids --------------------------------------------


class TestAssignVoiceIds:
    @pytest.fixture
    def voices(self):
        return [
            Voice("F1", "female", 1),
            Voice("F2", "female", 2),
            Voice("M1", "male", 1),
            Voice("M2", "male", 2),
            Voice("M3", "male", 3),
        ]

    def test_balanced_within_pool(self, voices):
        # 10 female senses across 2 voices -> 5 each
        gender = {f"f{i}": "female" for i in range(10)}
        gender.update({f"m{i}": "male" for i in range(15)})
        assignment = assign_voice_ids(gender, voices)
        counts = usage_counts(assignment)
        # F1 + F2 split 10 senses: 5+5
        assert counts["F1"] == 5
        assert counts["F2"] == 5
        # M1+M2+M3 split 15 senses: 5+5+5
        assert counts["M1"] == 5
        assert counts["M2"] == 5
        assert counts["M3"] == 5

    def test_off_by_one_when_uneven(self, voices):
        # 7 female senses across 2 voices -> 4+3 (or 3+4)
        gender = {f"f{i}": "female" for i in range(7)}
        gender.update({f"m{i}": "male" for i in range(8)})
        assignment = assign_voice_ids(gender, voices)
        counts = usage_counts(assignment)
        # F1 + F2: 7 split as 4/3 (round-robin)
        f_counts = sorted([counts["F1"], counts["F2"]])
        assert f_counts == [3, 4]
        # M1+M2+M3: 8 split as 3/3/2
        m_counts = sorted([counts["M1"], counts["M2"], counts["M3"]])
        assert m_counts == [2, 3, 3]

    def test_deterministic(self, voices):
        gender = {f"sid{i:04d}": ("female" if i % 2 == 0 else "male") for i in range(50)}
        a1 = assign_voice_ids(gender, voices)
        a2 = assign_voice_ids(gender, voices)
        assert a1 == a2

    def test_no_female_voices_raises(self):
        voices = [Voice("M1", "male", 1)]
        with pytest.raises(ValueError, match="no female"):
            assign_voice_ids({"a": "female"}, voices)

    def test_no_male_voices_raises(self):
        voices = [Voice("F1", "female", 1)]
        with pytest.raises(ValueError, match="no male"):
            assign_voice_ids({"a": "male"}, voices)

    def test_full_pipeline_with_committed_pool(self):
        """Sanity: ~5,725-row simulation produces balanced assignment within
        each pool."""
        voices = load_voices(CONFIG_DIR / "voices.tsv")
        # Simulate 5725 senses with realistic distribution: 70% neutral, 15% F, 15% M
        gender_input = {}
        for i in range(5725):
            mod = i % 20
            if mod < 14:
                gender_input[f"s{i:04d}"] = "neutral"
            elif mod < 17:
                gender_input[f"s{i:04d}"] = "female"
            else:
                gender_input[f"s{i:04d}"] = "male"

        resolved = resolve_neutrals(gender_input, seed=44)
        assignment = assign_voice_ids(resolved, voices)
        counts = usage_counts(assignment)

        # Per-pool balance check
        balanced, msg = verify_balanced_within_pool(assignment, voices)
        assert balanced, f"not balanced: {msg}"


# --- prompt formatting -------------------------------------------------------


class TestUserMessage:
    def test_includes_required_fields(self):
        row = {
            "pt": "casa",
            "en_primary": "house",
            "example_pt": "Minha casa é grande.",
            "example_en": "My house is big.",
        }
        msg = _build_user_message(row)
        assert "casa" in msg
        assert "house" in msg
        assert "Minha casa é grande." in msg
        assert "My house is big." in msg


# --- manual overrides -------------------------------------------------------


class TestManualOverrides:
    def test_load_empty_with_header(self, tmp_path):
        p = tmp_path / "manual.tsv"
        write_tsv(
            p,
            [],
            fieldnames=[
                "sense_id",
                "speaker_gender_override",
                "voice_gender_override",
                "voice_id_override",
                "notes",
            ],
        )
        assert _load_overrides(p) == {}

    def test_load_full_row(self, tmp_path):
        p = tmp_path / "manual.tsv"
        write_tsv(
            p,
            [
                {
                    "sense_id": "0001.00.01",
                    "speaker_gender_override": "female",
                    "voice_gender_override": "",
                    "voice_id_override": "",
                    "notes": "test",
                }
            ],
            fieldnames=[
                "sense_id",
                "speaker_gender_override",
                "voice_gender_override",
                "voice_id_override",
                "notes",
            ],
        )
        ov = _load_overrides(p)
        assert "0001.00.01" in ov
        assert ov["0001.00.01"]["speaker_gender"] == "female"


# --- end-to-end with mocked LLM ---------------------------------------------


class _MockOpenAIClient:
    """Returns canned responses keyed by simple substring match in user_message."""

    def __init__(self, responses: dict[str, dict]):
        self._responses = responses
        self.stats = {"calls": 0, "cache_hit_ratio": 0.0}

    def call_tool(self, *, system, user_message, tool_name, **kwargs):
        for needle, resp in self._responses.items():
            if needle in user_message:
                return dict(resp)
        # Fallback: neutral
        return {
            "speaker_gender": "neutral",
            "evidence": "no cue (mock fallback)",
            "confidence": "low",
        }


class TestEndToEnd:
    def _write_input(self, path: Path, rows: list[dict]) -> None:
        # Stage 4 schema (subset that Stage 4.5 actually reads)
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

    def _make_row(self, sid: str, pt: str, en: str, ex_pt: str, ex_en: str) -> dict:
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
            "en_primary": en,
            "en_all": en,
            "tags": "",
            "bp_status": "standard",
            "example_pt": ex_pt,
            "example_en": ex_en,
            "target_word_used": pt,
            "example_method": "llm",
            "example_token_match": "pass",
            "example_word_count_ok": "pass",
            "example_validation_status": "pass",
            "example_validation_reason": "",
            "example_policy": "",
        }

    def test_run_with_mocks(self, tmp_path):
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [
                self._make_row(
                    "0001.00.01", "obrigada", "thank you (F)",
                    "Obrigada pela ajuda!", "Thanks for the help!",
                ),
                self._make_row(
                    "0001.00.02", "obrigado", "thank you (M)",
                    "Obrigado pela ajuda!", "Thanks for the help!",
                ),
                self._make_row(
                    "0001.00.03", "casa", "house",
                    "Minha casa é grande.", "My house is big.",
                ),
            ],
        )

        mock_client = _MockOpenAIClient(
            {
                "Obrigada pela ajuda": {
                    "speaker_gender": "female",
                    "evidence": "Obrigada — female greeting form",
                    "confidence": "high",
                },
                "Obrigado pela ajuda": {
                    "speaker_gender": "male",
                    "evidence": "Obrigado — male greeting form",
                    "confidence": "high",
                },
                "Minha casa é grande": {
                    "speaker_gender": "neutral",
                    "evidence": "no first-person gendered cue",
                    "confidence": "high",
                },
            }
        )

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            voices_path=CONFIG_DIR / "voices.tsv",
            output_path=tmp_path / "045-speaker_gender.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=2,
            openai_client=mock_client,
        )

        assert summary["input_rows"] == 3
        assert summary["output_rows"] == 3
        assert summary["raw_gender_counts"]["female"] == 1
        assert summary["raw_gender_counts"]["male"] == 1
        assert summary["raw_gender_counts"]["neutral"] == 1

        out = read_tsv(tmp_path / "045-speaker_gender.tsv")
        # Female and male senses preserved; neutral resolved
        sid_to_row = {r["sense_id"]: r for r in out}
        assert sid_to_row["0001.00.01"]["voice_gender_assigned"] == "female"
        assert sid_to_row["0001.00.02"]["voice_gender_assigned"] == "male"
        assert sid_to_row["0001.00.03"]["voice_gender_assigned"] in ("male", "female")
        # voice_id is set on every row
        for r in out:
            assert r["voice_id"], f"empty voice_id for {r['sense_id']}"

    def test_manual_override_skips_llm(self, tmp_path):
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [
                self._make_row(
                    "0001.00.01", "casa", "house",
                    "Minha casa é grande.", "My house is big.",
                ),
            ],
        )

        # Mock raises if called
        class _RaisingMock:
            stats = {"calls": 0, "cache_hit_ratio": 0}
            def call_tool(self, **kwargs):
                raise RuntimeError("LLM should not be called when override exists")

        # Manual override forcing female
        overrides_path = tmp_path / "manual.tsv"
        write_tsv(
            overrides_path,
            [
                {
                    "sense_id": "0001.00.01",
                    "speaker_gender_override": "female",
                    "voice_gender_override": "",
                    "voice_id_override": "",
                    "notes": "test override",
                }
            ],
            fieldnames=[
                "sense_id",
                "speaker_gender_override",
                "voice_gender_override",
                "voice_id_override",
                "notes",
            ],
        )

        summary = run(
            input_path=input_path,
            overrides_path=overrides_path,
            voices_path=CONFIG_DIR / "voices.tsv",
            output_path=tmp_path / "045-speaker_gender.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            openai_client=_RaisingMock(),
        )

        assert summary["manual_overrides"] == 1
        out = read_tsv(tmp_path / "045-speaker_gender.tsv")
        assert out[0]["speaker_gender"] == "female"
        assert out[0]["voice_gender_assigned"] == "female"
        assert out[0]["assignment_method"] == "manual_override"


# --- confidence policy -----------------------------------------------------


class TestConfidencePolicy:
    """Only `confidence=high` keeps explicit gender; medium/low rows are
    demoted to neutral so they go through the seeded balanced shuffle."""

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

    def _make_row(self, sid: str, ex_pt: str) -> dict:
        return {
            "sense_id": sid,
            "rank": "1",
            "expansion_index": "0",
            "sense_index": "1",
            "pt": "x",
            "pt_display": "x",
            "pt_type": "single_word",
            "gender": "",
            "pos": "noun",
            "en_primary": "x",
            "en_all": "x",
            "tags": "",
            "bp_status": "standard",
            "example_pt": ex_pt,
            "example_en": "x",
            "target_word_used": "x",
            "example_method": "llm",
            "example_token_match": "pass",
            "example_word_count_ok": "pass",
            "example_validation_status": "pass",
            "example_validation_reason": "",
            "example_policy": "",
        }

    def test_high_confidence_male_keeps_male(self, tmp_path):
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [self._make_row("0001.00.01", "Estou cansado.")],
        )

        class _Mock:
            stats = {"calls": 0, "cache_hit_ratio": 0}
            def call_tool(self, **kwargs):
                return {
                    "speaker_gender": "male",
                    "evidence": "cansado",
                    "confidence": "high",
                }

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            voices_path=CONFIG_DIR / "voices.tsv",
            output_path=tmp_path / "045.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            openai_client=_Mock(),
        )
        assert summary["demoted_to_neutral_due_to_low_confidence"] == 0
        out = read_tsv(tmp_path / "045.tsv")
        assert out[0]["voice_gender_assigned"] == "male"

    def test_medium_confidence_male_demoted_to_neutral(self, tmp_path):
        # Build a corpus where ALL rows are medium-confidence so the demotion
        # is observable. With 4 senses (all neutral after demote), the seeded
        # shuffle splits 2/2.
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [self._make_row(f"000{i}.00.01", "minha mãe") for i in range(1, 5)],
        )

        class _Mock:
            stats = {"calls": 0, "cache_hit_ratio": 0}
            def call_tool(self, **kwargs):
                return {
                    "speaker_gender": "male",  # weak inference
                    "evidence": "minha mãe (weak)",
                    "confidence": "medium",
                }

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            voices_path=CONFIG_DIR / "voices.tsv",
            output_path=tmp_path / "045.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            openai_client=_Mock(),
        )
        # All 4 rows should be demoted (confidence=medium)
        assert summary["demoted_to_neutral_due_to_low_confidence"] == 4

        out = read_tsv(tmp_path / "045.tsv")
        # Raw classifier output preserved
        for r in out:
            assert r["speaker_gender"] == "male"
            assert r["confidence"] == "medium"
        # But voice_gender_assigned should split 2/2 (balanced shuffle on neutrals)
        from collections import Counter
        c = Counter(r["voice_gender_assigned"] for r in out)
        assert c["female"] == 2 and c["male"] == 2

    def test_low_confidence_female_demoted_to_neutral(self, tmp_path):
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [self._make_row(f"000{i}.00.01", "x") for i in range(1, 3)],
        )

        class _Mock:
            stats = {"calls": 0, "cache_hit_ratio": 0}
            def call_tool(self, **kwargs):
                return {
                    "speaker_gender": "female",
                    "evidence": "weak",
                    "confidence": "low",
                }

        summary = run(
            input_path=input_path,
            overrides_path=tmp_path / "no_manual.tsv",
            voices_path=CONFIG_DIR / "voices.tsv",
            output_path=tmp_path / "045.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            openai_client=_Mock(),
        )
        assert summary["demoted_to_neutral_due_to_low_confidence"] == 2

    def test_manual_override_bypasses_demotion(self, tmp_path):
        input_path = tmp_path / "04-examples.tsv"
        self._write_input(
            input_path,
            [self._make_row("0001.00.01", "x")],
        )
        # Manual override forces "male" — should bypass any LLM call AND the
        # confidence demotion.
        overrides_path = tmp_path / "manual.tsv"
        write_tsv(
            overrides_path,
            [
                {
                    "sense_id": "0001.00.01",
                    "speaker_gender_override": "male",
                    "voice_gender_override": "",
                    "voice_id_override": "",
                    "notes": "manual",
                }
            ],
            fieldnames=[
                "sense_id",
                "speaker_gender_override",
                "voice_gender_override",
                "voice_id_override",
                "notes",
            ],
        )

        class _RaisingMock:
            stats = {"calls": 0, "cache_hit_ratio": 0}
            def call_tool(self, **kwargs):
                raise RuntimeError("LLM should not be called")

        summary = run(
            input_path=input_path,
            overrides_path=overrides_path,
            voices_path=CONFIG_DIR / "voices.tsv",
            output_path=tmp_path / "045.tsv",
            audit_path=tmp_path / "audit.jsonl",
            concurrency=1,
            openai_client=_RaisingMock(),
        )
        assert summary["demoted_to_neutral_due_to_low_confidence"] == 0
        out = read_tsv(tmp_path / "045.tsv")
        assert out[0]["voice_gender_assigned"] == "male"


# --- schema -----------------------------------------------------------------


class TestSchema:
    def test_classifier_schema_required_fields(self):
        assert set(CLASSIFIER_TOOL_SCHEMA["required"]) == {
            "speaker_gender",
            "evidence",
            "confidence",
        }

    def test_speaker_gender_enum(self):
        enum = CLASSIFIER_TOOL_SCHEMA["properties"]["speaker_gender"]["enum"]
        assert set(enum) == {"male", "female", "neutral"}
