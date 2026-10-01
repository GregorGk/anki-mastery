"""Stage 19 / Step 0 — library prep: no network, no API spend.

Covers the small library changes the v4 re-render relies on:
  * final_postfix: EP→BP spelling map + card-IPA precedence
  * tsv.write_tsv: atomic replace
  * llm: Claude 5.x tool-call params (no forced tool_choice) + batch id alias
  * elevenlabs_client: v4 voice settings, seed override, Retry-After, quota
  * judge_prompts: J1 byte-identical, J1p/J2 derived at fixed anchors
  * asr.phonetic_distance: espeak timeouts don't escape
  * verify_all: eleven_v4 accepted
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import final_postfix as FP  # noqa: E402
from build.lib import judge_prompts as JP  # noqa: E402
from build.lib import llm  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402


# ── final_postfix ────────────────────────────────────────────────────────────
MAP = {"génio": "gênio", "cómodo": "cômodo", "polémica": "polêmica"}


@pytest.mark.parametrize("text,expected,n", [
    ("o génio", "o gênio", 1),
    ("Génio é raro.", "Gênio é raro.", 1),
    ("GÉNIO", "GÊNIO", 1),
    ("A polémica sobre o cómodo.", "A polêmica sobre o cômodo.", 2),
    ("genial e geniozinho", "genial e geniozinho", 0),   # whole tokens only
    ("", "", 0),
])
def test_apply_spelling_map(text, expected, n):
    assert FP.apply_spelling_map(text, MAP) == (expected, n)


def test_apply_spelling_to_row_only_text_fields():
    row = {"pt": "cómodo", "pt_display": "o cómodo", "example_pt": "O cómodo é grande.",
           "target_word_used": "cómodo", "source_pt": "cómodo", "ipa_word": "kˈɔmodu"}
    assert FP.apply_spelling_to_row(row, MAP) == 4
    assert row["pt_display"] == "o cômodo" and row["example_pt"] == "O cômodo é grande."
    assert row["source_pt"] == "cómodo"          # provenance untouched
    assert row["ipa_word"] == "kˈɔmodu"


def test_spelling_map_file_has_stage19_entries():
    m = FP.load_spelling_map(REPO_ROOT / "data" / "_ep_spelling_map.tsv")
    for ep, bp in [("génio", "gênio"), ("ingénuo", "ingênuo"), ("polémica", "polêmica"),
                   ("cómodo", "cômodo"), ("incómodo", "incômodo")]:
        assert m[ep] == bp


def test_resolve_ipa_precedence():
    s5 = {"ipa_word_final": "s5w", "ipa_example_final": "s5e"}
    v2 = {"x": {"ipa_word": "v2w", "ipa_example": "v2e"}}
    manual = {"x": {"ipa_word_override": "mw", "ipa_example_override": ""}}
    assert FP.resolve_ipa("x", s5, v2, manual) == ("mw", "v2e", "manual", "ipa_v2")
    assert FP.resolve_ipa("x", s5, v2, {}) == ("v2w", "v2e", "ipa_v2", "ipa_v2")
    assert FP.resolve_ipa("y", s5, v2, manual) == ("s5w", "s5e", "stage5", "stage5")


# ── tsv atomic write ─────────────────────────────────────────────────────────
def test_write_tsv_atomic(tmp_path):
    p = tmp_path / "t.tsv"
    write_tsv(p, [{"a": "1", "b": 'x "q" y'}], fieldnames=["a", "b"])
    assert read_tsv(p) == [{"a": "1", "b": 'x "q" y'}]

    def boom():
        yield {"a": "2", "b": "z"}
        raise RuntimeError("crash mid-write")

    with pytest.raises(RuntimeError):
        write_tsv(p, boom(), fieldnames=["a", "b"])
    assert read_tsv(p) == [{"a": "1", "b": 'x "q" y'}]   # original intact
    assert [x.name for x in tmp_path.iterdir()] == ["t.tsv"]  # no temp left


# ── llm: Claude 5.x params ───────────────────────────────────────────────────
SCHEMA = {"type": "object", "properties": {"ipa": {"type": "string"}},
          "required": ["ipa"], "additionalProperties": False}


def test_tool_params_legacy_model_forces_tool():
    p = llm.tool_request_params(model_id="claude-sonnet-4-5", system_param="sys",
                                tool_name="t", tool_description="d",
                                tool_input_schema=SCHEMA, max_tokens=1024)
    assert p["tool_choice"] == {"type": "tool", "name": "t"}
    assert p["max_tokens"] == 1024 and p["system"] == "sys"
    assert "output_config" not in p


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5"])
def test_tool_params_claude5_uses_auto(model):
    sysp = [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    p = llm.tool_request_params(model_id=model, system_param=sysp, tool_name="give_ipa",
                                tool_description="d", tool_input_schema=SCHEMA,
                                max_tokens=1024, strict=True, effort="high")
    assert p["tool_choice"] == {"type": "auto"}
    assert p["max_tokens"] >= llm.THINKING_MODEL_MIN_MAX_TOKENS
    assert p["tools"][0]["strict"] is True
    assert p["output_config"] == {"effort": "high"}
    assert "give_ipa" in p["system"][-1]["text"]
    assert sysp[0]["text"] == "SYS"                 # caller's list not mutated
    assert p["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_submit_batch_accepts_id_alias():
    captured = {}

    class _Batches:
        def create(self, requests):
            captured["requests"] = requests

            class R:
                id = "batch_1"
            return R()

    class _Messages:
        batches = _Batches()

    client = llm.AnthropicClient.__new__(llm.AnthropicClient)
    client.model = "claude-opus-5-5"
    client.premium_model = "claude-opus-5-5"
    client.enable_caching = False
    client.client = type("C", (), {"messages": _Messages()})()
    bid = client.submit_batch([{"id": "row-1", "user_message": "hi"}], system="s",
                              tool_name="t", tool_input_schema=SCHEMA)
    assert bid == "batch_1"
    req = captured["requests"][0]
    assert req["custom_id"] == "row-1"
    assert req["params"]["tool_choice"] == {"type": "auto"}
    assert req["params"]["messages"] == [{"role": "user", "content": "hi"}]


# ── elevenlabs_client ────────────────────────────────────────────────────────
def _el():
    import os
    os.environ.setdefault("ELEVENLABS_API_KEY", "test-dummy")
    from build.lib import elevenlabs_client as E
    return E


def test_voice_settings_for_v4_drops_style_and_boost():
    E = _el()
    vs = E.voice_settings_for_model("eleven_v4", E.LOCKED_VOICE_SETTINGS)
    assert vs.stability == 0.65 and vs.similarity_boost == 0.80
    assert vs.style is None and vs.use_speaker_boost is None
    assert E.voice_settings_for_model("eleven_v3", E.LOCKED_VOICE_SETTINGS) is E.LOCKED_VOICE_SETTINGS


def test_generate_pcm_seed_override_and_settings(monkeypatch):
    E = _el()
    c = E.ElevenLabsClient(model_id="eleven_v4")
    seen = {}

    def fake_call(*, text, voice_id, seed, voice_settings=None):
        seen.update(seed=seed, vs=voice_settings)
        return b"\x00\x00" * 10

    monkeypatch.setattr(c, "_call_once", fake_call)
    r = c.generate_pcm(text="oi", voice_id="v", sense_id="0001.00.01",
                       clip_type="word", version=1, seed=12345)
    assert r.seed == 12345 and seen["seed"] == 12345
    assert seen["vs"].style is None                     # v4 settings applied
    r2 = c.generate_pcm(text="oi", voice_id="v", sense_id="0001.00.01",
                        clip_type="word", version=1)
    assert r2.seed == E.stable_seed("0001.00.01", "word", 1)


def test_retry_after_header_case_insensitive():
    E = _el()
    from elevenlabs.core.api_error import ApiError
    exc = ApiError(status_code=429, headers={"retry-after": "7"}, body="slow down")
    assert E.ElevenLabsClient._extract_retry_after(exc) == 7.0


def test_quota_exceeded_is_raised_not_retried(monkeypatch):
    E = _el()
    from elevenlabs.core.api_error import ApiError
    c = E.ElevenLabsClient(model_id="eleven_v4")
    calls = {"n": 0}

    def fake_call(**kw):
        calls["n"] += 1
        raise ApiError(status_code=401, headers={},
                       body={"detail": {"status": "quota_exceeded", "message": "no credits"}})

    monkeypatch.setattr(c, "_call_once", fake_call)
    with pytest.raises(E.QuotaExceeded):
        c.generate_pcm(text="oi", voice_id="v", sense_id="s", clip_type="word", version=1)
    assert calls["n"] == 1


# ── judge prompts ────────────────────────────────────────────────────────────
def test_judge_prompt_versions():
    from build.lib import gemini_audio_judge as G
    base = G.JUDGE_SYSTEM_PROMPT
    assert JP.build_system_prompt(base, "J1") is base
    j1p = JP.build_system_prompt(base, "J1p")
    j2 = JP.build_system_prompt(base, "J2")
    assert j1p.count("Coda r (before a consonant") == 1
    assert "Expected pronunciation check" not in j1p
    assert j2.count("Expected pronunciation check") == 1
    assert j2.index("Expected pronunciation check") < j2.index("Common drift directions:")
    with pytest.raises(ValueError):
        JP.build_system_prompt(base, "J9")


def test_j1_user_text_matches_production_wording():
    from build.lib import gemini_audio_judge as G
    legacy = ("Word: verde\nTarget IPA (Brazilian Portuguese): \nClip type: word\n\n"
              "Respond with ONE raw JSON object only — no preamble, no markdown.")
    assert JP.user_text(pt="verde", ipa="", clip_type="word", version="J1",
                        json_only_suffix=G.JSON_ONLY_SUFFIX) == legacy
    j2 = JP.user_text(pt="verde", ipa="ˈveɾdʒi", clip_type="word", version="J2")
    assert "/ˈveɾdʒi/" in j2


# ── asr ──────────────────────────────────────────────────────────────────────
def test_phonetic_distance_survives_espeak_timeout(monkeypatch):
    from build.lib import asr

    def slow(_):
        raise subprocess.TimeoutExpired(cmd="espeak-ng", timeout=5)

    monkeypatch.setattr(asr, "ipa_transcribe", slow)
    assert asr.phonetic_distance("o", "u") is None


# ── verify_all ───────────────────────────────────────────────────────────────
def test_verify_all_accepts_eleven_v4():
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_all", REPO_ROOT / "build" / "verify_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "eleven_v4" in mod.EXPECTED_MODEL_MARKERS
    assert "eleven_v3" in mod.EXPECTED_MODEL_MARKERS
