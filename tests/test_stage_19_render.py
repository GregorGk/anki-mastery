"""Stage 19 — render engine + full-run ladder with fake clients (no network, no spend).

Engine (build/lib/v4_tts.py): seeds unique per take, ASR/judge never see the
TTS string, takes cached in the ledger (a second call doesn't re-render).
Full run (build/19_4_render_v4.py): best-of-2 selection, accept-only manifest
writes, unresolved clips keep their v3 row byte-identical, en_ex untouched,
versions monotonic, budget stop.
"""
from __future__ import annotations

import importlib.util
import math
import struct
import sys
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import stage19_data as D  # noqa: E402
from build.lib import v4_tts as T  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("stage19_4", REPO_ROOT / "build" / "19_4_render_v4.py")


def _pcm(seconds=0.6, sr=44100):
    return b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 200 * i / sr)))
                    for i in range(int(seconds * sr)))


def _spec(**kw):
    base = dict(sense_id="0534.00.01", clip_type="word", display_text="verde",
                voice_id="Rw38T6bn0lTNOb1aUevR", expected_ipa="ˈveɾdʒi")
    base.update(kw)
    return T.RenderSpec(**base)


# ── seeds / spec ─────────────────────────────────────────────────────────────
def test_seeds_unique_per_take_variant_voice():
    seeds = {_spec(take=t, variant=v, voice_id=vo).seed
             for t in (1, 2, 3) for v in ("plain", "tag") for vo in ("A", "B")}
    assert len(seeds) == 12
    assert _spec(take=1).seed == _spec(take=1).seed


def test_tag_variant_only_changes_tts_text():
    s = _spec(variant="tag")
    assert s.tts_text.startswith(T.DEFAULT_TAG) and s.display_text == "verde"


# ── engine with fakes ────────────────────────────────────────────────────────
class FakeEL:
    def __init__(self):
        self.calls = []

    def generate_pcm_meta(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(audio_pcm=_pcm(), character_cost=2, latency_ms=5)


class FakeJudge:
    def __init__(self, verdict="bp_ok"):
        self.verdict = verdict
        self.seen = []

    def judge(self, **kw):
        self.seen.append(kw)
        return types.SimpleNamespace(pronunciation_verdict=self.verdict, drift="none",
                                     severity="low", evidence="ok")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "LEDGER", tmp_path / "takes.tsv")
    monkeypatch.setattr(T, "TAKES_DIR", tmp_path / "takes")
    (tmp_path / "takes").mkdir()
    import build.lib.loudness as L
    import build.lib.asr as A
    monkeypatch.setattr(L, "normalize_pcm_to_mp3_verified", lambda pcm: types.SimpleNamespace(
        mp3_bytes=b"ID3fake" + pcm[:64], final_mp3_lufs=-16.1, final_mp3_tp=-1.7,
        within_tolerance=True, tp_limited=False, applied_gain_db=1.0))
    seen_asr = []

    def fake_roundtrip(**kw):
        seen_asr.append(kw["input_text"])
        return types.SimpleNamespace(transcript="Verde.", biased_transcript="Verde.",
                                     decision="pass", text_similarity=1.0)
    monkeypatch.setattr(A, "asr_roundtrip", fake_roundtrip)
    e = T.Engine.__new__(T.Engine)
    e.el = FakeEL()
    e.el_dict = None
    e.asr = object()
    cfg = {"config": "j1", "backend": "openai", "model": "fake", "prompt_version": "J2"}
    e.judge_configs = [cfg]
    e.judge_hash = T.judge_hash([cfg])
    e.judges = [(cfg, FakeJudge())]
    e.el_sem = threading.BoundedSemaphore(2)
    e.asr_sem = threading.BoundedSemaphore(2)
    e.openai_judge_sem = threading.BoundedSemaphore(2)
    e.gemini_limiter = types.SimpleNamespace(wait_before_call=lambda: None)
    e.ledger_lock = threading.Lock()
    e.credits_used = 0
    e.cached = {}
    e.seen_asr = seen_asr
    return e


def test_render_take_passes_and_hides_tts_text(engine):
    res = engine.render_take(_spec(variant="tag"))
    assert res.qa_pass, (res.sanity, res.loudness, res.asr_mode, res.judges, res.error)
    assert engine.seen_asr == ["verde"]                              # ASR got display text
    judge_kw = engine.judges[0][1].seen[0]
    assert judge_kw["pt"] == "verde" and judge_kw["ipa_word_final"] == "ˈveɾdʒi"
    assert engine.el.calls[0]["text"].startswith(T.DEFAULT_TAG)      # TTS got the tag
    assert Path(res.path).exists() and res.credits == 2


def test_render_take_is_cached(engine):
    s = _spec()
    engine.render_take(s)
    engine.render_take(_spec())
    assert len(engine.el.calls) == 1


def test_judge_rejection_fails_gate(engine):
    engine.judges = [(engine.judge_configs[0], FakeJudge("non_bp"))]
    res = engine.render_take(_spec(take=7))
    assert not res.qa_pass and res.asr_pass and not res.gate_pass


# ── full-run ladder / manifest with a fake engine ────────────────────────────
def _job(sid="0001.00.01", ct="word", voice="Rw38T6bn0lTNOb1aUevR"):
    return D.ClipJob(sense_id=sid, clip_type=ct, display_text="o livro" if ct == "word"
                     else "O livro está na mesa.", voice_id=voice, voice_gender="female",
                     expected_ipa="u ˈlivɾu", spoken_reference="", rank=10, manual_pass=False,
                     v3_object_key=f"audio/{sid}-{'word' if ct == 'word' else 'ex'}-eleven_v3-v1.mp3",
                     v3_model="eleven_v3", order=1)


def _take(spec, ok, tmp_path, judges_ok=1, sim=0.99):
    p = tmp_path / f"{spec.take_id.replace('|', '_')}.mp3"
    p.write_bytes(b"ID3" + spec.take_id.encode())
    r = T.TakeResult(spec=spec, path=str(p), md5="m" * 32, lufs=-16.0, true_peak=-1.8,
                     asr_pass=ok, asr_mode="production" if ok else "fail", asr_similarity=sim,
                     judges={"j": {"verdict": "bp_ok" if ok else "non_bp"}} if judges_ok else {},
                     gate_pass=ok, generated_at="2026-10-01T00:00:00+00:00")
    return r


def _run(tmp_path, monkeypatch, outcomes):
    """outcomes: callable(spec) -> bool (qa pass)."""
    monkeypatch.setattr(R, "PROVENANCE", tmp_path / "prov.tsv")
    monkeypatch.setattr(R, "UNRESOLVED", tmp_path / "unres.tsv")
    run = R.Run.__new__(R.Run)
    run.args = types.SimpleNamespace(limit=0, no_upload=True, manifest_out=str(tmp_path / "m.tsv"))
    job_w, job_e = _job(), _job(ct="example")
    run.jobs = {job_w.key: job_w, job_e.key: job_e}
    run.manifest = [
        {"sense_id": "0001.00.01", "clip_type": "word", "tts_model": "eleven_v3", "voice_id": job_w.voice_id,
         "object_key": job_w.v3_object_key, "version": "1", "status": "uploaded", "notes": ""},
        {"sense_id": "0001.00.01", "clip_type": "example", "tts_model": "eleven_v3",
         "voice_id": job_e.voice_id, "object_key": job_e.v3_object_key, "version": "1",
         "status": "uploaded", "notes": ""},
        {"sense_id": "0001.00.01", "clip_type": "en_ex", "tts_model": "eleven_v3",
         "voice_id": "EN", "object_key": "audio/0001.00.01-en_ex-eleven_v3-v1.mp3",
         "version": "1", "status": "uploaded", "notes": ""},
    ]
    run.idx = {(r["sense_id"], r["clip_type"]): r for r in run.manifest}
    run.en_ex_before = [dict(run.manifest[2])]
    run.risky = set()
    run.variants = ["plain"]
    run.stability = 0.65
    run.dict_on = False
    run.voice_rank = {}
    run.short_policy = "keep_v3"
    rendered = []

    def render_take(spec):
        rendered.append(spec)
        return _take(spec, outcomes(spec), tmp_path)

    run.engine = types.SimpleNamespace(render_take=render_take, credits_used=0,
                                       judge_configs=[], judge_hash="h")
    run.r2 = None
    run.lock = threading.Lock()
    run.renders = run.accepted = 0
    import collections
    run.stats = collections.Counter()
    run.versions_used = {}
    run.stop = threading.Event()
    run.manifest_out = Path(run.args.manifest_out)
    monkeypatch.setattr(T, "mp3_duration", lambda p: 0.8)
    monkeypatch.setattr(D, "swap_pool", lambda g: ["Rw38T6bn0lTNOb1aUevR", "m151rjrbWXbBqyq56tly",
                                                   "PznTnBc8X6pvixs9UkQm"])
    return run, rendered


def test_word_best_of_two_accepts_and_writes_row(tmp_path, monkeypatch):
    run, rendered = _run(tmp_path, monkeypatch, lambda s: True)
    j = run.jobs[("0001.00.01", "word")]
    run.process(j)
    assert [s.take for s in rendered] == [1, 2]                     # best-of-2, no more
    row = run.idx[("0001.00.01", "word")]
    assert row["tts_model"] == "eleven_v4" and row["object_key"].endswith("-eleven_v4-v1.mp3")
    assert row["notes"] == "stage_19_v4:plain"
    assert run.idx[("0001.00.01", "en_ex")] == run.en_ex_before[0]


def test_unresolved_keeps_v3_row_identical(tmp_path, monkeypatch):
    run, rendered = _run(tmp_path, monkeypatch, lambda s: False)
    j = run.jobs[("0001.00.01", "word")]
    before = dict(run.idx[j.key])
    run.process(j)
    assert run.idx[j.key] == before
    assert len(rendered) == 6            # 2 + 2 takes + 2 voice swaps (cap 8)
    assert {s.voice_id for s in rendered[4:]} == {"m151rjrbWXbBqyq56tly", "PznTnBc8X6pvixs9UkQm"}
    assert (tmp_path / "unres.tsv").exists()


def test_example_ladder_cap(tmp_path, monkeypatch):
    run, rendered = _run(tmp_path, monkeypatch, lambda s: False)
    run.process(run.jobs[("0001.00.01", "example")])
    assert len(rendered) == 4            # t1, t2, 2 swaps (cap 5)


def test_swap_accepts_new_voice(tmp_path, monkeypatch):
    run, rendered = _run(tmp_path, monkeypatch,
                         lambda s: s.voice_id == "m151rjrbWXbBqyq56tly")
    run.process(run.jobs[("0001.00.01", "word")])
    assert run.idx[("0001.00.01", "word")]["voice_id"] == "m151rjrbWXbBqyq56tly"
    assert run.stats["accepted_after_voice_swap"] == 1


def test_versions_monotonic_per_clip(tmp_path, monkeypatch):
    run, _ = _run(tmp_path, monkeypatch, lambda s: True)
    j = run.jobs[("0001.00.01", "word")]
    run.versions_used[j.key] = 3
    assert run._next_version(j) == 4


def test_budget_stop_halts_without_writing(tmp_path, monkeypatch):
    run, rendered = _run(tmp_path, monkeypatch, lambda s: True)
    run.stop.set()
    j = run.jobs[("0001.00.01", "word")]
    before = dict(run.idx[j.key])
    run.process(j)
    assert rendered == [] and run.idx[j.key] == before
