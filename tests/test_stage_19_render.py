"""Stage 19 — render engine + full-run ladder with fake clients (no network, no spend).

Engine (build/lib/v4_tts.py): seeds unique per take, ASR/judge never see the
TTS string, takes cached in the ledger (a second call doesn't re-render), ASR
and judge errors re-checked from the saved MP3, TTS/ASR/judge outages stop the
run, one engine process at a time.
Full run (build/19_4_render_v4.py): best-of-2 selection, accept-only manifest
writes, unresolved clips keep their v3 row byte-identical, en_ex untouched,
versions monotonic, budget stop.
"""
from __future__ import annotations

import hashlib
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
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402


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
    e.qa_health = T.QaHealth()
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


def test_cache_survives_a_moved_checkout(engine):
    """Ledger paths are absolute: a take rendered in another checkout (the pilot
    began on a Mac) is reused, not re-bought, once its MP3 is in TAKES_DIR."""
    engine.render_take(_spec())
    rows = read_tsv(T.LEDGER)
    rows[0]["path"] = ("/Users/someone/anki-mastery/build/audio_cache/v4_takes/"
                       + Path(rows[0]["path"]).name)
    write_tsv(T.LEDGER, rows, fieldnames=T.LEDGER_FIELDS)
    engine.cached = engine._load_ledger()
    res = engine.render_take(_spec())
    assert len(engine.el.calls) == 1
    assert Path(res.path).parent == T.TAKES_DIR and Path(res.path).is_file()


def test_overwritten_take_is_rerendered(engine):
    """A cached MP3 whose bytes no longer match the ledger md5 is not trusted."""
    res = engine.render_take(_spec())
    Path(res.path).write_bytes(b"ID3 overwritten elsewhere")
    engine.cached = engine._load_ledger()
    again = engine.render_take(_spec())
    assert len(engine.el.calls) == 2
    assert hashlib.md5(Path(again.path).read_bytes()).hexdigest() == read_tsv(T.LEDGER)[-1]["md5"]


class FlakyJudge(FakeJudge):
    def __init__(self):
        super().__init__()
        self.fail = True

    def judge(self, **kw):
        if self.fail:
            raise RuntimeError("429 Too Many Requests")
        return super().judge(**kw)


def test_judge_error_is_retried_from_cache(engine):
    judge = FlakyJudge()
    engine.judges = [(engine.judge_configs[0], judge)]
    first = engine.render_take(_spec())
    assert not first.gate_pass and first.judges["j1"]["verdict"] == "error"
    judge.fail = False
    engine.cached = engine._load_ledger()
    again = engine.render_take(_spec())
    assert again.qa_pass and len(engine.el.calls) == 1               # re-judged, not re-rendered


def test_judge_rejection_fails_gate(engine):
    engine.judges = [(engine.judge_configs[0], FakeJudge("non_bp"))]
    res = engine.render_take(_spec(take=7))
    assert not res.qa_pass and res.asr_pass and not res.gate_pass


# ── outages: QA errors keep the MP3, provider outages stop the run ───────────
class FakeApiError(Exception):
    """Shaped like google-genai (`code`) / OpenAI (`status_code`) API errors."""

    def __init__(self, msg, code=None, status_code=None):
        super().__init__(msg)
        self.code, self.status_code = code, status_code


class DownJudge(FakeJudge):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def judge(self, **kw):
        if self.exc:
            raise self.exc
        return super().judge(**kw)


def test_judge_402_stops_the_run_after_the_row(engine):
    """Gemini 402 (prepaid credits depleted): the take's row is kept with its MP3,
    then JudgeUnavailable — and no further take is bought in this process."""
    engine.judges = [(engine.judge_configs[0], DownJudge(
        FakeApiError("402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted.", code=402)))]
    with pytest.raises(T.JudgeUnavailable, match="HTTP 402"):
        engine.render_take(_spec())
    row = read_tsv(T.LEDGER)[-1]
    assert row["md5"] and Path(row["path"]).is_file() and '"error"' in row["judges"]
    with pytest.raises(T.QaUnavailable):
        engine.render_take(_spec(take=2))
    assert len(engine.el.calls) == 1


def test_judge_error_streak_raises_and_resets(engine, monkeypatch):
    monkeypatch.setattr(T, "JUDGE_ERROR_STREAK", 3)
    judge = DownJudge(FakeApiError("503 UNAVAILABLE", code=503))
    engine.judges = [(engine.judge_configs[0], judge)]
    for t in (1, 2):
        res = engine.render_take(_spec(take=t))
        assert res.judge_error and res.qa_error and not res.qa_pass
    judge.exc = None
    assert not engine.render_take(_spec(take=3)).judge_error       # a verdict resets the streak
    judge.exc = FakeApiError("503 UNAVAILABLE", code=503)
    engine.render_take(_spec(take=4))
    engine.render_take(_spec(take=5))
    with pytest.raises(T.JudgeUnavailable, match="3 errors in a row"):
        engine.render_take(_spec(take=6))
    assert len(read_tsv(T.LEDGER)) == 6                              # the 6th row is kept too


def test_one_dead_judge_trips_its_own_streak(engine, monkeypatch):
    """Per-judge streaks: a healthy second judge must not mask a dead first one."""
    monkeypatch.setattr(T, "JUDGE_ERROR_STREAK", 2)
    cfg2 = dict(engine.judge_configs[0], config="j2")
    engine.judges = [(engine.judge_configs[0], DownJudge(FakeApiError("500 INTERNAL", code=500))),
                     (cfg2, FakeJudge())]
    engine.render_take(_spec(take=1))
    with pytest.raises(T.JudgeUnavailable, match="judge:j1"):
        engine.render_take(_spec(take=2))


def test_qa_health_streak_is_thread_safe():
    h = T.QaHealth()
    exc = RuntimeError("timeout")

    def hammer():
        for _ in range(500):
            h.error("asr", exc, 10**9, T.AsrUnavailable)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert "4001 errors in a row" in str(h.error("asr", exc, 4001, T.AsrUnavailable))


def test_unavailable_reason_separates_billing_from_rate_limits():
    assert T.unavailable_reason(FakeApiError("402 RESOURCE_EXHAUSTED", code=402)) == "HTTP 402"
    assert T.unavailable_reason(FakeApiError("Error code: 401", status_code=401)) == "HTTP 401"
    assert T.unavailable_reason(FakeApiError(
        "Error code: 429 - {'code': 'insufficient_quota'}", status_code=429)) == "insufficient_quota"
    # Gemini's per-minute rate limit mentions quota and billing too: transient.
    assert T.unavailable_reason(FakeApiError(
        "429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and "
        "billing details.", code=429)) == ""
    assert T.unavailable_reason(TimeoutError("The read operation timed out")) == ""


def _failing_asr(monkeypatch, exc):
    import build.lib.asr as A
    ok = A.asr_roundtrip
    calls = []

    def roundtrip(**kw):
        calls.append(kw)
        if exc[0]:
            raise exc[0]
        return ok(**kw)
    monkeypatch.setattr(A, "asr_roundtrip", roundtrip)
    return calls


def test_asr_error_keeps_mp3_and_is_retried_from_cache(engine, monkeypatch):
    exc = [FakeApiError("Connection error.")]
    calls = _failing_asr(monkeypatch, exc)
    first = engine.render_take(_spec())
    assert first.error.startswith("asr:") and first.qa_error and not first.judges
    row = read_tsv(T.LEDGER)[-1]
    assert row["md5"] and Path(row["path"]).is_file()                # the paid MP3 is kept
    exc[0] = None
    engine.cached = engine._load_ledger()
    again = engine.render_take(_spec())
    assert again.qa_pass and not again.error and len(calls) == 2
    assert len(engine.el.calls) == 1                                 # re-transcribed, not re-bought
    assert read_tsv(T.LEDGER)[-1]["error"] == ""


def test_asr_error_with_overwritten_mp3_is_rerendered(engine, monkeypatch):
    exc = [FakeApiError("Connection error.")]
    _failing_asr(monkeypatch, exc)
    first = engine.render_take(_spec())
    Path(first.path).write_bytes(b"ID3 overwritten elsewhere")
    exc[0] = None
    engine.cached = engine._load_ledger()
    assert engine.render_take(_spec()).qa_pass and len(engine.el.calls) == 2


def test_asr_quota_raises_after_the_row(engine, monkeypatch):
    _failing_asr(monkeypatch, [FakeApiError("Error code: 429 - insufficient_quota", status_code=429)])
    with pytest.raises(T.AsrUnavailable, match="insufficient_quota"):
        engine.render_take(_spec())
    row = read_tsv(T.LEDGER)[-1]
    assert row["error"].startswith("asr:") and row["md5"] and Path(row["path"]).is_file()
    with pytest.raises(T.AsrUnavailable):
        engine.render_take(_spec(take=2))                            # no new paid take
    assert len(engine.el.calls) == 1


def test_asr_error_streak(engine, monkeypatch):
    monkeypatch.setattr(T, "ASR_ERROR_STREAK", 2)
    _failing_asr(monkeypatch, [FakeApiError("503 Service Unavailable", status_code=503)])
    assert engine.render_take(_spec(take=1)).error.startswith("asr:")
    with pytest.raises(T.AsrUnavailable, match="2 errors in a row"):
        engine.render_take(_spec(take=2))


def test_sanity_failure_is_not_rebought(engine):
    """A runaway take (0.6 s vs a 0.25 s v3 clip) saves no MP3. The same verdict is
    served from the ledger on every later run; only a spec under which today's rules
    pass it (a longer v3 clip) renders it again."""
    first = engine.render_take(_spec(v3_duration_s=0.25))
    assert first.sanity == ["dur:vs_v3_2.40x"] and not first.path
    engine.cached = engine._load_ledger()
    again = engine.render_take(_spec(v3_duration_s=0.25))
    assert again.sanity == first.sanity and len(engine.el.calls) == 1
    assert len(read_tsv(T.LEDGER)) == 1                              # unchanged verdict, no new row
    ok = engine.render_take(_spec(v3_duration_s=0.5))
    assert ok.qa_pass and len(engine.el.calls) == 2


def test_sanity_now_rechecks_stored_issues(monkeypatch):
    row = {"sanity": "edge:lead_0.80s|dur:vs_v3_2.40x", "duration_s": "0.6"}
    assert T.sanity_now(row, _spec(v3_duration_s=0.25)) == ["edge:lead_0.80s", "dur:vs_v3_2.40x"]
    assert T.sanity_now(row, _spec(v3_duration_s=0.5)) == ["edge:lead_0.80s"]
    import build.lib.audio_qa as QA
    monkeypatch.setattr(QA, "MAX_EDGE_SILENCE_S", 1.0)
    assert T.sanity_now(row, _spec(v3_duration_s=0.5)) == []
    assert T.sanity_now({"sanity": "pcm:silent", "duration_s": "0.6"}, _spec()) == ["pcm:silent"]


def test_tts_unavailable_propagates_without_a_row(engine):
    from build.lib.elevenlabs_client import TTSUnavailable

    def refuse(**kw):
        raise TTSUnavailable("ElevenLabs HTTP 401: invalid_api_key")
    engine.el.generate_pcm_meta = refuse
    with pytest.raises(TTSUnavailable):
        engine.render_take(_spec())
    assert not T.LEDGER.exists()


# ── ElevenLabs client: account-wide refusals ─────────────────────────────────
def _el_client(monkeypatch, status, body="", attempts=6):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-dummy")
    from elevenlabs.core.api_error import ApiError

    from build.lib import elevenlabs_client as E
    c = E.ElevenLabsClient(model_id=T.V4, max_attempts=attempts)
    calls = []

    def fail(**kw):
        calls.append(kw)
        raise ApiError(status_code=status, headers={}, body=body)
    monkeypatch.setattr(c, "_call_once", fail)
    monkeypatch.setattr(c, "_compute_wait", lambda exc, attempt: 0)
    return E, c, calls


@pytest.mark.parametrize("status", [401, 402, 403])
def test_elevenlabs_account_errors_raise_tts_unavailable(monkeypatch, status):
    E, c, calls = _el_client(monkeypatch, status, body={"detail": {"status": "invalid_api_key"}})
    with pytest.raises(E.TTSUnavailable, match=f"HTTP {status}"):
        c.generate_pcm(text="oi", voice_id="v", sense_id="s", clip_type="word", version=1)
    assert len(calls) == 1


def test_elevenlabs_sustained_429_raises_tts_unavailable(monkeypatch):
    E, c, calls = _el_client(monkeypatch, 429, body="too_many_concurrent_requests", attempts=3)
    with pytest.raises(E.TTSUnavailable, match="429 after 3 attempts"):
        c.generate_pcm(text="oi", voice_id="v", sense_id="s", clip_type="word", version=1)
    assert len(calls) == 3


def test_elevenlabs_clip_errors_stay_plain(monkeypatch):
    """A 400 (bad text) or a 5xx that outlasts the retries may be clip-specific: it is
    recorded on the take, not raised as an outage. Quota stays QuotaExceeded."""
    from elevenlabs.core.api_error import ApiError
    E, c, _ = _el_client(monkeypatch, 400)
    with pytest.raises(ApiError):
        c.generate_pcm(text="oi", voice_id="v", sense_id="s", clip_type="word", version=1)
    E, c, calls = _el_client(monkeypatch, 503, attempts=2)
    with pytest.raises(ApiError) as e:
        c.generate_pcm(text="oi", voice_id="v", sense_id="s", clip_type="word", version=1)
    assert not isinstance(e.value, E.TTSUnavailable) and len(calls) == 2
    assert issubclass(E.QuotaExceeded, E.TTSUnavailable)


# ── Gemini judge: request deadline ───────────────────────────────────────────
def _gemini(monkeypatch, outcomes):
    """A judge client whose generate_content plays `outcomes` (exception or JSON text)."""
    from build.lib import gemini_audio_judge as G
    c = G.GeminiAudioJudgeClient(api_key="test-dummy", model="gemini-3.8-flash")
    calls = []

    def generate_content(**kw):
        calls.append(kw)
        o = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(o, Exception):
            raise o
        return types.SimpleNamespace(text=o, usage_metadata=types.SimpleNamespace(
            prompt_token_count=1, candidates_token_count=1, thoughts_token_count=0))
    c.client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(G, "_backoff", lambda attempt: 0)
    return G, c, calls


def _judge_kw():
    return dict(audio_bytes=b"ID3", pt="verde", ipa_word_final="", voice_id="v", sense_id="s")


def test_gemini_client_has_a_request_timeout(monkeypatch):
    from build.lib import gemini_audio_judge as G
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr(G.genai, "Client", FakeClient)
    c = G.GeminiAudioJudgeClient(api_key="test-dummy")
    assert seen["http_options"].timeout == G.JUDGE_TIMEOUT_MS
    assert isinstance(c.client, FakeClient)                          # referenced, not dropped


def test_gemini_timeout_is_retried_then_raised(monkeypatch):
    import httpx
    verdict = ('{"pronunciation_verdict": "bp_ok", "drift": "none", "severity": "low", '
               '"confidence": "high", "evidence": "ok"}')
    timeout = httpx.ReadTimeout("The read operation timed out")
    G, c, calls = _gemini(monkeypatch, [timeout, timeout, verdict])
    assert c.judge(**_judge_kw()).pronunciation_verdict == "bp_ok" and len(calls) == 3
    G, c, calls = _gemini(monkeypatch, [timeout])
    with pytest.raises(httpx.ReadTimeout):
        c.judge(**_judge_kw(), max_attempts=3)
    assert len(calls) == 3


def test_gemini_402_is_not_retried(monkeypatch):
    from google.genai import errors
    exc = errors.ClientError(402, {"error": {"code": 402, "status": "RESOURCE_EXHAUSTED",
                                             "message": "Your prepayment credits are depleted."}})
    G, c, calls = _gemini(monkeypatch, [exc])
    with pytest.raises(errors.ClientError):
        c.judge(**_judge_kw())
    assert len(calls) == 1 and T.unavailable_reason(exc) == "HTTP 402"


# ── one engine process at a time ─────────────────────────────────────────────
def test_engine_lock_refuses_a_second_process(tmp_path, monkeypatch):
    import os
    import subprocess
    lock = tmp_path / "cache" / "v4_engine.lock"
    monkeypatch.setattr(T, "ENGINE_LOCK", lock)
    monkeypatch.setattr(T, "_engine_lock_fd", None)
    lock.parent.mkdir()
    holder = subprocess.Popen(
        [sys.executable, "-c", "import fcntl, sys; f = open(sys.argv[1], 'a'); "
         "fcntl.flock(f, fcntl.LOCK_EX); print('locked', flush=True); sys.stdin.read()", str(lock)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="v4_engine.lock"):
            T.acquire_engine_lock()
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)
        holder.stdout.close()
    T.acquire_engine_lock()                                          # free now
    fd = T._engine_lock_fd
    T.acquire_engine_lock()                                          # same process: reused
    assert T._engine_lock_fd == fd and f"pid {os.getpid()}" in lock.read_text()
    os.close(fd)


def test_engine_takes_the_lock_before_any_client(monkeypatch):
    def locked():
        raise RuntimeError("held elsewhere")
    monkeypatch.setattr(T, "acquire_engine_lock", locked)
    with pytest.raises(RuntimeError, match="held elsewhere"):
        T.Engine(judge_configs=[], asr_model="gpt-4o-transcribe")


# ── full-run ladder / manifest with a fake engine ────────────────────────────
W, E = ("0001.00.01", "word"), ("0001.00.01", "example")
VOICE, SWAP1, SWAP2 = "Rw38T6bn0lTNOb1aUevR", "m151rjrbWXbBqyq56tly", "PznTnBc8X6pvixs9UkQm"
POLICY = {"go": "yes", "word_variant_order": "plain", "stability": "0.65", "dict_on": "no",
          "short_words": "keep_v3"}


def _job(sid="0001.00.01", ct="word", voice=VOICE, order=1):
    return D.ClipJob(sense_id=sid, clip_type=ct, display_text="o livro" if ct == "word"
                     else "O livro está na mesa.", voice_id=voice, voice_gender="female",
                     expected_ipa="u ˈlivɾu", spoken_reference="", rank=10, manual_pass=False,
                     v3_object_key=f"audio/{sid}-{'word' if ct == 'word' else 'ex'}-eleven_v3-v1.mp3",
                     v3_model="eleven_v3", order=order)


def _take(spec, outcome, tmp_path):
    """outcome: True (passes QA), False (a judge says non_bp), "judge_error" / "asr_error"
    (no verdict, MP3 kept), "tts_error" (no audio), "tampered" (passes, but the MP3 on
    disk no longer matches its md5)."""
    if outcome == "tts_error":
        return T.TakeResult(spec=spec, error="tts:ApiError: status_code: 503")
    p = tmp_path / "takes" / f"{spec.take_id.replace('|', '_')}.mp3"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"ID3" + spec.take_id.encode())
    ok = outcome is True or outcome == "tampered"
    verdict = {"verdict": "error", "error": "ServerError: 503"} if outcome == "judge_error" \
        else {"verdict": "bp_ok" if ok else "non_bp"}
    judges = {} if outcome == "asr_error" else {"j": verdict}
    md5 = "0" * 32 if outcome == "tampered" else hashlib.md5(p.read_bytes()).hexdigest()
    error = "asr:APIConnectionError: Connection error." if outcome == "asr_error" else ""
    return T.TakeResult(spec=spec, path=str(p), md5=md5, lufs=-16.0, true_peak=-1.8,
                        asr_pass=outcome != "asr_error", asr_mode="production", asr_similarity=0.99,
                        judges=judges, gate_pass=ok, error=error,
                        generated_at="2026-10-01T00:00:00+00:00")


class FakeEngine:
    def __init__(self, tmp_path, outcomes):
        self.tmp_path, self.outcomes = tmp_path, outcomes
        self.rendered = []
        self.credits_used = 0
        self.judge_configs = []

    def render_take(self, spec):
        self.rendered.append(spec)
        out = self.outcomes(spec)
        if isinstance(out, BaseException):
            raise out
        return _take(spec, out, self.tmp_path)


class FakeR2:
    """Listing (ETag = md5 of a single-part PUT), HEAD with metadata, PUT."""

    def __init__(self):
        self.objects = {}                         # key -> {"body", "meta", "etag"}
        self.puts = []
        self.config = types.SimpleNamespace(bucket="b")
        self._s3 = types.SimpleNamespace(list_objects_v2=self._list)

    def _list(self, Bucket, Prefix):  # noqa: N803 — boto3's keyword names
        return {"Contents": [{"Key": k, "ETag": f'"{o["etag"]}"'}
                             for k, o in self.objects.items() if k.startswith(Prefix)]}

    def head(self, key):
        o = self.objects.get(key)
        return None if o is None else {"ETag": f'"{o["etag"]}"', "Metadata": dict(o["meta"])}

    def upload_bytes(self, body, key, *, extra_metadata=None):
        self.objects[key] = {"body": body, "meta": extra_metadata or {},
                             "etag": hashlib.md5(body).hexdigest()}
        self.puts.append(key)
        return types.SimpleNamespace(url=self.public_url(key))

    def public_url(self, key):
        return f"https://r2.example/{key}"


@pytest.fixture
def full(tmp_path, monkeypatch):
    """19_4 wired to fakes: its data/, reports/ and tmp/ paths under tmp_path (a 4-clip
    manifest + one en_ex row), a fake engine and R2, one worker. `full.run(outcomes)`
    builds a Run through the real __init__; `outcomes(spec)` → a `_take` outcome or an
    exception to raise."""
    data = tmp_path / "data"
    data.mkdir()
    for name in ("MANIFEST", "BACKUP", "PROVENANCE", "UNRESOLVED"):
        monkeypatch.setattr(R, name, data / getattr(R, name).name)
    monkeypatch.setattr(R, "REPORT", tmp_path / "reports" / R.REPORT.name)
    monkeypatch.setattr(R, "SMOKE_MANIFEST", tmp_path / "tmp" / "19_4_smoke" / R.MANIFEST.name)
    jobs = [_job(), _job(ct="example"), _job("0002.00.01", order=2), _job("0003.00.01", order=3)]
    rows = [{"sense_id": j.sense_id, "clip_type": j.clip_type, "tts_model": "eleven_v3",
             "voice_id": j.voice_id, "object_key": j.v3_object_key, "version": "1",
             "status": "uploaded"} for j in jobs]
    rows.append({"sense_id": "0001.00.01", "clip_type": "en_ex", "tts_model": "eleven_v3",
                 "voice_id": "EN", "object_key": "audio/0001.00.01-en_ex-eleven_v3-v1.mp3",
                 "version": "1", "status": "uploaded"})
    R.write_manifest(rows, R.MANIFEST)
    monkeypatch.setattr(D, "load_jobs", lambda: {j.key: j for j in jobs})
    monkeypatch.setattr(D, "swap_pool", lambda g: [VOICE, SWAP1, SWAP2])
    monkeypatch.setattr(R, "risky_words", set)
    monkeypatch.setattr(R, "WORKERS", 1)
    monkeypatch.setattr(R.Run, "_poll_budget", lambda self: None)
    monkeypatch.setattr(T, "load_judge_configs", lambda: [])
    monkeypatch.setattr(T, "load_asr_model", lambda: "fake-asr")
    monkeypatch.setattr(T, "mp3_duration", lambda p: 0.8)
    import build.lib.r2_client as RC
    state = types.SimpleNamespace(outcomes=lambda s: True, r2=FakeR2(), engine=None)
    monkeypatch.setattr(RC, "R2Client", lambda cfg: state.r2)
    monkeypatch.setattr(RC.R2Config, "from_env", classmethod(lambda cls: None))

    def engine(**kw):
        state.engine = FakeEngine(tmp_path, lambda s: state.outcomes(s))
        return state.engine
    monkeypatch.setattr(T, "Engine", engine)

    def run(outcomes=None, *, no_upload=False, manifest_out=""):
        if outcomes is not None:
            state.outcomes = outcomes
        args = types.SimpleNamespace(limit=0, no_upload=no_upload, manifest_out=manifest_out,
                                     concurrency=2)
        return R.Run(args, dict(POLICY))
    state.run = run
    return state


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["19_4_render_v4.py", *argv])
    monkeypatch.setattr(T, "acquire_engine_lock", lambda: None)
    monkeypatch.setattr(R, "load_policy", lambda: dict(POLICY))
    monkeypatch.setattr(R.signal, "signal", lambda *a: None)
    return R.main()


def _rows(path):
    return {(r["sense_id"], r["clip_type"]): r for r in read_tsv(path)}


def _keys(specs):
    return [(s.sense_id, s.clip_type) for s in specs]


def _row_keys(path):
    return [(r["sense_id"], r["clip_type"]) for r in read_tsv(path)]


def test_word_best_of_two_accepts_and_writes_row(full):
    run = full.run(no_upload=True)
    run.process(run.jobs[W])
    assert [s.take for s in full.engine.rendered] == [1, 2]          # best-of-2, no more
    row = run.idx[W]
    assert row["tts_model"] == "eleven_v4" and row["object_key"].endswith("-eleven_v4-v1.mp3")
    assert row["notes"] == "stage_19_v4:plain"
    assert run.idx[("0001.00.01", "en_ex")] == run.en_ex_before[0]


def test_unresolved_keeps_v3_row_identical(full):
    run = full.run(lambda s: False, no_upload=True)
    before = dict(run.idx[W])
    run.process(run.jobs[W])
    assert run.idx[W] == before
    rendered = full.engine.rendered
    assert len(rendered) == 6            # 2 + 2 takes + 2 voice swaps (cap 8)
    assert {s.voice_id for s in rendered[4:]} == {SWAP1, SWAP2}
    assert _row_keys(run.unresolved_path) == [W]


def test_example_ladder_cap(full):
    run = full.run(lambda s: False, no_upload=True)
    run.process(run.jobs[E])
    assert len(full.engine.rendered) == 4            # t1, t2, 2 swaps (cap 5)


def test_swap_accepts_new_voice(full):
    run = full.run(lambda s: s.voice_id == SWAP1, no_upload=True)
    run.process(run.jobs[W])
    assert run.idx[W]["voice_id"] == SWAP1
    assert run.stats["accepted_after_voice_swap"] == 1


def test_versions_monotonic_per_clip(full):
    run = full.run(no_upload=True)
    j = run.jobs[W]
    run.versions_used[j.key] = 3
    assert run._version(j, "a" * 32) == (4, False)
    run = full.run()                                                  # with R2: listing + registry
    for v in (1, 5):
        full.r2.upload_bytes(b"older take %d" % v, f"audio/0001.00.01-word-eleven_v4-v{v}.mp3")
    run.versions_used[j.key] = 3
    assert run._version(j, "a" * 32) == (6, False)


def test_budget_stop_halts_without_writing(full):
    run = full.run(no_upload=True)
    run.stop.set()
    before = dict(run.idx[W])
    run.process(run.jobs[W])
    assert full.engine.rendered == [] and run.idx[W] == before


# ── outages: stop the climb, then the run ────────────────────────────────────
@pytest.mark.parametrize("bad", ["judge_error", "asr_error", "tts_error"])
def test_no_verdict_take_stops_the_climb_and_leaves_the_clip_pending(full, bad):
    """Take 1 fails QA, take 2 gets no verdict: no takes 3–4 and no voice swap (a swap
    voice would pass here), no unresolved row. The next run re-checks it."""
    run = full.run(lambda s: True if s.voice_id == SWAP1 else (bad if s.take == 2 else False),
                   no_upload=True)
    before = dict(run.idx[W])
    run.process(run.jobs[W])
    assert [(s.voice_id, s.take) for s in full.engine.rendered] == [(VOICE, 1), (VOICE, 2)]
    assert run.idx[W] == before and run.stats["pending"] == 1 and not run.stats["unresolved"]
    assert not run.unresolved_path.exists() and not run.stop.is_set()


@pytest.mark.parametrize("kind,code", [("judge", R.EXIT_OUTAGE), ("asr", R.EXIT_OUTAGE),
                                       ("tts", R.EXIT_OUTAGE), ("quota", R.EXIT_BUDGET)])
def test_provider_outage_stops_the_whole_run(full, kind, code):
    from build.lib import elevenlabs_client as EL
    exc = {"judge": T.JudgeUnavailable("judge:j1: HTTP 402"),
           "asr": T.AsrUnavailable("asr: insufficient_quota"),
           "tts": EL.TTSUnavailable("ElevenLabs HTTP 401: invalid_api_key"),
           "quota": EL.QuotaExceeded("quota_exceeded")}[kind]
    run = full.run(lambda s: exc, no_upload=True)
    for k in (W, E):
        run.process(run.jobs[k])
    assert run.exit_code == code and run.stop.is_set()
    assert len(full.engine.rendered) == 1                             # nothing after the outage
    assert run.idx[W]["tts_model"] == "eleven_v3" and not run.unresolved_path.exists()


def test_pending_streak_stops_the_run(full, monkeypatch):
    """TTS 5xx / timeouts that outlast the client's retries have no engine streak:
    PENDING_STREAK clips in a row without a verdict stop the run as an outage."""
    monkeypatch.setattr(R, "PENDING_STREAK", 2)
    run = full.run(lambda s: "tts_error", no_upload=True)
    for k in (W, E, ("0002.00.01", "word")):
        run.process(run.jobs[k])
    assert run.exit_code == R.EXIT_OUTAGE and "2 clips in a row" in run.stop_reason
    assert len(full.engine.rendered) == 2


def test_finished_run_with_pending_clips_exits_6(full, monkeypatch):
    full.outcomes = lambda s: "judge_error" if s.clip_type == "example" else True
    assert _main(monkeypatch, "--yes", "--no-upload") == R.EXIT_PENDING == 6


# ── smoke runs never touch the real manifest or data files ───────────────────
def test_smoke_run_leaves_real_files_byte_identical(full, monkeypatch):
    R.write_tsv(R.PROVENANCE, [{"object_key": "audio/x.mp3", "sense_id": "9999.00.01"}],
                fieldnames=R.PROVENANCE_FIELDS)
    R.write_tsv(R.UNRESOLVED, [{"sense_id": "0002.00.01", "clip_type": "word"}],
                fieldnames=R.UNRESOLVED_FIELDS)
    R.REPORT.parent.mkdir()
    R.REPORT.write_text("previous report")
    real = (R.MANIFEST, R.PROVENANCE, R.UNRESOLVED, R.REPORT)
    before = {p: p.read_bytes() for p in real}
    full.outcomes = lambda s: s.clip_type == "word"                   # 3 accepts, 1 unresolved
    assert _main(monkeypatch, "--yes", "--no-upload") == 0
    assert {p: p.read_bytes() for p in real} == before and not R.BACKUP.exists()
    smoke = R.SMOKE_MANIFEST
    assert _rows(smoke)[W]["tts_model"] == "eleven_v4"
    assert len(read_tsv(smoke.with_name("_audio_manifest.provenance.tsv"))) == 3
    assert _row_keys(smoke.with_name("_audio_manifest.unresolved.tsv")) == [E]
    assert smoke.with_name("_audio_manifest.report.html").is_file()


def test_no_upload_refuses_the_real_manifest_before_the_lock(full, monkeypatch):
    monkeypatch.setattr(T, "acquire_engine_lock", lambda: pytest.fail("lock taken"))
    out = str(R.MANIFEST.parent / ".." / "data" / R.MANIFEST.name)   # same file, other spelling
    monkeypatch.setattr(sys, "argv", ["19_4", "--yes", "--no-upload", "--manifest-out", out])
    with pytest.raises(R.RenderError, match="never writes the real manifest"):
        R.main()


def test_real_run_backs_up_before_the_first_manifest_write(full, monkeypatch):
    original = R.MANIFEST.read_bytes()
    assert _main(monkeypatch, "--yes") == 0
    assert R.BACKUP.read_bytes() == original
    assert _rows(R.MANIFEST)[W]["url"] == \
        "https://r2.example/audio/0001.00.01-word-eleven_v4-v1.mp3"
    assert len(full.r2.puts) == 4 and len(read_tsv(R.PROVENANCE)) == 4


def test_flush_refuses_the_real_manifest_without_a_backup(full):
    run = full.run()
    original = R.MANIFEST.read_bytes()
    with pytest.raises(R.RenderError, match="bak_pre_stage19 missing"):
        run.flush()
    assert R.MANIFEST.read_bytes() == original


def test_backup_refuses_a_manifest_already_on_v4(full):
    rows = read_tsv(R.MANIFEST)
    rows[0]["tts_model"] = "eleven_v4"
    R.write_manifest(rows, R.MANIFEST)
    with pytest.raises(R.RenderError, match="not the pre-Stage-19 state"):
        R.backup_manifest()
    assert not R.BACKUP.exists()


def test_engine_lock_is_checked_before_any_work(full, monkeypatch):
    def held():
        raise RuntimeError("another Stage 19 render engine holds .../v4_engine.lock (pid 1)")
    monkeypatch.setattr(T, "acquire_engine_lock", held)
    monkeypatch.setattr(R, "load_policy", lambda: pytest.fail("work before the lock"))
    monkeypatch.setattr(sys, "argv", ["19_4", "--dry-run"])
    with pytest.raises(R.RenderError, match="v4_engine.lock"):
        R.main()


# ── worker exceptions, Ctrl-C, resume ────────────────────────────────────────
def test_worker_exception_stops_the_run_and_flushes(full, monkeypatch):
    """The 2nd upload fails: the run stops (the 3rd and 4th clips are never rendered),
    the 1st accept is flushed to the manifest, and the exit code is EXIT_ERROR."""
    put = full.r2.upload_bytes

    def flaky(body, key, **kw):
        if full.r2.puts:
            raise RuntimeError(f"R2 upload failed for {key}: 503")
        return put(body, key, **kw)
    full.r2.upload_bytes = flaky
    assert _main(monkeypatch, "--yes") == R.EXIT_ERROR
    assert set(_keys(full.engine.rendered)) == {E, W}                 # study order: E, then W
    rows = _rows(R.MANIFEST)
    assert rows[E]["tts_model"] == "eleven_v4" and rows[W]["tts_model"] == "eleven_v3"
    assert R.BACKUP.exists() and not R.UNRESOLVED.exists()


def test_ctrl_c_drops_queued_clips_and_flushes(full, monkeypatch):
    """Ctrl-C (or SIGTERM) after the 1st clip, while the 2nd is mid-take: that take
    finishes, no further take or clip is bought, and the 1st accept is flushed."""
    run = full.run()
    R.backup_manifest()
    scope = run.scope()

    def outcomes(s):
        if (s.sense_id, s.clip_type) == scope[1].key:
            assert run.stop.wait(5)
            return False
        return True
    full.outcomes = outcomes

    def interrupted(futs):
        futs[0].result()
        yield futs[0]
        raise KeyboardInterrupt
    monkeypatch.setattr(R, "as_completed", interrupted)
    R.drive(run, scope, 0.0)
    assert run.exit_code == R.EXIT_INTERRUPTED
    assert _keys(full.engine.rendered) == [scope[0].key, scope[1].key]
    assert _rows(R.MANIFEST)[scope[0].key]["tts_model"] == "eleven_v4"
    assert _rows(R.MANIFEST)[scope[1].key]["tts_model"] == "eleven_v3"


def test_unflushed_upload_is_reused_not_uploaded_again(full):
    """A run uploaded v1 and died before flushing it. The resumed run picks the same
    winner from the cache, finds those bytes on R2 (ETag, else the md5 metadata) and
    reuses v1: no v2, no orphan."""
    key = "audio/0001.00.01-word-eleven_v4-v1.mp3"
    run = full.run()
    run.process(run.jobs[W])
    assert full.r2.puts == [key] and run.idx[W]["object_key"] == key      # never flushed
    run = full.run()                                                      # resume
    run.process(run.jobs[W])
    assert full.r2.puts == [key] and run.idx[W]["object_key"] == key
    assert run.stats["accepted_reused_unflushed_upload"] == 1
    full.r2.objects[key]["etag"] = "3858f62230ac3c915f300c664312c11f-2"   # a multipart ETag
    run = full.run()
    run.process(run.jobs[W])
    assert full.r2.puts == [key] and run.idx[W]["url"] == f"https://r2.example/{key}"


def test_new_winner_after_an_unflushed_upload_gets_the_next_version(full):
    run = full.run()
    run.process(run.jobs[W])                                          # v1 = take 1
    run = full.run(lambda s: s.take == 2)                             # now only take 2 passes
    run.process(run.jobs[W])
    assert full.r2.puts == ["audio/0001.00.01-word-eleven_v4-v1.mp3",
                            "audio/0001.00.01-word-eleven_v4-v2.mp3"]


def test_take_changed_on_disk_is_not_uploaded(full):
    run = full.run(lambda s: "tampered")
    run.process(run.jobs[E])
    assert run.exit_code == R.EXIT_ERROR and "no longer matches" in run.stop_reason
    assert full.r2.puts == [] and run.idx[E]["tts_model"] == "eleven_v3"


def test_unresolved_is_current_state_across_resumes(full):
    """One row per clip, rewritten: an older append-only file collapses, a resume that
    fails again doesn't duplicate, a pending resume keeps the row, an accept removes it."""
    old = {"sense_id": W[0], "clip_type": W[1], "attempts": "6", "v3_kept": "yes"}
    R.write_tsv(R.UNRESOLVED, [old, dict(old, attempts="7"), dict(old, clip_type="example")],
                fieldnames=R.UNRESOLVED_FIELDS)
    for outcome in (False, False, "judge_error"):
        run = full.run(lambda s, o=outcome: o)
        run.process(run.jobs[W])
        assert _row_keys(R.UNRESOLVED) == [W, E]
    run = full.run(lambda s: True)
    run.process(run.jobs[W])
    assert _row_keys(R.UNRESOLVED) == [E]
    assert read_tsv(R.UNRESOLVED)[0]["attempts"] == "6"               # untouched clip as it was


@pytest.mark.parametrize("go,gate_ok,same_gate,ok", [
    ("yes", "yes", True, True), ("yes", "PENDING", True, False), ("yes", "NO", True, False),
    ("PENDING", "yes", True, False), ("yes", "yes", False, False)])
def test_load_policy_needs_go_gate_ok_and_the_pinned_gate(tmp_path, monkeypatch, go, gate_ok,
                                                          same_gate, ok):
    pol = tmp_path / "policy.tsv"
    gate = T.judge_hash(T.load_judge_configs()) if same_gate else "0ldc0nf1g0"
    write_tsv(pol, [{"key": "go", "value": go, "evidence": ""},
                    {"key": "gate_ok", "value": gate_ok, "evidence": ""},
                    {"key": "judge_hash", "value": gate, "evidence": ""}],
              fieldnames=["key", "value", "evidence"])
    monkeypatch.setattr(R, "POLICY", pol)
    if ok:
        assert R.load_policy()["gate_ok"] == "yes"
    else:
        with pytest.raises(R.RenderError):
            R.load_policy()


# ── scoped judges (Flash ×2 everywhere, Pro on risky words) + verdict reuse ──
def _scoped(engine):
    cfgs = [{"config": "flash38-J2", "backend": "openai", "model": "f", "prompt_version": "J2", "scope": "all"},
            {"config": "flash38-J1p", "backend": "openai", "model": "f", "prompt_version": "J1p", "scope": "all"},
            {"config": "pro-J2", "backend": "openai", "model": "p", "prompt_version": "J2", "scope": "risky_word"}]
    engine.judge_configs = cfgs
    engine.judge_hash = T.judge_hash(cfgs)
    engine.judges = [(c, FakeJudge()) for c in cfgs]
    return {c["config"]: j for c, j in engine.judges}


def test_pro_judges_only_risky_word_clips(engine):
    js = _scoped(engine)
    plain = engine.render_take(_spec())
    risky = engine.render_take(_spec(take=2, risky=True))
    ex = engine.render_take(_spec(take=3, clip_type="example", display_text="O verde é bonito.",
                                  risky=True))
    assert set(plain.judges) == {"flash38-J2", "flash38-J1p"} and plain.qa_pass
    assert set(risky.judges) == {"flash38-J2", "flash38-J1p", "pro-J2"} and risky.qa_pass
    assert "pro-J2" not in ex.judges
    assert len(js["pro-J2"].seen) == 1


def test_gate_change_reuses_bought_verdicts(engine):
    """Judged under Pro+Flash J2; the new gate only buys the missing Flash J1p."""
    old = [{"config": "pro-J2", "backend": "openai", "model": "p", "prompt_version": "J2"},
           {"config": "flash38-J2", "backend": "openai", "model": "f", "prompt_version": "J2"},
           {"config": "flash-swapped", "backend": "openai", "model": "x", "prompt_version": "J1p"}]
    engine.judge_configs, engine.judge_hash = old, T.judge_hash(old)
    engine.judges = [(c, FakeJudge()) for c in old]
    engine.render_take(_spec())
    js = _scoped(engine)
    engine.cached = engine._load_ledger()
    res = engine.render_take(_spec())
    assert len(engine.el.calls) == 1                                  # no new TTS
    assert [len(js[n].seen) for n in ("flash38-J2", "flash38-J1p", "pro-J2")] == [0, 1, 0]
    assert set(res.judges) == {"flash38-J2", "flash38-J1p"}           # Pro dropped: not risky
    assert res.judge_hash == engine.judge_hash and res.qa_pass


def test_judge_hash_includes_scope():
    a = {"config": "pro-J2", "backend": "gemini", "model": "p", "prompt_version": "J2"}
    assert T.judge_hash([a]) == T.judge_hash([dict(a, scope="all")])
    assert T.judge_hash([a]) != T.judge_hash([dict(a, scope="risky_word")])


def test_verdict_reuse_needs_the_same_model_and_prompt():
    c = {"config": "flash38-J2", "model": "gemini-3.8-flash", "prompt_version": "J2"}
    assert T.reusable_verdict(c, {"verdict": "bp_ok"})                        # legacy name
    assert not T.reusable_verdict(dict(c, model="gemini-4-flash"), {"verdict": "bp_ok"})
    assert not T.reusable_verdict(c, {"verdict": "error"})
    assert T.reusable_verdict(dict(c, config="new"), {"verdict": "non_bp", "model": "gemini-3.8-flash",
                                                       "prompt_version": "J2"})
    assert not T.reusable_verdict(dict(c, config="new"), {"verdict": "bp_ok", "model": "gemini-3.8-flash",
                                                           "prompt_version": "J1p"})


def test_overwritten_ledger_cache_keeps_flags(engine):
    """_append caches the row as the TSV stores it: a second cache hit in the same
    process must not turn asr_pass/gate_pass "1"-vs-1 into False."""
    first = engine.render_take(_spec())
    engine.render_take(_spec())
    again = engine.render_take(_spec())
    assert first.qa_pass and again.asr_pass and again.gate_pass and again.qa_pass
    assert len(engine.el.calls) == 1


def test_narrower_gate_keeps_older_verdicts_reusable(engine):
    """Going Pro+Flash -> Flash-only -> Pro+Flash again buys nothing twice: the ledger
    pool still holds the Pro verdict the narrower gate's row dropped."""
    wide = [{"config": "pro-J2", "backend": "openai", "model": "p", "prompt_version": "J2"},
            {"config": "flash38-J2", "backend": "openai", "model": "f", "prompt_version": "J2"}]
    narrow = [wide[1]]
    def use(cfgs):
        engine.judge_configs, engine.judge_hash = cfgs, T.judge_hash(cfgs)
        engine.judges = [(c, FakeJudge()) for c in cfgs]
        engine.cached = engine._load_ledger()
        return {c["config"]: j for c, j in engine.judges}
    use(wide); engine.render_take(_spec())
    use(narrow); assert set(engine.render_take(_spec()).judges) == {"flash38-J2"}
    js = use(wide)
    res = engine.render_take(_spec())
    assert set(res.judges) == {"pro-J2", "flash38-J2"} and not js["pro-J2"].seen
    assert len(engine.el.calls) == 1


def test_unknown_judge_scope_is_refused(tmp_path):
    cfg = tmp_path / "models.tsv"
    write_tsv(cfg, [{"role": "judge", "config": "x", "backend": "gemini", "model": "m",
                     "prompt_version": "J2", "combine": "AND", "scope": "risky_words"}],
              fieldnames=["role", "config", "backend", "model", "prompt_version", "combine", "scope"])
    with pytest.raises(RuntimeError, match="unknown judge scope"):
        T.load_judge_configs(cfg)


def test_pinned_config_scopes():
    cfgs = {c["config"]: T.judge_scope(c) for c in T.load_judge_configs()}
    assert cfgs == {"flash38-J2": "all", "pro-J2": "risky_word"}
