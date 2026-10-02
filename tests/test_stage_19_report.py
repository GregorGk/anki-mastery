"""Stage 19 — QA report + residue page (build/19_5_qa_report.py) on synthetic data.

The takes ledger is append-only: a re-judge or ASR recheck appends a row with the
same take_id, md5, credits and generated_at as the render; a genuine re-render of
a take_id gets a new generated_at. The report counts credits and renders once per
(take_id, generated_at), takes verdicts from the latest row per take_id, and keeps
judge errors out of the failure reasons. _v4_unresolved.tsv and the provenance file
are read as current state against the manifest, whether old runs appended
duplicates or 19_4 rewrites them keyed.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import stage19_data as D  # noqa: E402
from build.lib import v4_tts as T  # noqa: E402
from build.lib.tsv import write_tsv  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


Q = _load("stage19_5", REPO_ROOT / "build" / "19_5_qa_report.py")

VOICE = "Rw38T6bn0lTNOb1aUevR"
MAC_DIR = "/Users/someone/anki-mastery/build/audio_cache/v4_takes"   # ledger began on a Mac
OK = {"verdict": "bp_ok", "drift": "none"}
UNRES_OLD = ["sense_id", "clip_type", "voice_id", "attempts", "reasons", "v3_kept", "logged_at"]
PROV_FIELDS = ["object_key", "sense_id", "clip_type", "voice_id", "variant", "tts_input", "seed",
               "take", "stability", "dict_on", "judge_hash", "judges", "asr_mode",
               "selected_from", "generated_at"]
MAN_FIELDS = ["sense_id", "clip_type", "tts_model", "voice_id", "object_key", "status"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "REPO_ROOT", tmp_path)
    for name, file in [("MANIFEST", "manifest.tsv"), ("BACKUP", "manifest.bak"),
                       ("PROVENANCE", "prov.tsv"), ("UNRESOLVED", "unres.tsv"),
                       ("TAKES", "takes.tsv"), ("REPORT", "reports/qa.html"),
                       ("RESIDUE", "reports/residue.html")]:
        monkeypatch.setattr(Q, name, tmp_path / file)
    monkeypatch.setattr(T, "TAKES_DIR", tmp_path / "takes")
    monkeypatch.setattr(D, "ANKI_MEDIA", tmp_path / "anki_media")
    monkeypatch.setattr(D, "voice_names", lambda: {})
    (tmp_path / "takes").mkdir()
    (tmp_path / "anki_media").mkdir()
    return tmp_path


# ── synthetic files ──────────────────────────────────────────────────────────
def _take(sid, *, ct="word", take=1, credits=10, gen="2026-10-01T10:00:00+00:00", md5="m1",
          judges=None, mp3=True, **kw):
    """One ledger row; defaults to a take that passed every check."""
    take_id = f"{sid}|{ct}|plain|{VOICE}|t{take}|s0.65|d0"
    judges = {"gemini_pro": OK, "gemini_flash": OK} if judges is None else judges
    row = {f: "" for f in T.LEDGER_FIELDS}
    row.update(take_id=take_id, sense_id=sid, clip_type=ct, variant="plain", voice_id=VOICE,
               take=take, md5=md5, asr_pass=1, asr_similarity=0.99,
               judges=json.dumps(judges), credits=credits, generated_at=gen,
               path=f"{MAC_DIR}/{sid}-{ct}-t{take}.mp3" if mp3 else "",
               gate_pass=int(bool(judges) and all(v["verdict"] == "bp_ok" for v in judges.values())))
    row.update(kw)
    row["qa_pass"] = int(not row["error"] and not row["sanity"] and not row["loudness"]
                         and row["asr_pass"] == 1 and row["gate_pass"] == 1)
    return row


def _mp3(env, row, data=b"ID3fake-take"):
    (env / "takes" / Path(row["path"]).name).write_bytes(data)


def _man(sid, ct="word", model="eleven_v3", version=1):
    short = "word" if ct == "word" else "ex"
    return {"sense_id": sid, "clip_type": ct, "tts_model": model, "voice_id": VOICE,
            "object_key": f"audio/{sid}-{short}-{model}-v{version}.mp3", "status": "uploaded"}


def _unres(sid, ct="word", attempts=5, reasons="{'judge': 5}", logged="2026-10-02T09:00"):
    return {"sense_id": sid, "clip_type": ct, "voice_id": VOICE, "attempts": attempts,
            "reasons": reasons, "v3_kept": "yes", "logged_at": logged}


def _prov(man_row, selected_from=1, variant="plain"):
    return {"object_key": man_row["object_key"], "sense_id": man_row["sense_id"],
            "clip_type": man_row["clip_type"], "voice_id": VOICE, "variant": variant,
            "selected_from": selected_from, "generated_at": "2026-10-02T09:00"}


def _write(env, *, takes=(), manifest=(), unres=(), prov=(), unres_fields=UNRES_OLD):
    write_tsv(Q.TAKES, takes, fieldnames=T.LEDGER_FIELDS)
    write_tsv(Q.MANIFEST, manifest, fieldnames=MAN_FIELDS)
    write_tsv(Q.UNRESOLVED, unres, fieldnames=unres_fields)
    write_tsv(Q.PROVENANCE, prov, fieldnames=PROV_FIELDS)


def _cards(doc: str) -> dict[str, str]:
    return {k: v for v, k in re.findall(r"<div class='card'><b>(.*?)</b>(.*?)</div>", doc)}


def _section(doc: str, title: str) -> str:
    return doc.split(f"<h2>{title}</h2>", 1)[1].split("<h2>", 1)[0]


def _cells(section: str) -> list[list[str]]:
    return [re.findall(r"<td>(.*?)</td>", tr) for tr in re.findall(r"<tr>(.*?)</tr>", section)
            if "<td>" in tr]


# ── report: takes ledger ─────────────────────────────────────────────────────
def test_rejudged_rows_count_one_render_and_credits_once(env):
    err = {"gemini_pro": {"verdict": "error", "error": "402"}, "gemini_flash": OK}
    a1 = _take("0001.00.01", judges=err, credits=10)                    # judge outage
    a2 = _take("0001.00.01", credits=10)                                # re-judged: same render
    b1 = _take("0002.00.01", credits=7, sanity="dur:vs_v3_0.4x", mp3=False,
               gen="2026-10-01T11:00:00+00:00", md5="")                 # failed PCM sanity
    b2 = _take("0002.00.01", credits=7, gen="2026-10-01T12:00:00+00:00", md5="m2")  # re-render
    b3 = dict(b2, asr_mode="recheck:relaxed")                           # ASR recheck: same render
    _write(env, takes=[a1, a2, b1, b2, b3], manifest=[_man("0001.00.01"), _man("0002.00.01")])
    Q.report()
    doc = Q.REPORT.read_text(encoding="utf-8")
    cards = _cards(doc)
    assert cards["credits (header sum)"] == "24"                        # 10 + 7 + 7, not 41
    assert cards["takes rendered"] == "3"                               # not 5 rows
    assert cards["takes passing QA (latest verdict)"] == "2 / 2"
    assert cards["takes not judged (judge error)"] == "0"
    assert _cells(_section(doc, "Why takes failed")) == []              # superseded rows don't count


def test_rejudge_rows_with_zeroed_credits_still_count_the_render(env):
    """If cached_result ever writes credits=0 on re-judge rows, the render keeps its cost."""
    _write(env, takes=[_take("0001.00.01", credits=10), _take("0001.00.01", credits=0)])
    Q.report()
    assert _cards(Q.REPORT.read_text(encoding="utf-8"))["credits (header sum)"] == "10"


def test_latest_judge_error_is_not_judged_not_a_failure(env):
    err = {"gemini_pro": {"verdict": "error", "error": "429"}, "gemini_flash": OK}
    rejected = {"gemini_pro": {"verdict": "ep_drift", "drift": "vowel"}, "gemini_flash": OK}
    _write(env, takes=[
        _take("0001.00.01", judges=err),                                # still errored
        _take("0002.00.01", judges=rejected),                           # judged and rejected
        _take("0003.00.01", asr_pass=0, asr_similarity=0.4, judges={}),
        _take("0004.00.01"),
    ])
    Q.report()
    doc = Q.REPORT.read_text(encoding="utf-8")
    cards = _cards(doc)
    assert cards["takes not judged (judge error)"] == "1"
    assert cards["takes passing QA (latest verdict)"] == "1 / 3"
    fails = dict(_cells(_section(doc, "Why takes failed")))
    assert fails == {"judge:gemini_pro:ep_drift/vowel": "1", "asr": "1"}


@pytest.mark.parametrize("row,expected", [
    (_take("1", error="tts:Timeout"), ["error"]),
    (_take("1", sanity="dur:vs_v3_0.4x|silence"), ["sanity:dur:vs"]),
    (_take("1", loudness="lufs"), ["loudness"]),
    (_take("1", asr_pass=0), ["asr"]),
    (_take("1"), []),
    (_take("1", judges={"j": {"verdict": "ep_drift", "drift": "x"}, "k": OK}), ["judge:j:ep_drift/x"]),
    (_take("1", judges={"j": {"verdict": "error"}, "k": {"verdict": "ep_drift"}}), None),
    (_take("1", judges={}), None),                                      # judges never ran
])
def test_fail_reasons(row, expected):
    row = {k: str(v) for k, v in row.items()}
    assert Q._fail_reasons(row) == expected


# ── report: unresolved + provenance as current state ─────────────────────────
def test_unresolved_keeps_last_row_and_drops_clips_now_on_v4(env):
    _write(env, manifest=[_man("0001.00.01"), _man("0002.00.01", model=T.V4),
                          _man("0003.00.01", ct="example")],
           unres=[_unres("0001.00.01", attempts=5), _unres("0002.00.01"),
                  _unres("0003.00.01", ct="example"), _unres("0001.00.01", attempts=8)])
    Q.report()
    doc = Q.REPORT.read_text(encoding="utf-8")
    assert _cards(doc)["unresolved (v3 kept)"] == "2"
    rows = _cells(_section(doc, "Unresolved (v3 kept)"))
    assert [(r[0], r[2]) for r in rows] == [("0001.00.01", "8"), ("0003.00.01", "5")]


def test_unresolved_keyed_shape_with_other_columns(env):
    """A keyed current-state rewrite (one row per clip, its own columns) still reads."""
    _write(env, manifest=[_man("0001.00.01")],
           unres=[{"sense_id": "0001.00.01", "clip_type": "word", "voice_id": VOICE}],
           unres_fields=["sense_id", "clip_type", "voice_id"])
    Q.report()
    doc = Q.REPORT.read_text(encoding="utf-8")
    assert _cards(doc)["unresolved (v3 kept)"] == "1"
    assert _cells(_section(doc, "Unresolved (v3 kept)")) == [["0001.00.01", "word", "", ""]]


def test_provenance_counts_only_uploads_the_manifest_points_at(env):
    on_v4 = _man("0001.00.01", model=T.V4, version=1)
    still_v3 = _man("0002.00.01")
    smoke = _prov(_man("0002.00.01", model=T.V4), selected_from=2)      # --no-upload smoke row
    _write(env, manifest=[on_v4, still_v3],
           prov=[smoke, _prov(on_v4, selected_from=4), _prov(on_v4, selected_from=3)])
    Q.report()
    doc = Q.REPORT.read_text(encoding="utf-8")
    assert _cards(doc)["accepted uploads"] == "1"
    assert _cells(_section(doc, "Takes needed before acceptance")) == [["3", "1"]]


# ── residue page ─────────────────────────────────────────────────────────────
def _job(sid, man_row, order, ct="word"):
    return D.ClipJob(sense_id=sid, clip_type=ct, display_text=f"palavra {sid}", voice_id=VOICE,
                     voice_gender="female", expected_ipa="", spoken_reference="", rank=10,
                     manual_pass=False, v3_object_key=man_row["object_key"],
                     v3_model=man_row["tts_model"], order=order)


def test_residue_skips_accepted_duplicate_and_unknown_clips(env, monkeypatch, capsys):
    x, y = _man("0001.00.01"), _man("0002.00.01", model=T.V4)          # y accepted since
    (env / "anki_media" / x["object_key"].split("/", 1)[1]).write_bytes(b"ID3v3")
    jobs = {("0001.00.01", "word"): _job("0001.00.01", x, 2),
            ("0002.00.01", "word"): _job("0002.00.01", y, 1)}           # its v4 MP3 isn't local
    monkeypatch.setattr(D, "load_jobs", lambda: jobs)
    t = _take("0001.00.01", judges={"gemini_pro": {"verdict": "ep_drift"}})
    _mp3(env, t)
    _write(env, takes=[t], manifest=[x, y],
           unres=[_unres("0001.00.01"), _unres("0002.00.01"), _unres("0001.00.01"),
                  _unres("0009.00.01")])                                # 0009: not a BP job now
    monkeypatch.setattr(sys, "argv", ["19_5_qa_report.py", "--residue"])
    assert Q.main() == 0
    page = Q.RESIDUE.read_text(encoding="utf-8")
    assert page.count("class='card'") == 1 and "data-sid='0001.00.01'" in page
    assert "0002.00.01" not in page and "0009.00.01" not in page
    assert page.count("Keep v3") == 1 and page.count("v4 option") == 1
    assert "residue page: 1 clips" in capsys.readouterr().out


def test_residue_handles_missing_take_and_v3_mp3s(env, monkeypatch):
    x = _man("0001.00.01")                                              # v3 MP3 not fetched yet
    monkeypatch.setattr(D, "load_jobs", lambda: {("0001.00.01", "word"): _job("0001.00.01", x, 1)})
    rej = {"gemini_pro": {"verdict": "ep_drift"}, "gemini_flash": OK}
    gone = _take("0001.00.01", take=1, judges=rej)                     # best ASR, MP3 pruned
    t2 = _take("0001.00.01", take=2, judges=rej, asr_similarity=0.9)
    t3 = _take("0001.00.01", take=3, judges=rej, asr_similarity=0.95)
    t4_old = _take("0001.00.01", take=4, judges=rej)
    t4 = dict(t4_old, path="", sanity="dur:vs_v3_0.4x", generated_at="2026-10-02")  # latest: no MP3
    for row in (t2, t3, t4_old):
        _mp3(env, row, data=row["take_id"].encode())
    _write(env, takes=[gone, t2, t3, t4_old, t4], manifest=[x], unres=[_unres("0001.00.01")])
    Q.residue()
    page = Q.RESIDUE.read_text(encoding="utf-8")
    assert "MP3 missing: 0001.00.01-word-eleven_v3-v1.mp3" in page    # Keep v3 still offered
    offered = re.findall(r"data-c='take:([^']*)'", page)
    assert offered == [t3["take_id"], t2["take_id"]]                   # by ASR similarity
    assert gone["take_id"] not in page and t4["take_id"] not in page
    assert page.count("<audio") == 2


def test_player_never_raises(tmp_path):
    assert "MP3 missing" in Q._player(tmp_path / "nope.mp3")
    assert "MP3 missing" in Q._player("")
    (tmp_path / "a.mp3").write_bytes(b"ID3")
    assert Q._player(tmp_path / "a.mp3").startswith("<audio")
