"""Stage 19 — pilot decisions (19_3 --decide), the v3 baseline (19_3 --baseline) and
the example retake rule of 19_3 --render, on synthetic ledgers, baselines,
manifests, listening keys and labels (no network, no spend).

decide(): only the latest ledger row per take counts; rows without a valid verdict
(judge error, another judge config, a TTS/ASR error, scored on another text) are
left out of every rate, and arms are compared on the same clips; v3 clips rendered
from another text than v4 says are left out of the v3-vs-v4 rules; too little data,
missing or partial human labels, or an unconfirmed gate give PENDING, never yes.
baseline(): checkpoints as it goes, records what each v3 clip says, keeps a clip
that raised as an error row and retries it, keeps paid clips on Ctrl-C and on a
judge outage, and re-judges a row whose only problem is its judges without ASR.
"""
from __future__ import annotations

import _thread
import importlib.util
import json
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


P = _load("stage19_3", REPO_ROOT / "build" / "19_3_pilot.py")

CFG = {"config": "j1", "backend": "openai", "model": "fake", "prompt_version": "J2"}
JH = T.judge_hash([CFG])
V1, V2, V3 = "Rw38T6bn0lTNOb1aUevR", "m151rjrbWXbBqyq56tly", "x3mAOLD9WpEGKEvYvNnG"
OLD_BASELINE_FIELDS = ["sense_id", "clip_type", "voice_id", "object_key", "asr_transcript",
                       "asr_pass", "asr_mode", "judges", "judge_hash", "gate_pass", "qa_pass"]


def _job(sid, ct, text, voice=V1):
    return D.ClipJob(sense_id=sid, clip_type=ct, display_text=text, voice_id=voice,
                     voice_gender="female", expected_ipa="", spoken_reference="", rank=10,
                     manual_pass=False, order=1, v3_model="eleven_v3",
                     v3_object_key=f"audio/{sid}-{T.SHORT_CLIP[ct]}-eleven_v3-v1.mp3")


class World:
    """A pilot on disk: sample, jobs, manifest, takes ledger, baseline, listening key
    and labels. `healthy()` builds one where every rule passes. run() gives every
    sample clip the test left without one a passing first take and v3 row (except
    the senses in `bare`)."""

    def __init__(self, tmp_path, monkeypatch):
        self.sample, self.jobs, self.manifest = [], {}, []
        self.ledger, self.base, self.key, self.labels = [], [], [], []
        self.base_fields = P.BASELINE_FIELDS
        self.write_labels = True
        self.bare = set()
        monkeypatch.setattr(T, "LEDGER", tmp_path / "takes.tsv")
        monkeypatch.setattr(T, "TAKES_DIR", tmp_path / "takes")
        monkeypatch.setattr(T, "load_judge_configs", lambda *a, **kw: [CFG])
        monkeypatch.setattr(D, "load_jobs", lambda: self.jobs)
        for name in ("BASELINE", "LISTEN_KEY", "HUMAN_LABELS", "POLICY", "MANIFEST"):
            monkeypatch.setattr(P, name, tmp_path / getattr(P, name).name)
        monkeypatch.setattr(P, "REPORT", tmp_path / "report.html")

    def sense(self, sid, stratum, word="verde", *, v3_word=None, voice=V1):
        self.sample.append({"sense_id": sid, "stratum": stratum, "pt": word, "voice_id": voice})
        for ct, text, v3 in (("word", word, v3_word or word),
                             ("example", f"{word} aqui.", f"{word} aqui.")):
            j = _job(sid, ct, text, voice)
            self.jobs[(sid, ct)] = j
            self.manifest.append({"sense_id": sid, "clip_type": ct, "text_input": v3,
                                  "object_key": j.v3_object_key})

    def take(self, sid, ct, qa, *, variant="plain", take=1, stability=0.65, dict_on=0,
             judge_hash=JH, verdict=None, asr_pass=1, error="", display_text=None):
        j = self.jobs[(sid, ct)]
        verdict = verdict or ("bp_ok" if qa else "non_bp")
        path = (f"/Users/someone/v4_takes/{sid}-{T.SHORT_CLIP[ct]}-{variant}-{j.voice_id[:6]}"
                f"-t{take}-s{stability:g}{'-d' if dict_on else ''}.mp3")
        self.ledger.append({
            "take_id": f"{sid}|{ct}|{variant}|{j.voice_id}|t{take}|s{stability:g}|d{dict_on}",
            "sense_id": sid, "clip_type": ct, "variant": variant, "voice_id": j.voice_id,
            "take": take, "stability": stability, "dict_on": dict_on,
            "display_text": display_text or j.display_text, "path": "" if error else path,
            "asr_pass": asr_pass, "judge_hash": judge_hash, "error": error,
            "judges": json.dumps({"j1": {"verdict": verdict}} if asr_pass and not error else {}),
            "gate_pass": int(verdict == "bp_ok"), "qa_pass": int(qa)})
        return path

    def v3(self, sid, ct, qa, *, judge_hash=JH, verdict=None, **extra):
        verdict = verdict or ("bp_ok" if qa else "non_bp")
        self.base.append({"sense_id": sid, "clip_type": ct, "voice_id": V1,
                          "object_key": self.jobs[(sid, ct)].v3_object_key, "asr_pass": 1,
                          "asr_mode": "production",
                          "judges": json.dumps({"j1": {"verdict": verdict}}),
                          "judge_hash": judge_hash, "gate_pass": int(verdict == "bp_ok"),
                          "qa_pass": int(qa), **extra})

    def listen(self, row, sid, label, version, path, verdict="good", label_sid=None):
        self.key.append({"row": row, "sense_id": sid, "label": label, "version": version,
                         "path": path})
        self.labels.append({"row": row, "sense_id": label_sid or sid, "label": label,
                            "verdict": verdict, "notes": ""})

    def healthy(self):
        for s in ("C1", "C2"):
            self.sense(s, "C:control")
            for ct in ("word", "example"):
                self.take(s, ct, True)
                self.v3(s, ct, True)
        for s in ("A1", "A2"):
            self.sense(s, "A:user_reported")
            self.take(s, "word", True)
            self.v3(s, "word", False)
        self.listen(1, "C1", "A", "v4_plain", self.ledger[0]["path"])
        self.listen(1, "C1", "B", "v3", str(self.jobs[("C1", "word")].v3_path))
        self.listen(2, "C2", "A", "v3", str(self.jobs[("C2", "word")].v3_path))
        self.listen(2, "C2", "B", "v4_plain", self.ledger[2]["path"])
        return self

    def fill(self):
        t1 = {(r["sense_id"], r["clip_type"]) for r in self.ledger if r["variant"] == "plain"
              and r["take"] == 1 and r["stability"] == 0.65 and not r["dict_on"]}
        b = {(r["sense_id"], r["clip_type"]) for r in self.base}
        for r in self.sample:
            for k in ((r["sense_id"], "word"), (r["sense_id"], "example")):
                if r["sense_id"] in self.bare:
                    continue
                if k not in t1:
                    self.take(*k, True)
                if k not in b:
                    self.v3(*k, True)

    def run(self):
        self.fill()
        write_tsv(T.LEDGER, self.ledger, fieldnames=T.LEDGER_FIELDS)
        write_tsv(P.BASELINE, self.base, fieldnames=self.base_fields)
        write_tsv(P.MANIFEST, self.manifest,
                  fieldnames=["sense_id", "clip_type", "text_input", "object_key"])
        if self.key:
            write_tsv(P.LISTEN_KEY, self.key, fieldnames=["row", "sense_id", "label", "version",
                                                          "path"])
        if self.write_labels:
            write_tsv(P.HUMAN_LABELS, self.labels,
                      fieldnames=["row", "sense_id", "label", "verdict", "notes"])
        P.decide(self.sample)
        return {r["key"]: (r["value"], r["evidence"]) for r in read_tsv(P.POLICY)}


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


# ── decide: the healthy pilot ────────────────────────────────────────────────
def test_healthy_pilot_goes(world):
    pol = world.healthy().run()
    assert pol["go"][0] == "yes", pol["go"]
    assert "controls v4 1.000 of 4 vs v3 1.000 of 4" in pol["go"][1]
    assert "fixes 2/2" in pol["go"][1]
    assert pol["gate_ok"] == ("yes", "0/4 human-Good rejected")
    assert pol["human_labels"][0] == "4" and pol["excluded"][0] == "0"


def test_old_baseline_header_still_reads(world):
    world.healthy().base_fields = OLD_BASELINE_FIELDS
    assert world.run()["go"][0] == "yes"


# ── decide: latest row per take ──────────────────────────────────────────────
def test_fixed_counts_only_the_latest_row_of_each_take(world):
    """A1 passed once on both takes, but both takes' latest rows fail: not a fix."""
    world.healthy()
    world.take("A1", "word", True, take=2)
    world.take("A1", "word", False, take=1)
    world.take("A1", "word", False, take=2)
    pol = world.run()
    assert "fixes 1/2" in pol["go"][1]
    assert "weak v4 0.500 of 2 vs v3 0.000 of 2" in pol["go"][1]


def test_unknown_fix_counts_as_not_fixed(world):
    """Judge-error noise on the takes of a v3 fail may only lower the fix rate:
    dropping the sense from the denominator would bias go towards yes."""
    world.healthy()
    world.take("A1", "word", False, verdict="error")             # latest t1: judge outage
    world.take("A1", "word", False, take=2)
    pol = world.run()
    assert "fixes 1/2 (1 v3 fails without a valid v4 verdict counted as not fixed)" in pol["go"][1]


# ── decide: rows without a valid verdict ─────────────────────────────────────
def test_judge_errors_stale_judges_and_raised_rows_are_left_out(world):
    world.healthy()
    world.take("C2", "word", False, verdict="error")             # judge outage (latest row)
    world.take("C2", "example", False, judge_hash="0ldc0nf1g0")   # judged under another config
    world.base[1].update(qa_pass=0, gate_pass=0,                  # C1 example: judge outage
                         judges=json.dumps({"j1": {"verdict": "error"}}))
    world.base[5].update(error="RuntimeError: ASR retries exhausted")   # A2 word raised
    pol = world.run()
    go = pol["go"][1]
    # old code: v4 0.5 vs v3 0.75 → NO on outage noise; now too little is left to decide
    assert pol["go"][0] == "PENDING", go
    assert ("too few valid verdicts (controls 3/4, stratum A 1/2 clips without one; max 5%)"
            in go)
    assert "controls v4 1.000 of 1 vs v3 1.000 of 1; left out: 3 without a valid verdict" in go
    assert "weak v4 1.000 of 1 vs v3 0.000 of 1; left out: 1 without a valid verdict" in go
    assert "fixes 1/1" in go
    # C2's v4_plain take was rated Good, but its latest row has no verdict
    assert pol["gate_ok"] == ("yes", "0/3 human-Good rejected; left out: 1 without a valid "
                                     "gate verdict")
    assert pol["excluded"][0] == "4"


def test_a_few_rows_without_a_verdict_do_not_hold_up_go(world):
    """1 of 22 control clips (4.5%) without a verdict is within MAX_UNVERIFIED."""
    world.healthy()
    for i in range(3, 12):
        world.sense(f"C{i}", "C:control")
    world.take("C5", "word", False, verdict="error")
    pol = world.run()
    assert pol["go"][0] == "yes", pol["go"]
    assert "controls v4 1.000 of 21 vs v3 1.000 of 21; left out: 1 without a valid verdict" \
        in pol["go"][1]


def test_pre_judge_fail_stands_whatever_the_judge_config(world):
    """An ASR failure never reaches the judges (empty judge_hash): a real fail."""
    world.healthy()
    world.take("C2", "word", False, asr_pass=0, judge_hash=T.judge_hash([]))
    pol = world.run()
    assert pol["go"][0] == "NO"
    assert "NO: v4 is more than 1 pt below v3 on controls" in pol["go"][1]
    assert "controls v4 0.750 of 4 vs v3 1.000 of 4" in pol["go"][1]
    assert pol["excluded"][0] == "0"


def test_tts_error_row_has_no_verdict(world):
    world.healthy()
    world.take("C2", "example", False, error="tts:ConnectionError: reset")
    pol = world.run()
    assert "controls v4 1.000 of 3 vs v3 1.000 of 3; left out: 1 without a valid verdict" \
        in pol["go"][1]


def test_take_or_baseline_scored_on_another_text_or_clip_has_no_verdict(world):
    """06-final changed after --render, or v3 was re-rendered after --baseline."""
    world.healthy()
    world.take("C2", "word", True, display_text="verde antigo")
    world.base[3]["object_key"] = "audio/C2-ex-eleven_v3-v0.mp3"  # C2 example
    pol = world.run()
    assert "controls v4 1.000 of 2 vs v3 1.000 of 2; left out: 2 without a valid verdict" \
        in pol["go"][1]
    assert pol["excluded"][0] == "2"


def test_judge_errors_do_not_demote_a_voice(world):
    world.healthy()
    for i, voice in enumerate((V2, V2, V2, V2, V3), 1):
        world.sense(f"B{i}", "B:risk", voice=voice)
    world.take("B1", "word", True)
    for s in ("B2", "B3", "B4"):
        world.take(s, "word", False, verdict="error")
    world.take("B5", "word", True)
    pol = world.run()
    assert pol["voices_demoted"][0] == ""
    assert "(3 takes without a valid verdict left out)" in pol["voices_demoted"][1]
    assert json.loads(pol["voice_order"][1])[V2] == 0.0


# ── decide: no data is PENDING, not a verdict ────────────────────────────────
def test_everything_stale_is_pending(world):
    """A judge config change with no re-judge: nothing comparable, so no vacuous yes."""
    world.healthy()
    world.sense("S1", "D:short", "é")
    world.fill()
    for r in world.ledger + world.base:
        r["judge_hash"] = "0ldc0nf1g0"
    pol = world.run()
    assert pol["go"][0] == "PENDING"
    assert "no comparable verdicts on controls under judges" in pol["go"][1]
    assert "controls v4 0.000 of 0" in pol["go"][1]
    assert pol["short_words"][0] == "keep_v3"
    assert pol["gate_ok"][0] == "PENDING"


@pytest.mark.parametrize("why", ["stale", "another_text"])
def test_no_comparable_controls_is_pending_even_when_stratum_a_improves(world, why):
    """D1 must not pass on an empty control comparison (0.0 ≥ 0.0 − 0.01)."""
    world.healthy()
    for r in world.ledger[:4] + world.base[:4]:                   # C1 and C2, both clips
        if why == "stale":
            r["judge_hash"] = "0ldc0nf1g0"
        else:
            r["v3_text"] = "o/a verde"
    pol = world.run()
    assert pol["go"][0] == "PENDING", pol["go"]
    assert "no comparable verdicts on controls" in pol["go"][1]
    assert "controls v4 0.000 of 0" in pol["go"][1]
    assert "weak v4 1.000 of 2 vs v3 0.000 of 2" in pol["go"][1]


def test_partial_baseline_or_missing_takes_are_pending(world):
    """A --baseline killed early covers the most frequent words only: no go on that."""
    world.healthy()
    world.sense("B1", "B:risk")
    world.take("B1", "word", True)
    world.take("B1", "example", True)                              # no v3 rows
    world.sense("B2", "B:risk")
    world.v3("B2", "word", True)
    world.v3("B2", "example", True)                                # no v4 takes
    world.bare |= {"B1", "B2"}
    pol = world.run()
    assert pol["go"][0] == "PENDING"
    assert "2/12 sample clips have no v3 baseline row: run --baseline" in pol["go"][1]
    assert pol["excluded"][0] == "4"
    assert "2 sample clips without a v3 baseline row and 2 without a v4 first take" \
        in pol["excluded"][1]


# ── decide: v3 clips that say another text ───────────────────────────────────
def test_v3_clip_with_another_text_is_left_out_of_d1_and_d5(world):
    world.healthy()
    world.sense("C3", "C:control", "o artista, a artista", v3_word="o/a artista")
    c3 = world.take("C3", "word", True)
    world.v3("C3", "word", False)                                  # scored on the new text
    world.take("C3", "example", True)
    world.v3("C3", "example", True)
    world.sense("S1", "D:short", "pê", v3_word="pé")
    world.take("S1", "word", True)
    world.v3("S1", "word", False)
    world.sense("S2", "D:short", "ir")
    world.take("S2", "word", True)
    world.v3("S2", "word", True)
    world.sense("B1", "B:risk", "o/a juiz", v3_word="o juiz")      # not a D1/D5 clip
    world.listen(3, "C3", "A", "v3", str(world.jobs[("C3", "word")].v3_path), verdict="bad")
    world.listen(3, "C3", "B", "v4_plain", c3)
    pol = world.run()
    go = pol["go"][1]
    assert ("controls v4 1.000 of 5 vs v3 1.000 of 5; left out: 1 where v3 says another text"
            in go)
    assert "human bad v3=(0.0, 2) v4=(0.0, 2) (1 words whose v3 clip says another text" in go
    assert pol["go"][0] == "yes"
    assert pol["short_words"] == ("v4_if_both_takes_pass", "v4 1.000 of 1 vs v3 1.000 of 1; "
                                  "left out: 1 where v3 says another text")
    assert pol["excluded"][0] == "2"                               # C3 and S1, not B1
    assert "2 v3 clips of D1/D5 rendered from another text" in pol["excluded"][1]


def test_recorded_v3_text_wins_over_the_manifest(world):
    """New baseline rows say what the clip said; old rows fall back to the manifest."""
    world.healthy()
    world.base[0].update(v3_text="verde antigo", reference_text="verde")   # C1 word
    world.manifest[2]["text_input"] = "outro texto"                          # C2 word
    world.base[2].update(v3_text="verde", reference_text="verde")
    pol = world.run()
    assert ("controls v4 1.000 of 3 vs v3 1.000 of 3; left out: 1 where v3 says another text"
            in pol["go"][1])


def test_v3_scored_against_an_older_reference_has_no_verdict(world):
    world.healthy()
    world.base[0].update(v3_text="verde", reference_text="verdes")         # C1 word
    pol = world.run()
    assert "left out: 1 without a valid verdict" in pol["go"][1]


def test_gate_skips_a_v3_good_label_on_a_clip_that_says_another_text(world):
    """The v3 o/a clip says the old text and failed the gate on the new one: a gate
    rejection of a recording rated Good would be an artefact, so it is left out."""
    world.healthy()
    world.sense("C3", "C:control", "o artista, a artista", v3_word="o/a artista")
    c3 = world.take("C3", "word", True)
    world.v3("C3", "word", False)
    world.listen(3, "C3", "A", "v3", str(world.jobs[("C3", "word")].v3_path))
    world.listen(3, "C3", "B", "v4_plain", c3)
    pol = world.run()
    assert pol["gate_ok"] == ("yes", "0/5 human-Good rejected; left out: 1 whose v3 clip says "
                                     "another text")


# ── decide: human labels and the gate ────────────────────────────────────────
def test_missing_labels_are_pending_not_yes(world):
    world.healthy().write_labels = False
    pol = world.run()
    assert pol["go"][0] == "PENDING" and "_v4_human_labels.tsv missing" in pol["go"][1]
    assert pol["gate_ok"][0] == "PENDING" and "0/0 human-Good rejected" in pol["gate_ok"][1]
    assert "human labels pending" in pol["word_variant_order"][1]


def test_missing_listen_key_is_pending(world):
    world.healthy().key = []
    pol = world.run()
    assert pol["go"][0] == "PENDING" and "run --listen-page" in pol["go"][1]


def test_partly_rated_labels_are_pending(world):
    world.healthy().labels[3]["verdict"] = ""
    pol = world.run()
    assert pol["go"][0] == "PENDING" and "3/4 recordings rated" in pol["go"][1]
    assert pol["gate_ok"][0] == "PENDING"


def test_labels_for_another_sense_are_reported(world):
    world.healthy().labels[2]["sense_id"] = "A1"
    pol = world.run()
    assert pol["gate_ok"][0] == "PENDING"
    assert "1 label rows don't match the key's (row, label, sense_id)" in pol["gate_ok"][1]
    assert "3/4 recordings rated" in pol["human_labels"][1]


def test_duplicate_labels_are_pending(world):
    """Two exports in one file: the last row must not silently win."""
    world.healthy()
    world.labels.append({**world.labels[0], "verdict": "bad"})
    pol = world.run()
    assert pol["go"][0] == "PENDING" and pol["gate_ok"][0] == "PENDING"
    assert "1 duplicate (row, label) label rows" in pol["go"][1]
    assert "1 duplicate (row, label) label rows" in pol["human_labels"][1]


def test_no_good_labels_leave_the_gate_pending(world):
    world.healthy()
    for r in world.labels:
        r["verdict"] = "unsure"
    pol = world.run()
    assert pol["gate_ok"][0] == "PENDING" and "no human-Good" in pol["gate_ok"][1]
    assert pol["go"][0] == "PENDING" and "no Good/Bad human labels" in pol["go"][1]


def test_human_bad_rate_can_stop_go(world):
    world.healthy().labels[0]["verdict"] = "bad"                  # C1's v4_plain
    pol = world.run()
    assert pol["go"][0] == "NO" and "your Bad-rate for v4 is above v3's" in pol["go"][1]


def test_failed_automated_rule_is_no_even_without_labels(world):
    world.healthy().write_labels = False
    world.take("C1", "word", False)
    world.take("C2", "word", False)
    assert world.run()["go"][0] == "NO"


def test_a_gate_that_rejects_good_recordings_stops_go(world):
    """D3 NO: 19_4 checks only go, so go must not be yes on an unconfirmed gate."""
    world.healthy()
    world.base[0].update(qa_pass=0, gate_pass=0)                   # C1 v3 word, rated Good
    world.base[2].update(qa_pass=0, gate_pass=0)                   # C2 v3 word, rated Good
    pol = world.run()
    assert pol["gate_ok"][0] == "NO"
    assert "2/4 human-Good rejected" in pol["gate_ok"][1]
    assert pol["go"][0] == "NO"
    assert "gate_ok is NO (D3)" in pol["go"][1]


def test_go_waits_for_a_gate_without_verdicts(world):
    """Every recording rated, none Good with a valid verdict: gate PENDING, go too."""
    world.healthy()
    for r in world.labels:
        r["verdict"] = "bad"                                       # v3 and v4: 2/2 bad each
    tag = world.take("A1", "word", False, variant="tag", verdict="error")
    world.listen(3, "A1", "A", "v4_tag", tag)                      # the only Good: no verdict
    pol = world.run()
    assert pol["gate_ok"][0] == "PENDING"
    assert pol["go"][0] == "PENDING"
    assert "gate_ok is PENDING (D3): no human-Good recording" in pol["go"][1]


# ── decide: v4 arms compared on the same clips (D2, D6, D7) ──────────────────
def test_tag_order_compares_the_same_words(world):
    """A1's tag take has no verdict and its plain take fails: comparing the arms
    over different words would rate tag 1.0 vs plain 0.5 and put tag first."""
    world.healthy()
    world.take("A1", "word", False)                                # latest plain: fail
    world.take("A1", "word", False, variant="tag", verdict="error")
    world.take("A2", "word", True, variant="tag")
    pol = world.run()
    assert pol["word_variant_order"][0] == "plain,tag"
    assert ("tag 1.000 of 1 vs plain 1.000 of 1 on A∪B; 1 words without a valid verdict on "
            "every arm left out" in pol["word_variant_order"][1])


def test_stability_and_dict_compare_the_same_clips(world):
    world.healthy()
    for s in ("B1", "B2", "B3"):
        world.sense(s, "B:risk")
    world.take("B1", "word", False, verdict="error")               # 0.65: no verdict
    world.take("B2", "word", False)
    world.take("B3", "word", True)
    for s, qa in (("B1", True), ("B2", False), ("B3", True)):
        world.take(s, "word", qa, stability=0.5)                   # 0.5: 2/3 over all words
    for s in ("B1", "B2", "B3"):
        world.take(s, "word", False, stability=0.8)
    world.sense("H1", "D:hospital", "hospital")
    world.take("H1", "word", False, verdict="error")               # dict off: no verdict
    world.take("H1", "word", True, dict_on=1)
    world.take("H1", "example", False)
    world.take("H1", "example", False, dict_on=1)
    pol = world.run()
    stab = pol["stability"]
    assert stab[0] == "0.65", stab                                 # 0.5 vs 0.65: 0.5 = 0.5
    assert json.loads(stab[1].split("; ")[0]) == {"0.5": "0.500 of 2", "0.65": "0.500 of 2",
                                                  "0.8": "0.000 of 2"}
    assert "1 words without a valid verdict on every arm left out" in stab[1]
    assert pol["dict_on"] == ("no", "with 0.000 of 1 vs without 0.000 of 1; 1 clips without "
                                    "a valid verdict on every arm left out")


def test_probe_arm_without_verdicts_keeps_the_defaults(world):
    """An outage on one arm only: no comparison, so no switch."""
    world.healthy()
    for s in ("B1", "B2", "B3"):
        world.sense(s, "B:risk")
        world.take(s, "word", False, verdict="error")              # every 0.65 take
        world.take(s, "word", True, stability=0.5)
        world.take(s, "word", True, stability=0.8)
    world.sense("H1", "D:hospital", "hospital")
    for ct in ("word", "example"):
        world.take("H1", ct, False, verdict="error")               # every dict-off take
        world.take("H1", ct, True, dict_on=1)
    pol = world.run()
    assert pol["stability"][0] == "0.65"
    assert "no word has a verdict on every probe: default kept" in pol["stability"][1]
    assert pol["dict_on"][0] == "no"
    assert pol["dict_on"][1].startswith("with 0.000 of 0 vs without 0.000 of 0; 2 clips")


# ── render: the example retake ───────────────────────────────────────────────
@pytest.mark.parametrize("t1, retake", [
    ({"error": "asr:TimeoutError: read timed out"}, False),        # unchecked, not failed
    ({"asr_pass": True, "judges": {"j1": {"verdict": "error"}}}, False),
    ({"asr_pass": True, "judges": {"j1": {"verdict": "non_bp"}}}, True),   # a real fail
])
def test_render_buys_an_example_retake_only_for_a_real_fail(world, monkeypatch, t1, retake):
    world.sense("C1", "C:control")
    calls = []

    def render_take(spec):
        calls.append((spec.clip_type, spec.take))
        if spec.clip_type == "example" and spec.take == 1:
            return T.TakeResult(spec=spec, judge_hash=JH, **t1)
        return T.TakeResult(spec=spec, asr_pass=True, gate_pass=True, judge_hash=JH,
                            judges={"j1": {"verdict": "bp_ok"}})
    eng = types.SimpleNamespace(render_take=render_take, judges=[(CFG, None)], credits_used=0)
    monkeypatch.setattr(P, "_engine", lambda **kw: eng)
    monkeypatch.setattr(P, "_with_durations", lambda arms, jobs: None)
    monkeypatch.setattr(P, "_tags_enabled", lambda: False)
    monkeypatch.setattr(T, "load_asr_model", lambda *a, **kw: "fake-asr")
    P.render(world.sample, yes=True)
    assert sorted(calls) == [("example", 1)] + [("example", 2)] * retake + [("word", 1),
                                                                            ("word", 2)]


# ── baseline ─────────────────────────────────────────────────────────────────
class RefusedError(Exception):
    status_code = 402                                             # billing: provider down


class FakeJudge:
    def __init__(self):
        self.refuse = set()
        self.calls = []

    def judge(self, **kw):
        self.calls.append(kw["pt"])
        if kw["pt"] in self.refuse:
            raise RefusedError("credits are depleted")
        return types.SimpleNamespace(pronunciation_verdict="bp_ok", drift="none",
                                     severity="low", evidence="ok")


@pytest.fixture
def bl(world, tmp_path, monkeypatch):
    """baseline() on three senses (one o/a word) with fake ASR and judge."""
    import build.lib.asr as A

    world.sense("0001.00.01", "C:control", "o artista, a artista", v3_word="o/a artista")
    world.sense("0002.00.01", "C:control", "verde")
    world.sense("0003.00.01", "A:user_reported", "casa")
    write_tsv(P.MANIFEST, world.manifest,
              fieldnames=["sense_id", "clip_type", "text_input", "object_key"])
    monkeypatch.setattr(D, "ANKI_MEDIA", tmp_path / "media")
    (tmp_path / "media").mkdir()
    for j in world.jobs.values():
        j.v3_path.write_bytes(b"ID3 v3 " + j.sense_id.encode())
    e = T.Engine.__new__(T.Engine)
    e.asr, e.judge_configs, e.judge_hash = object(), [CFG], JH
    e.judges = [(CFG, FakeJudge())]
    e.openai_judge_sem = threading.BoundedSemaphore(2)
    e.gemini_limiter = types.SimpleNamespace(wait_before_call=lambda: None)
    e.qa_health = T.QaHealth()
    monkeypatch.setattr(P, "_engine", lambda **kw: e)
    monkeypatch.setattr(P, "WORKERS", 1)
    ns = types.SimpleNamespace(world=world, calls=[], fail={}, hook={}, judge=e.judges[0][1],
                               engine=e)

    def fake_roundtrip(**kw):
        text = kw["input_text"]
        ns.calls.append(text)
        if text in ns.hook:
            ns.hook[text]()
        if text in ns.fail:
            raise ns.fail[text]
        return types.SimpleNamespace(transcript=text, biased_transcript=text, decision="pass")
    monkeypatch.setattr(A, "asr_roundtrip", fake_roundtrip)
    return ns


ALL_TEXTS = ["o artista, a artista", "o artista, a artista aqui.", "verde", "verde aqui.",
             "casa", "casa aqui."]


def _baseline_rows():
    return {(r["sense_id"], r["clip_type"]): r for r in read_tsv(P.BASELINE)}


def test_baseline_checkpoints_and_records_what_v3_said(bl, monkeypatch):
    sizes = []
    real = P.write_tsv

    def spy(path, rows, **kw):
        rows = list(rows)
        sizes.append(len(rows))
        return real(path, rows, **kw)
    monkeypatch.setattr(P, "write_tsv", spy)
    monkeypatch.setattr(P, "BASELINE_SAVE_EVERY", 2)
    P.baseline(bl.world.sample, yes=True)
    assert sizes == [2, 4, 6, 6]                                  # checkpoints, then the final save
    rows = _baseline_rows()
    assert list(read_tsv(P.BASELINE)[0]) == P.BASELINE_FIELDS
    oa = rows[("0001.00.01", "word")]
    assert oa["v3_text"] == "o/a artista" and oa["reference_text"] == "o artista, a artista"
    assert all(r["qa_pass"] == "1" and r["error"] == "" and r["judge_hash"] == JH
               for r in rows.values())


def test_baseline_clip_that_raises_is_kept_as_error_and_retried(bl):
    bl.fail["verde"] = RuntimeError("ASR retries exhausted")
    P.baseline(bl.world.sample, yes=True)
    rows = _baseline_rows()
    assert len(rows) == 6                                         # the batch carried on
    bad = rows[("0002.00.01", "word")]
    assert bad["error"].startswith("RuntimeError") and bad["qa_pass"] == "0"
    assert sum(r["qa_pass"] == "1" for r in rows.values()) == 5
    bl.fail.clear()
    bl.calls.clear()
    P.baseline(bl.world.sample, yes=True)
    assert bl.calls == ["verde"]                                  # only the error row re-run
    assert _baseline_rows()[("0002.00.01", "word")]["qa_pass"] == "1"


def test_baseline_interrupt_in_a_worker_keeps_finished_clips(bl):
    bl.fail["verde"] = KeyboardInterrupt()                        # the 3rd clip submitted
    with pytest.raises(KeyboardInterrupt):
        P.baseline(bl.world.sample, yes=True)
    rows = _baseline_rows()
    assert {("0001.00.01", "word"), ("0001.00.01", "example")} <= set(rows)
    assert ("0002.00.01", "word") not in rows


def test_baseline_ctrl_c_keeps_every_paid_clip_and_resumes(bl):
    """Ctrl-C arrives in the main thread while a worker is mid-clip: the clip in
    flight finishes and is saved; the next run does only the rest."""
    bl.hook["verde"] = _thread.interrupt_main
    with pytest.raises(KeyboardInterrupt):
        P.baseline(bl.world.sample, yes=True)
    rows = _baseline_rows()
    assert {("0001.00.01", "word"), ("0001.00.01", "example"), ("0002.00.01", "word")} \
        <= set(rows)
    assert all(r["qa_pass"] == "1" for r in rows.values())
    done = {bl.world.jobs[k].display_text for k in rows}
    bl.hook.clear()
    bl.calls.clear()
    P.baseline(bl.world.sample, yes=True)
    assert sorted(bl.calls) == sorted(t for t in ALL_TEXTS if t not in done)
    assert len(_baseline_rows()) == 6


def test_baseline_second_ctrl_c_while_waiting_keeps_collected_clips(bl, monkeypatch):
    """The first Ctrl-C stops the loop while a clip is still in flight; a second one
    while shutdown waits for that clip must not lose it."""
    release = threading.Event()

    class Executor(P.ThreadPoolExecutor):
        def shutdown(self, *a, **kw):
            release.set()
            super().shutdown(*a, **kw)
            raise KeyboardInterrupt                               # pressed again while waiting
    monkeypatch.setattr(P, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(P, "WORKERS", 2)
    bl.hook["o artista, a artista"] = lambda: release.wait(5)    # in flight until shutdown
    bl.fail["o artista, a artista aqui."] = KeyboardInterrupt()   # the first Ctrl-C
    with pytest.raises(KeyboardInterrupt):
        P.baseline(bl.world.sample, yes=True)
    rows = _baseline_rows()
    assert rows[("0001.00.01", "word")]["qa_pass"] == "1"
    assert ("0001.00.01", "example") not in rows


def test_baseline_keeps_rows_from_the_old_header(bl):
    old = {"sense_id": "0002.00.01", "clip_type": "word", "voice_id": V1,
           "object_key": bl.world.jobs[("0002.00.01", "word")].v3_object_key,
           "asr_transcript": "verde", "asr_pass": 1, "asr_mode": "production",
           "judges": json.dumps({"j1": {"verdict": "bp_ok"}}), "judge_hash": JH,
           "gate_pass": 1, "qa_pass": 1}
    write_tsv(P.BASELINE, [old], fieldnames=OLD_BASELINE_FIELDS)
    P.baseline(bl.world.sample, yes=True)
    assert "verde" not in bl.calls and len(bl.calls) == 5
    kept = _baseline_rows()[("0002.00.01", "word")]
    assert kept["qa_pass"] == "1" and kept["v3_text"] == "" and kept["error"] == ""


@pytest.mark.parametrize("change, asr_again", [
    ({"judge_hash": "0ldc0nf1g0"}, False),                        # another judge config
    ({"judges": json.dumps({"j1": {"verdict": "error"}})}, False),  # a judge errored
    ({"reference_text": "verdes"}, True),                         # scored on another text
    ({"object_key": "audio/0002.00.01-word-eleven_v3-v0.mp3"}, True),   # another v3 clip
])
def test_baseline_rejudges_on_the_stored_asr_when_only_the_judges_are_stale(bl, change,
                                                                            asr_again):
    old = {"sense_id": "0002.00.01", "clip_type": "word", "voice_id": V1,
           "object_key": bl.world.jobs[("0002.00.01", "word")].v3_object_key,
           "asr_transcript": "verdi", "asr_pass": 0, "asr_mode": "fail",
           "judges": json.dumps({"j1": {"verdict": "bp_ok"}}), "judge_hash": JH,
           "gate_pass": 1, "qa_pass": 0, "v3_text": "verde", "reference_text": "verde",
           "error": ""}
    write_tsv(P.BASELINE, [{**old, **change}], fieldnames=P.BASELINE_FIELDS)
    P.baseline(bl.world.sample, yes=True)
    assert ("verde" in bl.calls) is asr_again
    # The stored ASR fail stands, and judges never count on an ASR fail: none are bought.
    assert ("verde" in bl.judge.calls) is asr_again
    row = _baseline_rows()[("0002.00.01", "word")]
    assert row["judge_hash"] == JH
    if asr_again:
        assert json.loads(row["judges"])["j1"]["verdict"] == "bp_ok"
    else:                                                         # the stored ASR result stands
        assert (row["asr_transcript"], row["asr_pass"], row["asr_mode"]) == ("verdi", "0", "fail")
        assert row["qa_pass"] == "0" and json.loads(row["judges"]) == {}


def test_baseline_stops_when_a_judge_is_down_and_keeps_the_paid_asr(bl):
    """A billing refusal is not a per-clip error: stop paying ASR for the rest. The
    clip whose ASR was bought is kept without a verdict and only re-judged later."""
    bl.judge.refuse.add("verde")                                  # the 3rd clip submitted
    with pytest.raises(T.JudgeUnavailable):
        P.baseline(bl.world.sample, yes=True)
    assert bl.calls == ALL_TEXTS[:3]                              # no ASR after the outage
    rows = _baseline_rows()
    assert set(rows) == {("0001.00.01", "word"), ("0001.00.01", "example"),
                         ("0002.00.01", "word")}
    kept = rows[("0002.00.01", "word")]
    assert kept["asr_transcript"] == "verde" and kept["error"] == ""
    assert json.loads(kept["judges"])["j1"]["verdict"] == "error"
    bl.judge.refuse.clear()
    bl.engine.qa_health = T.QaHealth()                            # a new process
    bl.calls.clear()
    P.baseline(bl.world.sample, yes=True)
    assert bl.calls == ALL_TEXTS[3:]                              # "verde" only re-judged
    rows = _baseline_rows()
    assert len(rows) == 6 and all(r["qa_pass"] == "1" for r in rows.values())


def test_baseline_buys_no_asr_once_a_provider_is_down(bl):
    bl.engine.qa_health.error("judge:j1", RefusedError("credits are depleted"), 20,
                              T.JudgeUnavailable)
    with pytest.raises(T.JudgeUnavailable):
        P.baseline(bl.world.sample, yes=True)
    assert bl.calls == [] and _baseline_rows() == {}
