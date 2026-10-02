"""Stage 19 / Step 4 — eleven_v4 pilot: render arms, v3 baseline, listening page, decisions.

Modes (run in this order):
  --sample                 build data/_v4_pilot_sample.tsv (seed 19, ~650 senses)
  --render [--yes]         render all pilot arms (cost table first; --yes to spend)
  --baseline [--yes]       judge the CURRENT v3 clips of the pilot senses with the
                           same gate (ASR re-run + pinned judges) — fair comparison
  --listen-page            reports/19_3_listen.html (30 reported words × 3 blind
                           versions: v3 / v4 plain / v4 tag) + data/_v4_listen_key.tsv
  --decide                 apply the pre-registered rules D1–D8 using the takes
                           ledger, the baseline and data/_v4_human_labels.tsv →
                           config/stage19_policy.tsv + reports/19_3_pilot.html

Arms (IPA is OFF — the Step-1 probe showed eleven_v4 reads Portuguese IPA
with English-like letter sounds; "don't force it"):
  word    plain t1 + t2 (all), tag t1 (strata A ∪ B ∪ heterophones)
  example plain t1 (all), t2 when t1 fails QA
  probes  stability 0.5 / 0.8 on 100 risky words; v6 alias dictionary on the
          "hospital" senses
Nothing is uploaded; the manifest is untouched.
"""
from __future__ import annotations

import argparse
import base64
import collections
import html
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)

from build.lib import bp_ipa as B  # noqa: E402
from build.lib import stage19_data as D  # noqa: E402
from build.lib import v4_tts as T  # noqa: E402
from build.lib.anki_pilot import SPIKE_BAD_SENSE_IDS  # noqa: E402
from build.lib.tsv import iter_tsv, read_tsv, write_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
CONFIG = REPO_ROOT / "config"
SAMPLE = DATA / "_v4_pilot_sample.tsv"
MANIFEST = DATA / "_audio_manifest.tsv"
BASELINE = DATA / "_v4_pilot_v3_baseline.tsv"
LISTEN_PAGE = REPO_ROOT / "reports" / "19_3_listen.html"
LISTEN_KEY = DATA / "_v4_listen_key.tsv"
HUMAN_LABELS = DATA / "_v4_human_labels.tsv"
POLICY = CONFIG / "stage19_policy.tsv"
REPORT = REPO_ROOT / "reports" / "19_3_pilot.html"
PREFLIGHT_GATE = CONFIG / "stage19_preflight.json"
PT_DICT_LOCATOR = {"pronunciation_dictionary_id": "Ht0OpvDQAJQkJMHRv7Rs",
                   "version_id": "th9qzGumY1q3fkvV3Fi3"}
SEED = 19
CAP_A, N_B, N_C = 300, 100, 150
MIN_PER_VOICE = 45
LISTEN_WORDS = 30
STABILITY_PROBES = (0.5, 0.8)
N_STABILITY = 100
WORKERS = 24
CREDITS_PER_CHAR = 0.133   # measured in Step 1 (character-cost header)


class PilotError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (19_3_pilot): {msg}")


# ── sampling ─────────────────────────────────────────────────────────────────
def _senses_by_pt(final: list[dict]) -> dict[str, list[str]]:
    out = collections.defaultdict(list)
    for r in final:
        out[r["pt"].strip().lower()].append(r["sense_id"])
    return out


def _relaxed_asr_fail_v3(jobs: dict) -> set[str]:
    """v3 word clips whose stored transcript fails even the relaxed rule."""
    from build.lib import audio_qa as QA
    man = {(r["sense_id"], r["clip_type"]): r for r in read_tsv(MANIFEST)}
    out = set()
    for (sid, ct), j in jobs.items():
        if ct != "word":
            continue
        m = man[(sid, ct)]
        ok, _, _ = QA.relaxed_asr_pass(
            transcripts=[m.get("asr_transcript", "")], reference=j.display_text, clip_type=ct,
            is_top_1000=j.is_top_1000, production_decision=m.get("asr_decision", ""),
            manual_pass=j.manual_pass)
        if not ok and m.get("asr_transcript"):
            out.add(sid)
    return out


def build_sample() -> list[dict]:
    jobs = D.load_jobs()
    final = read_tsv(DATA / "06-final.tsv")
    by_pt = _senses_by_pt(final)
    fin = {r["sense_id"]: r for r in final}
    rng = random.Random(SEED)
    voice_of = {sid: jobs[(sid, "word")].voice_id for sid in fin}

    def pts_to_sids(path: Path, col: str = "pt") -> list[str]:
        return [s for r in read_tsv(path) for s in by_pt.get(r[col].strip().lower(), [])]

    pools_a = [
        ("calibration", [r["sense_id"] for r in read_tsv(DATA / "_audio_calibration_labels.tsv")
                         if r["label"] in ("MISPRONOUNCED", "UNCLEAR")]),
        ("confirmed_bad", [r["sense_id"] for r in
                           read_tsv(DATA / "_audio_mispronunciation_confirmed.tsv")]),
        ("user_reported", pts_to_sids(DATA / "_audio_user_reported_failures.tsv")),
        ("v3_rerendered", sorted({r["sense_id"] for r in read_tsv(MANIFEST)
                                  if "stage_16_9" in (r.get("notes") or "")})),
        ("v3_asr_fail", sorted(_relaxed_asr_fail_v3(jobs))),
        ("spike_bad", list(SPIKE_BAD_SENSE_IDS)),
        ("alias_winners", pts_to_sids(DATA / "_pronunciation_alias_v3_winners.tsv")),
    ]
    chosen: dict[str, str] = {}
    for label, sids in pools_a:
        for s in sids:
            if s in fin and s not in chosen and len(chosen) < CAP_A:
                chosen[s] = f"A:{label}"
    risk = [r["sense_id"] for r in read_tsv(DATA / "_audio_risk_classification.tsv")
            if r["priority"] in ("P0", "P1") and r["sense_id"] not in chosen]
    for s in rng.sample(risk, min(N_B, len(risk))):
        chosen[s] = "B:risk"
    special = []
    special += [(s, "D:heterophone") for s, r in fin.items()
                if r["pt"].strip().lower() in B.HETEROPHONES]
    special += [(s, "D:short") for s, r in fin.items() if len(r["pt"].strip()) <= 2]
    special += [(s, "D:article") for s, r in sorted(fin.items())
                if r["pt_display"].split(" ")[0] in ("o", "a") and r["pt_display"] != r["pt"]][:10]
    special += [(s, "D:multiword") for s, r in sorted(fin.items()) if " " in r["pt"].strip()][:10]
    special += [(s, "D:hospital") for s, r in fin.items()
                if "hospital" in (r["pt"] + " " + r["example_pt"]).lower()]
    special += [(s, "D:digits") for s, r in sorted(fin.items())
                if any(ch.isdigit() for ch in r["example_pt"])][:10]
    special += [(s, "D:ep_spelling") for s in ("3359.00.01", "3755.00.01", "3887.00.01",
                                                 "4052.00.01", "4052.00.02")]
    for s, lab in special:
        chosen.setdefault(s, lab)
    rest = [s for s in fin if s not in chosen]
    rng.shuffle(rest)
    per_voice = collections.Counter(voice_of[s] for s in chosen)
    n_c = 0
    for s in rest:                               # controls: fill thin voices first
        v = voice_of[s]
        if per_voice[v] < MIN_PER_VOICE or n_c < N_C:
            chosen[s] = "C:control"
            per_voice[v] += 1
            n_c += 1
        if n_c >= N_C and all(per_voice[v] >= MIN_PER_VOICE for v in set(voice_of.values())
                              if v in per_voice):
            break
    rows = [{"sense_id": s, "stratum": lab, "pt": fin[s]["pt"], "voice_id": voice_of[s]}
            for s, lab in sorted(chosen.items())]
    write_tsv(SAMPLE, rows, fieldnames=["sense_id", "stratum", "pt", "voice_id"])
    return rows


# ── arms ─────────────────────────────────────────────────────────────────────
def _tags_enabled() -> bool:
    gate = json.loads(PREFLIGHT_GATE.read_text()) if PREFLIGHT_GATE.exists() else {}
    return bool(gate.get("direction_tags_leak_free", {}).get(T.DEFAULT_TAG, False))


_RISKY: set[str] | None = None


def risky() -> set[str]:
    global _RISKY
    if _RISKY is None:
        _RISKY = D.risky_word_sids()
    return _RISKY


def spec_for(job: D.ClipJob, **kw) -> T.RenderSpec:
    return T.RenderSpec(
        sense_id=job.sense_id, clip_type=job.clip_type, display_text=job.display_text,
        voice_id=kw.pop("voice_id", job.voice_id), expected_ipa=job.expected_ipa,
        spoken_reference=job.spoken_reference, is_top_1000=job.is_top_1000,
        manual_pass=job.manual_pass, v3_duration_s=kw.pop("v3_duration_s", None),
        risky=job.sense_id in risky(), **kw)


def plan_arms(sample: list[dict], jobs: dict) -> list[tuple[str, T.RenderSpec]]:
    rng = random.Random(SEED)
    tags = _tags_enabled()
    strata = {r["sense_id"]: r["stratum"] for r in sample}
    arms: list[tuple[str, T.RenderSpec]] = []
    risky = [s for s, st in strata.items() if st.startswith(("A:", "B:"))]
    stab_set = set(rng.sample(risky, min(N_STABILITY, len(risky))))
    for sid, st in strata.items():
        w = jobs[(sid, "word")]
        arms += [("word_plain", spec_for(w, take=1)), ("word_plain", spec_for(w, take=2))]
        if tags and (st.startswith(("A:", "B:")) or st == "D:heterophone"):
            arms.append(("word_tag", spec_for(w, take=1, variant="tag")))
        if sid in stab_set:
            for s in STABILITY_PROBES:
                arms.append((f"word_stab_{s:g}", spec_for(w, take=1, stability=s)))
        e = jobs[(sid, "example")]
        arms.append(("example_plain", spec_for(e, take=1)))
        if st == "D:hospital":
            arms += [("word_dict", spec_for(w, take=1, dict_on=True)),
                     ("example_dict", spec_for(e, take=1, dict_on=True))]
    return arms


def _engine(*, allow_no_judges: bool = False) -> T.Engine:
    judges = T.load_judge_configs(allow_empty=allow_no_judges)
    if not judges:
        print("  ! no judge pinned yet (Gemini blocked?) — rendering TTS + ASR only; re-run "
              "--render after build/19_2_model_selection.py pins a judge: cached takes are "
              "judged then, without re-rendering")
    return T.Engine(judge_configs=judges, asr_model=T.load_asr_model(),
                    dict_locator=PT_DICT_LOCATOR)


def _with_durations(arms: list[tuple[str, T.RenderSpec]], jobs: dict) -> None:
    cache: dict[tuple, float | None] = {}

    def dur(spec):
        k = (spec.sense_id, spec.clip_type)
        if k not in cache:
            cache[k] = T.mp3_duration(jobs[k].v3_path)
        return cache[k]

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda a: dur(a[1]), arms))
    for _, s in arms:
        s.v3_duration_s = cache[(s.sense_id, s.clip_type)]


def render(sample: list[dict], yes: bool) -> None:
    jobs = D.load_jobs()
    arms = plan_arms(sample, jobs)
    chars = sum(len(s.tts_text) for _, s in arms)
    by = collections.Counter(a for a, _ in arms)
    judges = [c["config"] for c in T.load_judge_configs(allow_empty=True)]
    print(f"  arms: {dict(by)} = {len(arms)} takes, {chars:,} chars ≈ "
          f"{chars * CREDITS_PER_CHAR:,.0f} credits (+ example t2 retries); judges "
          f"{judges or 'PENDING'}, ASR {T.load_asr_model()}")
    if not yes:
        print("  (cost table only — pass --yes to render)")
        return
    _with_durations(arms, jobs)
    eng = _engine(allow_no_judges=True)
    done = collections.Counter()

    def run(arm_spec):
        arm, spec = arm_spec
        res = eng.render_take(spec)
        passed = res.qa_pass if eng.judges else res.pre_judge_pass
        if arm == "example_plain" and not passed and not res.qa_error:   # unchecked ≠ failed
            j = jobs[(spec.sense_id, "example")]
            res = eng.render_take(spec_for(j, take=2, v3_duration_s=spec.v3_duration_s))
        done[arm] += 1
        n = sum(done.values())
        if n % 100 == 0:
            print(f"    {n}/{len(arms)} takes, credits so far {eng.credits_used:,}", flush=True)
        return arm, res

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run, arms))
    qa = collections.defaultdict(lambda: [0, 0])
    for arm, res in results:
        qa[arm][0] += res.qa_pass
        qa[arm][1] += 1
    print("  QA pass by arm: " + ", ".join(f"{a} {p}/{n}" for a, (p, n) in sorted(qa.items())))
    print(f"  credits used (header sum): {eng.credits_used:,}")


def rejudge(sample: list[dict], yes: bool) -> None:
    """Judge-only pass after a gate change: every planned arm whose take is cached is
    brought up to the pinned judges, buying only the verdicts it lacks (see
    Engine.prior_verdicts). Never renders: a take that isn't cached is skipped and
    counted, and no example retake is tried."""
    jobs = D.load_jobs()
    arms = plan_arms(sample, jobs)
    eng = _engine()
    configs = [c for c, _ in eng.judges]
    calls, misses, todo = collections.Counter(), 0, []
    for _, spec in arms:
        row = eng.cached.get(spec.take_id)
        if not row or row.get("tts_text") != spec.tts_text or not row.get("path"):
            misses += 1
            continue
        res = T._result_from_row(spec, row)
        if res.error or res.sanity or res.loudness or not res.asr_pass:
            continue                                   # the judges never apply here
        need = eng.missing_judges(res)
        if need or res.judge_hash != eng.judge_hash or set(res.judges) != {
                c["config"] for c in configs if T.judge_applies(c, spec)}:
            calls.update(need)
            todo.append(spec)
    print(f"  re-judge: {len(todo)} cached takes; judge calls {dict(calls)} ≈ "
          f"${T.judge_spend(calls, configs):.2f}; {misses} arms not cached (skipped, never rendered)")
    if not yes or not todo:
        return
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(eng.cached_result, todo))
    print(f"  re-judged {sum(r is not None for r in results)}; QA pass "
          f"{sum(bool(r and r.qa_pass) for r in results)}/{len(todo)}")


# ── v3 baseline ──────────────────────────────────────────────────────────────
# v3_text / reference_text / error were added after the first baseline run; rows
# without them still read (decide() falls back to the manifest for v3_text).
BASELINE_FIELDS = ["sense_id", "clip_type", "voice_id", "object_key", "asr_transcript",
                   "asr_pass", "asr_mode", "judges", "judge_hash", "gate_pass", "qa_pass",
                   "v3_text", "reference_text", "error"]
BASELINE_SAVE_EVERY = 50      # checkpoint: a kill loses at most this many paid clips


def _v3_texts() -> dict[tuple[str, str], dict]:
    """Manifest rows by (sense_id, clip_type). `text_input` is the verbatim text
    sent to ElevenLabs for the clip at `object_key`, i.e. what the v3 clip says."""
    return {(m["sense_id"], m["clip_type"]): m for m in iter_tsv(MANIFEST)
            if m["clip_type"] in ("word", "example")}


def baseline(sample: list[dict], yes: bool) -> None:
    """Score each v3 clip with the gate against today's display text
    (`reference_text`; the J2 judges' expected IPA is for that text). `v3_text`
    records what the clip was rendered from. Where the two differ ("o/a X" before
    "o X, a X", EP spellings) the clip fails by construction, so decide() leaves it
    out of the v3-vs-v4 rules. Scoring it on its own text instead would need IPA
    for that text and a paid re-run, so it is not done.

    Rows are checkpointed as they finish. A clip that raises becomes an `error`
    row, and the next --baseline retries it. A row whose only problem is its judges
    (one errored or went down, or another judge config) keeps its paid ASR result
    and is only re-judged."""
    from build.lib import audio_qa as QA
    from build.lib.asr import asr_roundtrip

    jobs = D.load_jobs()
    man = _v3_texts()
    keys = [(r["sense_id"], ct) for r in sample for ct in ("word", "example")]
    have = {(r["sense_id"], r["clip_type"]): r for r in read_tsv(BASELINE)}
    eng = _engine()

    def asr_current(k, r):   # ASR ran on the deck's v3 clip today, against today's text
        j = jobs.get(k)
        return (j is not None and not r.get("error") and bool(r.get("asr_mode"))
                and r.get("object_key") == j.v3_object_key
                and (r.get("reference_text") or j.display_text) == j.display_text)

    def stale(k, r):         # ...and was judged, without errors, by the pinned judges
        return not asr_current(k, r) or r["judge_hash"] != eng.judge_hash or any(
            v.get("verdict") == "error" for v in json.loads(r["judges"] or "{}").values())

    todo = [k for k in keys if k not in have or stale(k, have[k])]
    rejudge = sum(1 for k in todo if k in have and asr_current(k, have[k]))
    print(f"  v3 baseline: {len(todo)} clips to judge ({len(keys) - len(todo)} cached"
          + (f"; {rejudge} keep their ASR and are only re-judged" if rejudge else "") + ")")
    configs = [c for c, _ in eng.judges]
    calls = collections.Counter()
    for k in todo:
        prev = have.get(k)
        if prev is not None and asr_current(k, prev) and prev.get("asr_pass") != "1":
            continue
        prior = json.loads(prev.get("judges") or "{}") if prev and asr_current(k, prev) else {}
        calls.update(T.judges_to_buy(configs, spec_for(jobs[k]), prior))
    asr_buy = len(todo) - rejudge
    print(f"  baseline spend: judge calls {dict(calls)} ≈ ${T.judge_spend(calls, configs):.2f}"
          f" + {asr_buy} ASR calls")
    if not yes or not todo:
        return
    out = dict(have)

    def one(k):
        """(k, row, the QaUnavailable that stops the run or None)."""
        down = None
        try:
            eng.qa_health.check()           # a provider is down: buy no more ASR
            j = jobs[k]
            v3_text = man[k]["text_input"]
            mp3 = j.v3_path.read_bytes()
            prev = have.get(k)
            if prev is not None and asr_current(k, prev):
                transcript, mode = prev["asr_transcript"], prev["asr_mode"]
                ok = prev["asr_pass"] == "1"
            else:
                a = asr_roundtrip(asr=eng.asr, mp3_bytes=mp3, input_text=j.display_text,
                                  clip_type=j.clip_type, sense_id=j.sense_id,
                                  is_top_1000=j.is_top_1000)
                transcript = a.transcript
                ok, mode, _ = QA.relaxed_asr_pass(
                    transcripts=[a.transcript, a.biased_transcript], reference=j.display_text,
                    clip_type=j.clip_type, is_top_1000=j.is_top_1000,
                    production_decision=a.decision, spoken_reference=j.spoken_reference,
                    manual_pass=j.manual_pass)
            res = T.TakeResult(spec=spec_for(j), asr_pass=ok, asr_mode=mode)
            if prev is not None and asr_current(k, prev):   # same clip and text: keep verdicts
                res.judges = json.loads(prev.get("judges") or "{}")
            try:   # the ASR is paid for: a judge outage keeps the row (no verdict yet)
                eng.qa_health.check()
                if ok:
                    eng._judge(res, mp3)
                else:                                  # judges never count on an ASR fail
                    res.judges, res.judge_hash = {}, eng.judge_hash
            except T.QaUnavailable as exc:
                down = exc
        except T.QaUnavailable:
            raise                           # nothing paid for this clip yet
        except Exception as exc:  # noqa: BLE001 — one clip must not sink the batch
            return k, {"sense_id": k[0], "clip_type": k[1], "judges": "{}", "asr_pass": 0,
                       "gate_pass": 0, "qa_pass": 0,
                       "error": f"{type(exc).__name__}: {exc}"[:300]}, None
        return k, {"sense_id": j.sense_id, "clip_type": j.clip_type, "voice_id": j.voice_id,
                   "object_key": j.v3_object_key, "asr_transcript": transcript,
                   "asr_pass": int(ok), "asr_mode": mode,
                   "judges": json.dumps(res.judges, ensure_ascii=False),
                   "judge_hash": res.judge_hash, "gate_pass": int(res.gate_pass),
                   "qa_pass": int(ok and res.gate_pass), "v3_text": v3_text,
                   "reference_text": j.display_text, "error": ""}, down

    def save():   # atomic rewrite of everything so far, in sample order
        rows = ({k: out[k] for k in keys if k in out} | out).values()
        write_tsv(BASELINE, list(rows), fieldnames=BASELINE_FIELDS)

    ex = ThreadPoolExecutor(max_workers=WORKERS)
    futs = [ex.submit(one, k) for k in todo]
    try:
        for n, fut in enumerate(as_completed(futs), 1):
            k, row, down = fut.result()
            out[k] = row
            if down:
                raise down                  # stop; in-flight clips still land below
            if n % BASELINE_SAVE_EVERY == 0:
                save()
                print(f"    {n}/{len(todo)} clips judged (saved)", flush=True)
    except BaseException:
        save()      # before waiting on in-flight clips: a second Ctrl-C keeps these
        raise
    finally:        # drop queued clips, keep every one already paid for
        try:
            ex.shutdown(wait=True, cancel_futures=True)
        finally:
            out.update((k, row) for f in futs
                       if f.done() and not f.cancelled() and f.exception() is None
                       for k, row, _ in [f.result()])
            save()
    p = sum(int(r["qa_pass"]) for r in out.values())
    errors = sum(1 for r in out.values() if r.get("error"))
    print(f"  v3 baseline QA pass: {p}/{len(out)}"
          + (f"; {errors} clips raised (re-run --baseline to retry them)" if errors else ""))


# ── listening page ───────────────────────────────────────────────────────────
def _best(rows: list[dict]) -> dict | None:
    def score(r):
        judges = json.loads(r["judges"] or "{}")
        passes = sum(1 for v in judges.values() if v.get("verdict") == "bp_ok")
        pre = int(not r["error"] and not r["sanity"] and not r["loudness"]
                  and r["asr_pass"] == "1")
        return (int(r["qa_pass"]), pre, passes, float(r["asr_similarity"] or 0))
    rows = [r for r in rows if r.get("path") and Path(r["path"]).exists()]
    return max(rows, key=score) if rows else None


def listen_page(sample: list[dict]) -> None:
    jobs = D.load_jobs()
    takes = collections.defaultdict(list)
    for r in T.read_ledger():
        takes[(r["sense_id"], r["clip_type"], r["variant"], r["stability"], r["dict_on"])].append(r)
    in_sample = {r["sense_id"] for r in sample}
    final_rows = read_tsv(DATA / "06-final.tsv")
    by_pt_all = _senses_by_pt(final_rows)
    reported = {s for r in read_tsv(DATA / "_audio_user_reported_failures.tsv")
                for s in by_pt_all.get(r["pt"].strip().lower(), []) if s in in_sample}
    rng = random.Random(SEED)
    # one sense per reported headword
    by_pt: dict[str, str] = {}
    for s in sorted(reported):
        by_pt.setdefault(jobs[(s, "word")].display_text, s)
    sids = rng.sample(sorted(by_pt.values()), min(LISTEN_WORDS, len(by_pt)))
    key_rows, cards = [], []
    final = {r["sense_id"]: r for r in read_tsv(DATA / "06-final.tsv")}
    for n, sid in enumerate(sids, 1):
        j = jobs[(sid, "word")]
        stab = f"{T.DEFAULT_STABILITY:g}"
        plain = _best(takes[(sid, "word", "plain", stab, "0")])
        tag = _best(takes[(sid, "word", "tag", stab, "0")])
        versions = [("v3", str(j.v3_path))]
        if plain:
            versions.append(("v4_plain", plain["path"]))
        if tag:
            versions.append(("v4_tag", tag["path"]))
        rng.shuffle(versions)
        players = []
        for i, (ver, path) in enumerate(versions):
            label = "ABC"[i]
            key_rows.append({"row": n, "sense_id": sid, "label": label, "version": ver,
                             "path": path})
            b64 = base64.b64encode(Path(path).read_bytes()).decode()
            players.append(
                f"<div class='take' data-label='{label}'><span class='lbl'>{label}</span>"
                f"<audio controls preload='none' src='data:audio/mpeg;base64,{b64}'></audio>"
                f"<span class='btns'>"
                + "".join(f"<button data-v='{v}'>{t}</button>" for v, t in
                          (("good", "Good"), ("bad", "Bad"), ("unsure", "Unsure")))
                + "</span></div>")
        f = final[sid]
        cards.append(
            f"<section class='card' data-row='{n}' data-sid='{sid}'>"
            f"<h3>{n}. {html.escape(j.display_text)} <span class='gloss'>— "
            f"{html.escape(f['en_primary'])}</span></h3>{''.join(players)}"
            f"<details><summary>Show expected IPA</summary><code>{html.escape(j.expected_ipa)}"
            f"</code></details><textarea placeholder='notes (optional)'></textarea></section>")
    write_tsv(LISTEN_KEY, key_rows, fieldnames=["row", "sense_id", "label", "version", "path"])
    page = _LISTEN_TEMPLATE.replace("__CARDS__", "".join(cards)).replace(
        "__N__", str(len(cards))).replace("__TOTAL__", str(len(key_rows)))
    LISTEN_PAGE.parent.mkdir(parents=True, exist_ok=True)
    LISTEN_PAGE.write_text(page, encoding="utf-8")
    print(f"  wrote {LISTEN_PAGE.relative_to(REPO_ROOT)}: {len(cards)} words, "
          f"{len(key_rows)} recordings; key → {LISTEN_KEY.relative_to(REPO_ROOT)}")


_LISTEN_TEMPLATE = """<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>v4 listening test</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--border:#2c3140;--text:#e6e9ef;--muted:#9098a6;
  --good:#4ade80;--bad:#ff6b6b;--unsure:#fbbf24;--accent:#5aa9ff}
*{box-sizing:border-box} body{background:var(--bg);color:var(--text);margin:0;padding:20px 16px;
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:820px;margin:0 auto} h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 14px} .bar{position:sticky;top:0;background:var(--bg);
  padding:8px 0;border-bottom:1px solid var(--border);display:flex;gap:10px;align-items:center;z-index:2}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin:12px 0}
.card h3{margin:0 0 8px;font-size:17px} .gloss{color:var(--muted);font-weight:400}
.take{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0}
.lbl{font-weight:700;width:18px} audio{height:32px;max-width:100%}
button{background:#232833;color:var(--text);border:1px solid var(--border);border-radius:6px;
  padding:4px 10px;cursor:pointer;font-size:13px}
button.on[data-v=good]{background:var(--good);color:#0b0d10} button.on[data-v=bad]{background:var(--bad);color:#0b0d10}
button.on[data-v=unsure]{background:var(--unsure);color:#0b0d10}
textarea{width:100%;min-height:34px;margin-top:6px;background:#11141a;color:var(--text);
  border:1px solid var(--border);border-radius:6px;padding:6px;font:inherit}
code{font-family:ui-monospace,Menlo,monospace} details{color:var(--muted);margin-top:6px}
#export{background:var(--accent);color:#0b0d10;border:none;font-weight:600}
</style></head><body><div class='wrap'>
<h1>eleven_v4 listening test</h1>
<p class='sub'>__N__ words you reported earlier, three blind versions each (__TOTAL__ recordings).
Mark every version Good / Bad / Unsure, then press Export and save the file as
<code>data/_v4_human_labels.tsv</code>. Progress is saved in this browser.</p>
<div class='bar'><button id='export'>Export TSV</button><span id='count' class='sub'></span></div>
__CARDS__
</div><script>
const KEY='v4_listen_state_v1';
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}; count()}
function count(){const n=document.querySelectorAll('.take').length;
  const d=Object.keys(st).filter(k=>st[k].v).length;document.getElementById('count').textContent=d+' / '+n+' rated'}
document.querySelectorAll('.card').forEach(card=>{
  const row=card.dataset.row, ta=card.querySelector('textarea');
  ta.value=(st['n'+row]||{}).n||''; ta.addEventListener('input',()=>{st['n'+row]={n:ta.value};save()});
  card.querySelectorAll('.take').forEach(t=>{const k=row+t.dataset.label;
    t.querySelectorAll('button').forEach(b=>{if((st[k]||{}).v===b.dataset.v)b.classList.add('on');
      b.addEventListener('click',()=>{st[k]={v:b.dataset.v};
        t.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));save()})})})});
document.getElementById('export').addEventListener('click',()=>{
  const lines=['row\\tsense_id\\tlabel\\tverdict\\tnotes'];
  document.querySelectorAll('.card').forEach(card=>{const row=card.dataset.row,sid=card.dataset.sid;
    const notes=((st['n'+row]||{}).n||'').replace(/[\\t\\n]/g,' ');
    card.querySelectorAll('.take').forEach(t=>{const k=row+t.dataset.label;
      lines.push([row,sid,t.dataset.label,(st[k]||{}).v||'',notes].join('\\t'))})});
  const blob=new Blob([lines.join('\\n')+'\\n'],{type:'text/tab-separated-values'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='_v4_human_labels.tsv';a.click()});
count();
</script></body></html>"""


# ── decisions ────────────────────────────────────────────────────────────────
VERDICTS = ("good", "bad", "unsure")
MAX_UNVERIFIED = 0.05   # go needs a valid verdict on ≥ 95% of the control and stratum-A clips


def _verdict_ok(r: dict, judge_hash: str) -> bool:
    """Is this ledger or baseline row's qa_pass a real verdict under the pinned
    judges? A row that raised (TTS/ASR error), whose judges errored, or that was
    judged under another judge config has none: outage noise is not a QA fail.
    A clip that failed before the judges (sanity, loudness, ASR) never reaches
    them, so that fail stands whatever the judge config."""
    if r.get("error"):
        return False
    if r.get("sanity") or r.get("loudness") or r.get("asr_pass") != "1":
        return True
    judges = json.loads(r.get("judges") or "{}")
    return r.get("judge_hash") == judge_hash and not any(
        v.get("verdict") == "error" for v in judges.values())


def _like_for_like(b: dict, display_text: str, man: dict) -> bool:
    """Was this baseline row's v3 clip rendered from the text v4 says today? Rows
    from before `v3_text` was recorded take the manifest's text_input, if the
    manifest still points at the same clip."""
    v3_text = b.get("v3_text")
    if not v3_text:
        m = man.get((b["sense_id"], b["clip_type"]))
        v3_text = m["text_input"] if m and m["object_key"] == b.get("object_key") else None
    return v3_text == display_text


def _human_labels() -> tuple[list[dict], str]:
    """data/_v4_human_labels.tsv joined to the listening key on (row, label) with a
    matching sense_id → (rated rows + the key's version and path, problem). The
    problem is "" only when every recording in the key has exactly one rating."""
    key = {(r["row"], r["label"]): r for r in read_tsv(LISTEN_KEY)}
    if not key:
        return [], f"no {LISTEN_KEY.name}: run --listen-page"
    if not HUMAN_LABELS.exists():
        return [], f"{HUMAN_LABELS.name} missing: export it from the listening page"
    rated, mismatched, seen = {}, 0, collections.Counter()
    for r in read_tsv(HUMAN_LABELS):
        rl = (r.get("row"), r.get("label"))
        k = key.get(rl)
        if k is None or k["sense_id"] != r.get("sense_id"):
            mismatched += 1
            continue
        seen[rl] += 1
        if r.get("verdict") in VERDICTS:
            rated[rl] = {**r, "version": k["version"], "path": k["path"]}
    problems = []
    if mismatched:
        problems.append(f"{mismatched} label rows don't match the key's (row, label, sense_id)"
                        " — labels from another listening page?")
    dups = sum(n - 1 for n in seen.values())
    if dups:
        problems.append(f"{dups} duplicate (row, label) label rows — two exports in one file?")
    if len(rated) < len(key):
        problems.append(f"{len(rated)}/{len(key)} recordings rated")
    return list(rated.values()), "; ".join(problems)


def _fmt(r: tuple) -> str:
    """A rate() triple for the evidence column."""
    return f"{r[0]:.3f} of {r[1]}" + (f" (+{r[2]} without a verdict)" if r[2] else "")


def _left_out(c: collections.Counter) -> str:
    return "; left out: " + ", ".join(f"{n} {why}" for why, n in c.items()) if c else ""


def decide(sample: list[dict]) -> None:
    jh = T.judge_hash(T.load_judge_configs())
    jobs = D.load_jobs()
    man = _v3_texts()
    ledger = T.read_ledger()
    takes = {(r["sense_id"], r["clip_type"], r["variant"], r["stability"], r["dict_on"],
              r["take"]): r for r in ledger}        # latest row per take (re-judged takes append)
    base = {(r["sense_id"], r["clip_type"]): r for r in read_tsv(BASELINE)}
    strata = {r["sense_id"]: r["stratum"] for r in sample}
    stab = f"{T.DEFAULT_STABILITY:g}"

    def first(sid, ct, variant="plain", s=stab, d="0", take="1"):
        return takes.get((sid, ct, variant, s, d, take))

    def ok(r):
        """A valid verdict under the pinned judges on what the clip says today: a v4
        take scored on an older display text, or a baseline row for another v3 clip
        or scored against another text, has none (re-run --render / --baseline).
        Baseline rows from before `reference_text` count as scored on today's text."""
        if r is None or not _verdict_ok(r, jh):
            return False
        j = jobs.get((r["sense_id"], r["clip_type"]))
        if j is None:
            return False
        if "take_id" in r:                                       # a v4 take
            return r["display_text"] == j.display_text
        return (r.get("object_key") == j.v3_object_key
                and (r.get("reference_text") or j.display_text) == j.display_text)

    def like(k):
        b, j = base.get(k), jobs.get(k)
        return b is not None and j is not None and _like_for_like(b, j.display_text, man)

    def rate(rows):
        """(pass rate, n, n without a valid verdict); missing rows are skipped."""
        rows = [r for r in rows if r is not None]
        kept = [int(r["qa_pass"]) for r in rows if ok(r)]
        return (sum(kept) / len(kept) if kept else 0.0), len(kept), len(rows) - len(kept)

    def paired(keys):
        """v4 first take vs v3 on the same clips: only where both have a valid
        verdict on the same text → (v4 rate, v3 rate, what was left out)."""
        v4, v3, skip = [], [], collections.Counter()
        for k in keys:
            t, b = first(*k), base.get(k)
            if t is None or b is None:
                skip["missing a take or baseline row"] += 1
            elif not (ok(t) and ok(b)):
                skip["without a valid verdict"] += 1
            elif not like(k):
                skip["where v3 says another text"] += 1
            else:
                v4.append(t)
                v3.append(b)
        return rate(v4), rate(v3), skip

    def same_clips(groups, n_arms=2):
        """v4 arms compared on the same clips: only where every arm has a valid
        verdict → ([rate() per arm], clips left out). A clip that an arm never
        rendered is not part of the comparison."""
        cols, out = [[] for _ in range(n_arms)], 0
        for rows in groups:
            if any(r is None for r in rows):
                continue
            if not all(ok(r) for r in rows):
                out += 1
                continue
            for col, r in zip(cols, rows, strict=True):
                col.append(r)
        return [rate(c) for c in cols], out

    ctrl = [s for s, st in strata.items() if st == "C:control"]
    weak = [s for s, st in strata.items() if st.startswith("A:")]
    ab = [s for s, st in strata.items() if st.startswith(("A:", "B:"))]
    short = [s for s, st in strata.items() if st == "D:short"]
    ctrl_keys = [(s, ct) for s in ctrl for ct in ("word", "example")]
    weak_keys = [(s, "word") for s in weak]
    v4_ctrl, v3_ctrl, ctrl_skip = paired(ctrl_keys)
    v4_weak, v3_weak, weak_skip = paired(weak_keys)
    # A v3 fail is fixed when either latest v4 plain take passes. When no take passes
    # and one has no valid verdict the outcome is unknown; it still counts as NOT
    # fixed, so judge-error noise can only lower the fix rate, never lift go to yes.
    v3_fail = [s for s in weak if ok(base.get((s, "word"))) and like((s, "word"))
               and base[(s, "word")]["qa_pass"] == "0"]
    fixed, unknown = [], []
    for s in v3_fail:
        ts = [t for t in (first(s, "word", take=n) for n in ("1", "2")) if t is not None]
        if any(ok(t) and t["qa_pass"] == "1" for t in ts):
            fixed.append(s)
        elif not ts or not all(ok(t) for t in ts):
            unknown.append(s)
    n_fix = len(v3_fail)
    # "No v3 fails to fix" passes only when stratum A has comparable verdicts at all.
    fix_rate = (len(fixed) / n_fix if n_fix
                else 1.0 if v3_weak[1] and not v3_fail else 0.0)

    labels, label_problem = _human_labels()

    def bad_rate(ver, skip=frozenset()):
        c = collections.Counter(r["verdict"] for r in labels
                                if r["version"] == ver and r["sense_id"] not in skip)
        n = c["good"] + c["bad"]
        return (c["bad"] / n if n else None), n

    def v3_says_other(sid):   # the v3 word clip on the listening page says another text
        j, m = jobs.get((sid, "word")), man.get((sid, "word"))
        return j is None or m is None or m["text_input"] != j.display_text

    human_skip = {r["sense_id"] for r in labels if v3_says_other(r["sense_id"])}
    bad_v3, bad_v4 = bad_rate("v3", human_skip), bad_rate("v4_plain", human_skip)

    # D3 first: go needs a gate confirmed on the recordings you rated Good.
    good_rejected = good_total = 0
    gate_skip = collections.Counter()
    latest = {t["take_id"]: t for t in ledger}      # last write wins (re-judged takes append)
    take_of_path = {t["path"]: t["take_id"] for t in ledger if t["path"]}
    for r in labels:
        if r["verdict"] != "good":
            continue
        is_v3 = r["version"] == "v3"
        v = (base.get((r["sense_id"], "word")) if is_v3
             else latest.get(take_of_path.get(T.local_take_path(r["path"]))))
        if not ok(v):
            gate_skip["without a valid gate verdict"] += 1
        elif is_v3 and not like((r["sense_id"], "word")):
            gate_skip["whose v3 clip says another text"] += 1
        else:
            good_total += 1
            good_rejected += int(v["qa_pass"] == "0")
    if label_problem:
        gate_ok, gate_why = "PENDING", f"human labels: {label_problem}"
    elif not good_total:
        gate_ok, gate_why = "PENDING", "no human-Good recording has a valid gate verdict"
    elif good_rejected / good_total > 0.10:
        gate_ok, gate_why = "NO", "the gate rejects more than 10% of the recordings rated Good"
    else:
        gate_ok, gate_why = "yes", ""

    # D1. Too little data is PENDING (re-run to fill it), never a verdict either way.
    sample_keys = [(s, ct) for s in strata for ct in ("word", "example")]
    no_base = [k for k in sample_keys if k not in base]
    no_t1 = [k for k in sample_keys if first(*k) is None]

    def gaps(skip):   # clips a re-run can still give a verdict ("another text" is for good)
        return skip["missing a take or baseline row"] + skip["without a valid verdict"]

    thin = [f"{name} {gaps(skip)}/{len(keys)}" for name, skip, keys in
            (("controls", ctrl_skip, ctrl_keys), ("stratum A", weak_skip, weak_keys))
            if keys and gaps(skip) / len(keys) > MAX_UNVERIFIED]
    rerun = "re-run --render / --baseline"
    if no_base:
        go, go_why = "PENDING", (f"{len(no_base)}/{len(sample_keys)} sample clips have no v3 "
                                 "baseline row: run --baseline")
    elif not v4_ctrl[1] or not v4_weak[1]:
        empty = "controls" if not v4_ctrl[1] else "stratum A"
        go, go_why = "PENDING", f"no comparable verdicts on {empty} under judges {jh}: {rerun}"
    elif thin:
        go, go_why = "PENDING", (f"too few valid verdicts ({', '.join(thin)} clips without one; "
                                 f"max {MAX_UNVERIFIED:.0%}): {rerun}")
    elif v4_ctrl[0] < v3_ctrl[0] - 0.01:
        go, go_why = "NO", "v4 is more than 1 pt below v3 on controls"
    elif not (v4_weak[0] >= v3_weak[0] + 0.10 or fix_rate >= 0.30):
        go, go_why = "NO", "on stratum A v4 neither gains 10 pts nor fixes 30% of v3 fails"
    elif label_problem or bad_v3[0] is None or bad_v4[0] is None:
        go, go_why = "PENDING", (f"human labels: {label_problem}" if label_problem
                                 else "no Good/Bad human labels for v3 or v4_plain")
    elif bad_v4[0] > bad_v3[0]:
        go, go_why = "NO", "your Bad-rate for v4 is above v3's"
    elif gate_ok != "yes":
        go, go_why = gate_ok, f"gate_ok is {gate_ok} (D3): {gate_why}"
    else:
        go, go_why = "yes", ""

    (tag_rate, plain_ab), tag_out = same_clips(
        [(first(s, "word", "tag"), first(s, "word")) for s in ab])
    tag_better = tag_rate[1] and tag_rate[0] >= plain_ab[0] + 0.05 and (
        bad_rate("v4_tag")[0] is None or bad_rate("v4_plain")[0] is None
        or bad_rate("v4_tag")[0] <= bad_rate("v4_plain")[0])
    tag_ok = tag_rate[1] and tag_rate[0] >= plain_ab[0] - 0.01
    word_order = ("tag,plain" if tag_better else "plain,tag" if tag_ok else "plain")
    voice_fail = collections.defaultdict(lambda: [0, 0])
    voice_skip = 0
    for s in strata:
        for ct in ("word", "example"):
            t = first(s, ct)
            if t is None:
                continue
            if not ok(t):
                voice_skip += 1
                continue
            voice_fail[t["voice_id"]][0] += int(t["qa_pass"] == "0")
            voice_fail[t["voice_id"]][1] += 1
    vr = {v: f / n for v, (f, n) in voice_fail.items() if n}
    med = sorted(vr.values())[len(vr) // 2] if vr else 0
    voice_order = ",".join(sorted(vr, key=vr.get))
    demoted = [v for v, r in vr.items() if r > 2 * med and r > 0.05]
    probe = [s for s in strata if takes.get((s, "word", "plain", "0.5", "0", "1"))]
    svs = ("0.5", "0.65", "0.8")
    stab_rates, stab_out = same_clips([[first(s, "word", s=sv) for sv in svs] for s in probe],
                                      len(svs))
    stab_res = dict(zip(svs, stab_rates, strict=True))
    n_probe = stab_rates[0][1]
    best_stab = max(stab_res, key=lambda k: stab_res[k][0]) if n_probe else stab
    stability = best_stab if n_probe and stab_res.get(best_stab, (0,))[0] >= \
        stab_res.get(stab, (0,))[0] + 0.05 else stab
    hosp = [s for s, st in strata.items() if st == "D:hospital"]
    (dict_on, dict_off), dict_out = same_clips(
        [(first(s, ct, d="1"), first(s, ct)) for s in hosp for ct in ("word", "example")])
    use_dict = bool(dict_on[1]) and dict_on[0] > dict_off[0]
    short_v4, short_v3, short_skip = paired([(s, "word") for s in short])
    short_policy = ("v4_if_both_takes_pass" if short_v4[1] and short_v4[0] >= short_v3[0]
                    else "keep_v3")
    d15_keys = set(ctrl_keys) | set(weak_keys) | {(s, "word") for s in short}
    n_takes = sum(1 for k, t in takes.items() if k[0] in strata and not ok(t))
    n_base = sum(1 for k, b in base.items() if k[0] in strata and not ok(b))
    n_text = sum(1 for k in d15_keys if ok(base.get(k)) and not like(k))
    pending = "; human labels pending" if label_problem else ""

    def both_out(n, what="clips"):
        return f"; {n} {what} without a valid verdict on every arm left out" if n else ""

    policy = [
        ("go", go, (f"{go}: {go_why}; " if go_why else "")
         + f"controls v4 {_fmt(v4_ctrl)} vs v3 {_fmt(v3_ctrl)}{_left_out(ctrl_skip)}; "
           f"weak v4 {_fmt(v4_weak)} vs v3 {_fmt(v3_weak)}{_left_out(weak_skip)}; "
           f"fixes {len(fixed)}/{n_fix}"
         + (f" ({len(unknown)} v3 fails without a valid v4 verdict counted as not fixed)" if unknown else "")
         + f"; human bad v3={bad_v3} v4={bad_v4}"
         + (f" ({len(human_skip)} words whose v3 clip says another text left out)"
            if human_skip else "")),
        ("word_variant_order", word_order, f"tag {_fmt(tag_rate)} vs plain {_fmt(plain_ab)} on "
                                           f"A∪B{both_out(tag_out, 'words')}; human bad "
                                           f"tag={bad_rate('v4_tag')}{pending}"),
        ("gate_ok", gate_ok, (f"{gate_ok}: {gate_why}; " if gate_why else "")
         + f"{good_rejected}/{good_total} human-Good rejected{_left_out(gate_skip)}"),
        ("voice_order", voice_order, json.dumps({v: round(r, 3) for v, r in vr.items()})),
        ("voices_demoted", ",".join(demoted), f"median fail rate {med:.3f}"
         + (f" ({voice_skip} takes without a valid verdict left out)" if voice_skip else "")),
        ("stability", stability, json.dumps({k: _fmt(v) for k, v in stab_res.items()})
         + both_out(stab_out, "words")
         + ("; no word has a verdict on every probe: default kept" if probe and not n_probe
            else "")),
        ("dict_on", "yes" if use_dict else "no",
         f"with {_fmt(dict_on)} vs without {_fmt(dict_off)}{both_out(dict_out)}"),
        ("short_words", short_policy,
         f"v4 {_fmt(short_v4)} vs v3 {_fmt(short_v3)}{_left_out(short_skip)}"),
        ("judge_hash", T.judge_hash(T.load_judge_configs()),
         "the judge gate these decisions were made under; 19_4 refuses another gate"),
        ("human_labels", str(len(labels)), (f"PENDING: {label_problem}; " if label_problem else "")
         + json.dumps({v: dict(collections.Counter(r["verdict"] for r in labels
                                                   if r["version"] == v))
                       for v in sorted({r["version"] for r in labels})})),
        ("excluded", str(n_takes + n_base + n_text + len(no_base) + len(no_t1)),
         f"{n_takes} v4 takes and {n_base} v3 baseline rows without a valid verdict under "
         f"judges {jh} (judge error, other judge config, a TTS/ASR error, or scored on "
         f"another text or v3 clip); {n_text} v3 clips of D1/D5 rendered from another text "
         f"than v4 says; {len(no_base)} sample clips without a v3 baseline row and "
         f"{len(no_t1)} without a v4 first take"),
    ]
    write_tsv(POLICY, [{"key": k, "value": v, "evidence": e} for k, v, e in policy],
              fieldnames=["key", "value", "evidence"])
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td><b>{html.escape(v)}</b></td>"
                   f"<td>{html.escape(e)}</td></tr>" for k, v, e in policy)
    REPORT.write_text(
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><title>v4 pilot decisions</title><style>"
        ":root{--bg:#0f1115;--border:#2c3140;--text:#e6e9ef;--muted:#9098a6}body{background:var(--bg);"
        "color:var(--text);font:14px/1.5 -apple-system,sans-serif;padding:24px 16px;margin:0}"
        ".wrap{max-width:1000px;margin:0 auto}table{border-collapse:collapse;width:100%}"
        "td,th{padding:6px 10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}"
        "</style></head><body><div class='wrap'><h1>Stage 19 pilot — decisions</h1>"
        f"<table><tr><th>rule</th><th>decision</th><th>evidence</th></tr>{rows}</table>"
        "</div></body></html>", encoding="utf-8")
    if n_takes or n_base:
        print(f"  ! {n_takes} v4 takes / {n_base} v3 baseline rows have no valid verdict under "
              f"judges {jh}: left out of every rate. Re-run --render / --baseline to fill them.")
    if no_base or no_t1:
        print(f"  ! {len(no_base)} sample clips have no v3 baseline row and {len(no_t1)} no v4 "
              "first take: re-run --baseline / --render")
    if n_text:
        print(f"  ! {n_text} v3 clips were rendered from another text than v4 says: left out "
              f"of the v3-vs-v4 rules (D1, D5)")
    if label_problem:
        print(f"  ! human labels: {label_problem} → go / gate_ok cannot be yes")
    if go_why:
        print(f"  ! go {go}: {go_why}")
    print("  ┌─ pilot decisions " + "─" * 40)
    for k, v, e in policy:
        print(f"  │ {k:<20} {v:<28} {e[:110]}")
    print("  └" + "─" * 58)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--rejudge", action="store_true",
                    help="judge-only pass over cached takes after a gate change (never renders)")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--listen-page", action="store_true")
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    print("=== Stage 19.3 — v4 pilot ===")
    if args.sample or not SAMPLE.exists():
        rows = build_sample()
        c = collections.Counter(r["stratum"].split(":")[0] for r in rows)
        v = collections.Counter(r["voice_id"] for r in rows)
        names = D.voice_names()
        print(f"  sample: {len(rows)} senses {dict(c)}; per voice "
              f"{ {names.get(k, k): n for k, n in v.items()} }")
    sample = read_tsv(SAMPLE)
    if args.rejudge:
        rejudge(sample, args.yes)
    if args.render:
        render(sample, args.yes)
    if args.baseline:
        baseline(sample, args.yes)
    if args.listen_page:
        listen_page(sample)
    if args.decide:
        decide(sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
