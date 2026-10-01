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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)

from build.lib import bp_ipa as B  # noqa: E402
from build.lib import stage19_data as D  # noqa: E402
from build.lib import v4_tts as T  # noqa: E402
from build.lib.anki_pilot import SPIKE_BAD_SENSE_IDS  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
CONFIG = REPO_ROOT / "config"
SAMPLE = DATA / "_v4_pilot_sample.tsv"
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
    man = {(r["sense_id"], r["clip_type"]): r for r in read_tsv(DATA / "_audio_manifest.tsv")}
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
        ("v3_rerendered", sorted({r["sense_id"] for r in read_tsv(DATA / "_audio_manifest.tsv")
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


def spec_for(job: D.ClipJob, **kw) -> T.RenderSpec:
    return T.RenderSpec(
        sense_id=job.sense_id, clip_type=job.clip_type, display_text=job.display_text,
        voice_id=kw.pop("voice_id", job.voice_id), expected_ipa=job.expected_ipa,
        spoken_reference=job.spoken_reference, is_top_1000=job.is_top_1000,
        manual_pass=job.manual_pass, v3_duration_s=kw.pop("v3_duration_s", None), **kw)


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


def _engine() -> T.Engine:
    return T.Engine(judge_configs=T.load_judge_configs(), asr_model=T.load_asr_model(),
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
    print(f"  arms: {dict(by)} = {len(arms)} takes, {chars:,} chars ≈ "
          f"{chars * CREDITS_PER_CHAR:,.0f} credits (+ example t2 retries); judges "
          f"{[c['config'] for c in T.load_judge_configs()]}, ASR {T.load_asr_model()}")
    if not yes:
        print("  (cost table only — pass --yes to render)")
        return
    _with_durations(arms, jobs)
    eng = _engine()
    done = collections.Counter()

    def run(arm_spec):
        arm, spec = arm_spec
        res = eng.render_take(spec)
        if arm == "example_plain" and not res.qa_pass:
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


# ── v3 baseline ──────────────────────────────────────────────────────────────
BASELINE_FIELDS = ["sense_id", "clip_type", "voice_id", "object_key", "asr_transcript",
                   "asr_pass", "asr_mode", "judges", "judge_hash", "gate_pass", "qa_pass"]


def baseline(sample: list[dict], yes: bool) -> None:
    from build.lib import audio_qa as QA
    from build.lib.asr import asr_roundtrip

    jobs = D.load_jobs()
    keys = [(r["sense_id"], ct) for r in sample for ct in ("word", "example")]
    have = {(r["sense_id"], r["clip_type"]): r for r in read_tsv(BASELINE)}
    eng = _engine()
    todo = [k for k in keys if k not in have or have[k]["judge_hash"] != eng.judge_hash]
    print(f"  v3 baseline: {len(todo)} clips to judge ({len(keys) - len(todo)} cached)")
    if not yes or not todo:
        return
    out = dict(have)

    def one(k):
        j = jobs[k]
        mp3 = j.v3_path.read_bytes()
        a = asr_roundtrip(asr=eng.asr, mp3_bytes=mp3, input_text=j.display_text,
                          clip_type=j.clip_type, sense_id=j.sense_id, is_top_1000=j.is_top_1000)
        ok, mode, _ = QA.relaxed_asr_pass(
            transcripts=[a.transcript, a.biased_transcript], reference=j.display_text,
            clip_type=j.clip_type, is_top_1000=j.is_top_1000, production_decision=a.decision,
            spoken_reference=j.spoken_reference, manual_pass=j.manual_pass)
        res = T.TakeResult(spec=spec_for(j), asr_pass=ok, asr_mode=mode)
        eng._judge(res, mp3)
        return k, {"sense_id": j.sense_id, "clip_type": j.clip_type, "voice_id": j.voice_id,
                   "object_key": j.v3_object_key, "asr_transcript": a.transcript,
                   "asr_pass": int(ok), "asr_mode": mode,
                   "judges": json.dumps(res.judges, ensure_ascii=False),
                   "judge_hash": eng.judge_hash, "gate_pass": int(res.gate_pass),
                   "qa_pass": int(ok and res.gate_pass)}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, row in ex.map(one, todo):
            out[k] = row
    write_tsv(BASELINE, list(out.values()), fieldnames=BASELINE_FIELDS)
    p = sum(int(r["qa_pass"]) for r in out.values())
    print(f"  v3 baseline QA pass: {p}/{len(out)}")


# ── listening page ───────────────────────────────────────────────────────────
def _best(rows: list[dict]) -> dict | None:
    def score(r):
        judges = json.loads(r["judges"] or "{}")
        passes = sum(1 for v in judges.values() if v.get("verdict") == "bp_ok")
        return (int(r["qa_pass"]), passes, float(r["asr_similarity"] or 0))
    rows = [r for r in rows if r.get("path") and Path(r["path"]).exists()]
    return max(rows, key=score) if rows else None


def listen_page(sample: list[dict]) -> None:
    jobs = D.load_jobs()
    takes = collections.defaultdict(list)
    for r in read_tsv(T.LEDGER):
        takes[(r["sense_id"], r["clip_type"], r["variant"], r["stability"], r["dict_on"])].append(r)
    reported = {s for s in (r["sense_id"] for r in sample if r["stratum"] == "A:user_reported")}
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
def decide(sample: list[dict]) -> None:
    takes = collections.defaultdict(list)
    for r in read_tsv(T.LEDGER):
        takes[(r["sense_id"], r["clip_type"], r["variant"], r["stability"], r["dict_on"],
               r["take"])].append(r)
    base = {(r["sense_id"], r["clip_type"]): r for r in read_tsv(BASELINE)}
    strata = {r["sense_id"]: r["stratum"] for r in sample}
    stab = f"{T.DEFAULT_STABILITY:g}"

    def first(sid, ct, variant="plain", s=stab, d="0"):
        rows = takes.get((sid, ct, variant, s, d, "1"), [])
        return rows[-1] if rows else None

    def rate(pairs):
        pairs = [p for p in pairs if p is not None]
        return (sum(pairs) / len(pairs), len(pairs)) if pairs else (0.0, 0)

    ctrl = [s for s, st in strata.items() if st == "C:control"]
    weak = [s for s, st in strata.items() if st.startswith("A:")]
    ab = [s for s, st in strata.items() if st.startswith(("A:", "B:"))]
    v4_ctrl = rate([int(first(s, ct)["qa_pass"]) if first(s, ct) else None
                    for s in ctrl for ct in ("word", "example")])
    v3_ctrl = rate([int(base[(s, ct)]["qa_pass"]) if (s, ct) in base else None
                    for s in ctrl for ct in ("word", "example")])
    v4_weak = rate([int(first(s, "word")["qa_pass"]) if first(s, "word") else None for s in weak])
    v3_weak = rate([int(base[(s, "word")]["qa_pass"]) if (s, "word") in base else None for s in weak])
    v3_fail = [s for s in weak if (s, "word") in base and base[(s, "word")]["qa_pass"] == "0"]
    fixed = [s for s in v3_fail if any(int(r["qa_pass"]) for t in ("1", "2")
                                       for r in takes.get((s, "word", "plain", stab, "0", t), []))]
    fix_rate = len(fixed) / len(v3_fail) if v3_fail else 1.0

    labels = read_tsv(HUMAN_LABELS)
    key = {(r["row"], r["label"]): r["version"] for r in read_tsv(LISTEN_KEY)}
    human = collections.defaultdict(collections.Counter)
    for r in labels:
        ver = key.get((r["row"], r["label"]))
        if ver and r.get("verdict"):
            human[ver][r["verdict"]] += 1

    def bad_rate(ver):
        c = human.get(ver, collections.Counter())
        n = c["good"] + c["bad"]
        return (c["bad"] / n if n else None), n

    d1 = (v4_ctrl[0] >= v3_ctrl[0] - 0.01
          and (v4_weak[0] >= v3_weak[0] + 0.10 or fix_rate >= 0.30)
          and (bad_rate("v4_plain")[0] is None or bad_rate("v3")[0] is None
               or bad_rate("v4_plain")[0] <= bad_rate("v3")[0]))
    tag_rate = rate([int(first(s, "word", "tag")["qa_pass"]) if first(s, "word", "tag") else None
                     for s in ab])
    plain_ab = rate([int(first(s, "word")["qa_pass"]) if first(s, "word") else None for s in ab])
    tag_better = tag_rate[1] and tag_rate[0] >= plain_ab[0] + 0.05 and (
        bad_rate("v4_tag")[0] is None or bad_rate("v4_plain")[0] is None
        or bad_rate("v4_tag")[0] <= bad_rate("v4_plain")[0])
    tag_ok = tag_rate[1] and tag_rate[0] >= plain_ab[0] - 0.01
    word_order = ("tag,plain" if tag_better else "plain,tag" if tag_ok else "plain")
    good_rejected = 0
    good_total = 0
    for r in labels:
        ver = key.get((r["row"], r["label"]))
        if r.get("verdict") != "good" or not ver:
            continue
        good_total += 1
        path = next((k["path"] for k in read_tsv(LISTEN_KEY)
                     if k["row"] == r["row"] and k["label"] == r["label"]), "")
        led = next((t for t in read_tsv(T.LEDGER) if t["path"] == path), None)
        if ver == "v3":
            b = base.get((r["sense_id"], "word"))
            good_rejected += int(b is not None and b["qa_pass"] == "0")
        elif led is not None:
            good_rejected += int(led["qa_pass"] == "0")
    d3 = good_total == 0 or good_rejected / good_total <= 0.10
    voice_fail = collections.defaultdict(lambda: [0, 0])
    for s in strata:
        for ct in ("word", "example"):
            t = first(s, ct)
            if t:
                voice_fail[t["voice_id"]][0] += int(t["qa_pass"] == "0")
                voice_fail[t["voice_id"]][1] += 1
    vr = {v: f / n for v, (f, n) in voice_fail.items() if n}
    med = sorted(vr.values())[len(vr) // 2] if vr else 0
    voice_order = ",".join(sorted(vr, key=vr.get))
    demoted = [v for v, r in vr.items() if r > 2 * med and r > 0.05]
    stab_res = {}
    probe = [s for s in strata if takes.get((s, "word", "plain", "0.5", "0", "1"))]
    for sv in ("0.5", "0.65", "0.8"):
        stab_res[sv] = rate([int(first(s, "word", s=sv)["qa_pass"]) if first(s, "word", s=sv)
                             else None for s in probe])
    best_stab = max(stab_res, key=lambda k: stab_res[k][0]) if probe else stab
    stability = best_stab if stab_res.get(best_stab, (0,))[0] >= stab_res.get(stab, (0,))[0] + 0.05 \
        else stab
    hosp = [s for s, st in strata.items() if st == "D:hospital"]
    dict_on = rate([int(first(s, ct, d="1")["qa_pass"]) if first(s, ct, d="1") else None
                    for s in hosp for ct in ("word", "example")])
    dict_off = rate([int(first(s, ct)["qa_pass"]) if first(s, ct) else None
                     for s in hosp for ct in ("word", "example")])
    use_dict = bool(dict_on[1]) and dict_on[0] > dict_off[0]
    short = [s for s, st in strata.items() if st == "D:short"]
    short_v4 = rate([int(first(s, "word")["qa_pass"]) if first(s, "word") else None for s in short])
    short_v3 = rate([int(base[(s, "word")]["qa_pass"]) if (s, "word") in base else None
                     for s in short])
    short_policy = "v4_if_both_takes_pass" if short_v4[0] >= short_v3[0] else "keep_v3"
    policy = [
        ("go", "yes" if d1 else "NO", f"controls v4 {v4_ctrl} vs v3 {v3_ctrl}; weak v4 {v4_weak} "
                                       f"vs v3 {v3_weak}; fixes {len(fixed)}/{len(v3_fail)}; "
                                       f"human bad v3={bad_rate('v3')} v4={bad_rate('v4_plain')}"),
        ("word_variant_order", word_order, f"tag {tag_rate} vs plain {plain_ab} on A∪B; "
                                           f"human bad tag={bad_rate('v4_tag')}"),
        ("gate_ok", "yes" if d3 else "NO", f"{good_rejected}/{good_total} human-Good rejected"),
        ("voice_order", voice_order, json.dumps({v: round(r, 3) for v, r in vr.items()})),
        ("voices_demoted", ",".join(demoted), f"median fail rate {med:.3f}"),
        ("stability", stability, json.dumps({k: v for k, v in stab_res.items()})),
        ("dict_on", "yes" if use_dict else "no", f"with {dict_on} vs without {dict_off}"),
        ("short_words", short_policy, f"v4 {short_v4} vs v3 {short_v3}"),
        ("human_labels", str(len(labels)), json.dumps({k: dict(v) for k, v in human.items()})),
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
    print("  ┌─ pilot decisions " + "─" * 40)
    for k, v, e in policy:
        print(f"  │ {k:<20} {v:<28} {e[:110]}")
    print("  └" + "─" * 58)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--render", action="store_true")
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
