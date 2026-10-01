"""Stage 19 / Step 2 — pick the audio judge (and check the ASR) on audio the user already labeled.

No ElevenLabs credits. Judges every labeled clip with every candidate config,
twice (to measure verdict flips), and applies the pre-registered rule.

Label sets
  L  data/_audio_calibration_labels.tsv — 185 user labels on legacy word clips
     (OK 83 / MISPRONOUNCED 53 / UNCLEAR 48 + 1 unanswered; UNCLEAR is reported,
     not scored). Audio: build/audio_cache/<object_key at commit ad1eb7d>.
  B  data/_audio_judge_ab_human_labels.tsv — 40 labels on v3 BP clips (Stage 16 A/B).
  S  phoneme_asr_spike_review.tsv — legacy human OK / BAD groups (+ verde v2, which
     the user rejects).

Candidates (models × prompt versions, see build/lib/judge_prompts.py)
  gemini-3.1-pro-preview × {J1, J1p, J2}   gemini-3.8-flash × {J1p, J2}
  gemini-3.5-flash × {J2}                   gpt-audio × {J1p, J2}
J2 gets the expected São Paulo IPA from data/_ipa_v2.tsv (Step 3).

Pre-registered rule
  * flag = verdict ≠ bp_ok. recall = flagged MISPRONOUNCED / scored MISPRONOUNCED
    (L ∪ S-BAD); false-reject = flagged OK / scored OK (L ∪ S-OK ∪ B bp_ok).
  * eligible: false-reject ≤ (J1 on 3.1-pro) + 10 pts, flip rate ≤ 10 %.
  * winner: highest recall among eligible single configs and 2-config AND
    gates (a clip passes only if both pass); ties → lower false-reject → cheaper.
  * runner-up documented as fallback (preview models can disappear mid-run).

ASR check: gpt-4o-transcribe vs gpt-transcribe vs gemini-3.5-transcribe vs
scribe_v2 on 300 seeded current v3 clips (200 word, 100 example) with the
production pass rule; the incumbent stays unless a challenger passes ≥ 1 pt
more AND agrees on ≥ 98 % of clips (no LM "fixing").

Outputs: data/_v4_model_selection.tsv, reports/19_2_model_selection.html,
         config/stage19_models.tsv, audit/19_2_judges.jsonl
Usage:
  uv run python build/19_2_model_selection.py --dry-run
  uv run python build/19_2_model_selection.py --yes [--configs a,b] [--skip-asr]
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import itertools
import json
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"
ANKI_MEDIA = DATA / "anki_media"
CACHE = REPO_ROOT / "build" / "cache" / "19_2_judge_cache.jsonl"
ASR_CACHE = REPO_ROOT / "build" / "cache" / "19_2_asr_cache.jsonl"
JUDGE_AUDIT = REPO_ROOT / "audit" / "19_2_judges.jsonl"
OUT_TSV = DATA / "_v4_model_selection.tsv"
REPORT = REPO_ROOT / "reports" / "19_2_model_selection.html"
PINNED = REPO_ROOT / "config" / "stage19_models.tsv"
IPA_V2 = DATA / "_ipa_v2.tsv"
CALIBRATION_COMMIT = "ad1eb7d"
SEED = 19
REPS = 2
WORKERS = {"gemini": 24, "openai": 8}

CONFIGS = [  # name, backend, model, prompt
    ("pro-J1", "gemini", "gemini-3.1-pro-preview", "J1"),
    ("pro-J1p", "gemini", "gemini-3.1-pro-preview", "J1p"),
    ("pro-J2", "gemini", "gemini-3.1-pro-preview", "J2"),
    ("flash38-J1p", "gemini", "gemini-3.8-flash", "J1p"),
    ("flash38-J2", "gemini", "gemini-3.8-flash", "J2"),
    ("flash35-J2", "gemini", "gemini-3.5-flash", "J2"),
    ("gptaudio-J1p", "openai", "gpt-audio", "J1p"),
    ("gptaudio-J2", "openai", "gpt-audio", "J2"),
]
BASELINE = "pro-J1"
MAX_FALSE_REJECT_DELTA = 0.10
MAX_FLIP_RATE = 0.10
ASR_MODELS = ("gpt-4o-transcribe", "gpt-transcribe", "gemini-3.5-transcribe", "scribe_v2")


class SelectionError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (19_2_model_selection): {msg}")


# ── label sets ───────────────────────────────────────────────────────────────
def load_clips() -> list[dict]:
    ipa = {r["sense_id"]: r for r in read_tsv(IPA_V2)}
    final = {r["sense_id"]: r for r in read_tsv(DATA / "06-final.tsv")}

    def expected(sid: str, clip_type: str = "word") -> str:
        r = ipa.get(sid) or {}
        if clip_type == "word":
            return r.get("ipa_word") or final.get(sid, {}).get("ipa_word", "")
        return r.get("ipa_example") or final.get(sid, {}).get("ipa_example", "")

    clips: list[dict] = []
    old = subprocess.run(["git", "show", f"{CALIBRATION_COMMIT}:data/_audio_manifest.tsv"],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    old_man = {(r["sense_id"], r["clip_type"]): r
               for r in csv.DictReader(io.StringIO(old), dialect="excel-tab")}
    for r in read_tsv(DATA / "_audio_calibration_labels.tsv"):
        lab = {"OK": "ok", "MISPRONOUNCED": "bad", "UNCLEAR": "unclear"}.get(r["label"])
        if not lab:
            continue
        key = old_man[(r["sense_id"], "word")]["object_key"].split("/", 1)[1]
        clips.append({"clip_id": f"L:{r['sense_id']}", "set": "L", "label": lab,
                      "sense_id": r["sense_id"], "clip_type": "word", "pt": r["pt"],
                      "path": str(AUDIO_CACHE / key), "url": "",
                      "ipa": expected(r["sense_id"]), "voice_id": r["voice_id"]})
    ab = {r["row_id"]: r for r in read_tsv(DATA / "_audio_judge_ab.tsv")}
    for r in read_tsv(DATA / "_audio_judge_ab_human_labels.tsv"):
        lab = {"bp_ok": "ok", "non_bp_ep": "bad", "non_bp_other": "bad"}.get(r["human_verdict"],
                                                                             "unclear")
        a = ab[r["row_id"]]
        key = a["url"].split("/audio/", 1)[-1] if "/audio/" in a["url"] else ""
        ct = a["clip_type"]
        clips.append({"clip_id": f"B:{r['row_id']}", "set": "B", "label": lab,
                      "sense_id": a["sense_id"], "clip_type": ct, "pt": a["text"],
                      "path": str(AUDIO_CACHE / key) if key else "", "url": a["url"],
                      "ipa": expected(a["sense_id"], "word" if ct == "word" else "example"),
                      "voice_id": a["voice_id"]})
    for r in read_tsv(REPO_ROOT / "phoneme_asr_spike_review.tsv"):
        g = r["group"]
        lab = {"legacy_human_OK": "ok", "legacy_human_BAD": "bad", "v3_contested": "bad"}.get(g)
        if not lab:
            continue
        clips.append({"clip_id": f"S:{r['filename']}", "set": "S", "label": lab,
                      "sense_id": r["sense_id"], "clip_type": "word", "pt": r["pt"],
                      "path": str(AUDIO_CACHE / r["filename"]), "url": "",
                      "ipa": expected(r["sense_id"]), "voice_id": ""})
    return clips


_HTTP = httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=60, follow_redirects=True)


def audio_bytes(c: dict) -> bytes:
    p = Path(c["path"]) if c["path"] else None
    if p and p.exists():
        return p.read_bytes()
    if c["url"]:
        r = _HTTP.get(c["url"])
        r.raise_for_status()
        return r.content
    raise SelectionError(f"no audio for {c['clip_id']}")


# ── judging ──────────────────────────────────────────────────────────────────
def _load_cache(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["k"]] = rec["v"]
    return out


def judge_all(clips: list[dict], configs: list[tuple]) -> dict[str, dict]:
    from build.lib.audio_judge import AudioJudgeClient
    from build.lib.gemini_audio_judge import GeminiAudioJudgeClient

    cache = _load_cache(CACHE)
    lock = threading.Lock()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = dict(cache)
    for name, backend, model, prompt in configs:
        client = (GeminiAudioJudgeClient(model=model, prompt_version=prompt, audit_path=JUDGE_AUDIT)
                  if backend == "gemini" else
                  AudioJudgeClient(model=model, prompt_version=prompt, audit_path=JUDGE_AUDIT))
        todo = [(c, rep) for c in clips for rep in range(REPS)
                if f"{name}|{c['clip_id']}|{rep}" not in cache]
        if not todo:
            continue
        print(f"  {name}: {len(todo)} judge calls", flush=True)
        errors = 0

        def one(job, name=name, client=client, prompt=prompt):
            nonlocal errors
            c, rep = job
            k = f"{name}|{c['clip_id']}|{rep}"
            try:
                r = client.judge(audio_bytes=audio_bytes(c), pt=c["pt"],
                                 ipa_word_final=c["ipa"] if prompt == "J2" else "",
                                 voice_id=c["voice_id"] or "label-set", sense_id=c["sense_id"],
                                 clip_type=c["clip_type"])
                v = {"verdict": r.pronunciation_verdict, "drift": r.drift,
                     "severity": r.severity, "evidence": r.evidence[:200],
                     "cost": r.cost_usd}
                with lock:
                    out[k] = v
                    with CACHE.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors += 1
                    out[k] = {"verdict": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}

        with ThreadPoolExecutor(max_workers=WORKERS[backend]) as ex:
            list(ex.map(one, todo))
        if errors:
            print(f"    {name}: {errors} errors (e.g. {next(v['error'] for k, v in out.items() if k.startswith(name + '|') and v.get('verdict') == 'error')})",
                  flush=True)
    return out


def metrics(clips: list[dict], verdicts: dict, names: tuple[str, ...]) -> dict:
    """names: one config, or several combined as an AND gate (flag if any flags)."""
    flagged_bad = scored_bad = flagged_ok = scored_ok = flips = flip_n = errors = 0
    b_agree = b_n = 0
    for c in clips:
        reps = []
        for rep in range(REPS):
            vs = [verdicts.get(f"{n}|{c['clip_id']}|{rep}", {}).get("verdict") for n in names]
            if any(v in (None, "error") for v in vs):
                reps.append(None)
                continue
            reps.append(any(v != "bp_ok" for v in vs))
        valid = [x for x in reps if x is not None]
        if not valid:
            errors += 1
            continue
        flag = valid[0]
        if len(valid) == REPS:
            flip_n += 1
            flips += int(len(set(valid)) > 1)
        if c["label"] == "bad":
            scored_bad += 1
            flagged_bad += flag
        elif c["label"] == "ok":
            scored_ok += 1
            flagged_ok += flag
        if c["set"] == "B" and c["label"] in ("ok", "bad"):
            b_n += 1
            b_agree += int(flag == (c["label"] == "bad"))
    return {
        "recall": flagged_bad / scored_bad if scored_bad else 0.0,
        "false_reject": flagged_ok / scored_ok if scored_ok else 0.0,
        "flip_rate": flips / flip_n if flip_n else 0.0,
        "b_agreement": b_agree / b_n if b_n else 0.0,
        "n_bad": scored_bad, "n_ok": scored_ok, "errors": errors,
    }


def decide(clips: list[dict], verdicts: dict, configs: list[tuple]) -> dict:
    names = [c[0] for c in configs]
    avail = [n for n in names if metrics(clips, verdicts, (n,))["errors"] < len(clips) * 0.2]
    table = {(n,): metrics(clips, verdicts, (n,)) for n in avail}
    for a, b in itertools.combinations(avail, 2):
        if configs[names.index(a)][2] != configs[names.index(b)][2] or True:
            table[(a, b)] = metrics(clips, verdicts, (a, b))
    base = table.get((BASELINE,))
    fr_cap = (base["false_reject"] if base else 0.15) + MAX_FALSE_REJECT_DELTA
    eligible = {k: m for k, m in table.items()
                if m["false_reject"] <= fr_cap and m["flip_rate"] <= MAX_FLIP_RATE}
    ranked = sorted(eligible.items(),
                    key=lambda kv: (-round(kv[1]["recall"], 3), kv[1]["false_reject"], len(kv[0])))
    winner = ranked[0][0] if ranked else None
    runner = next((k for k, _ in ranked[1:] if winner and set(k) != set(winner)
                   and not set(k) & set(winner)), ranked[1][0] if len(ranked) > 1 else None)
    return {"table": table, "fr_cap": fr_cap, "winner": winner, "runner_up": runner,
            "available": avail}


# ── ASR check ────────────────────────────────────────────────────────────────
def asr_check(n_word: int = 200, n_ex: int = 100) -> dict:
    from build.lib import asr_alt
    from build.lib.asr import pad_mp3_with_silence, text_similarity

    rows = [r for r in read_tsv(DATA / "_audio_manifest.tsv")
            if r["clip_type"] in ("word", "example") and r["tts_model"] == "eleven_v3"]
    rng = random.Random(SEED)
    words = rng.sample([r for r in rows if r["clip_type"] == "word"], n_word)
    exs = rng.sample([r for r in rows if r["clip_type"] == "example"], n_ex)
    sample = words + exs
    cache = _load_cache(ASR_CACHE)
    lock = threading.Lock()
    oa = asr_alt.make_openai_client()
    el = asr_alt.make_elevenlabs_client()
    try:
        from google import genai
        import os
        gm = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception:  # noqa: BLE001
        gm = None

    def one(job):
        model, r = job
        k = f"{model}|{r['object_key']}"
        if k in cache:
            return
        mp3 = (ANKI_MEDIA / r["object_key"].split("/", 1)[1]).read_bytes()
        if r["clip_type"] == "word":
            mp3 = pad_mp3_with_silence(mp3)
        try:
            if model == "scribe_v2":
                t = asr_alt.transcribe_elevenlabs(client=el, mp3_bytes=mp3).transcript
            elif model.startswith("gemini"):
                t = asr_alt.transcribe_gemini(client=gm, mp3_bytes=mp3, model=model).transcript
            else:
                t = asr_alt.transcribe_openai(client=oa, model=model, mp3_bytes=mp3).transcript
            v = {"t": t}
        except Exception as exc:  # noqa: BLE001
            v = {"error": f"{type(exc).__name__}: {exc}"[:160]}
        with lock:
            cache[k] = v
            if "t" in v:
                with ASR_CACHE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")

    jobs = [(m, r) for m in ASR_MODELS for r in sample]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, jobs))
    res = {}
    for m in ASR_MODELS:
        sims, errs, passed = [], 0, 0
        for r in sample:
            v = cache.get(f"{m}|{r['object_key']}", {})
            if "t" not in v:
                errs += 1
                continue
            s = text_similarity(v["t"], r["text_input"])
            sims.append(s)
            passed += s >= (0.95 if r["clip_type"] == "word" else 0.92)
        n = len(sample) - errs
        res[m] = {"pass_rate": passed / n if n else 0.0, "n": n, "errors": errs}
    inc = ASR_MODELS[0]
    for m in ASR_MODELS[1:]:
        same = tot = 0
        for r in sample:
            a = cache.get(f"{inc}|{r['object_key']}", {}).get("t")
            b = cache.get(f"{m}|{r['object_key']}", {}).get("t")
            if a is None or b is None:
                continue
            tot += 1
            same += text_similarity(a, b) >= 0.99
        res[m]["agree_with_incumbent"] = same / tot if tot else 0.0
    best = inc
    for m in ASR_MODELS[1:]:
        if (res[m]["n"] and res[m]["pass_rate"] >= res[inc]["pass_rate"] + 0.01
                and res[m].get("agree_with_incumbent", 0) >= 0.98):
            best = m
    return {"models": res, "winner": best, "sample": len(sample)}


# ── outputs ──────────────────────────────────────────────────────────────────
def write_outputs(clips, verdicts, configs, dec, asr) -> None:
    rows = []
    for c in clips:
        row = {"clip_id": c["clip_id"], "set": c["set"], "label": c["label"], "pt": c["pt"]}
        for name, *_ in configs:
            row[name] = "/".join(verdicts.get(f"{name}|{c['clip_id']}|{rep}", {}).get("verdict", "")
                                 for rep in range(REPS))
        rows.append(row)
    write_tsv(OUT_TSV, rows, fieldnames=["clip_id", "set", "label", "pt"] + [c[0] for c in configs])
    cfg = {c[0]: c for c in configs}
    pinned = []
    for role, key in (("judge", dec["winner"]), ("judge_fallback", dec["runner_up"])):
        if not key:
            continue
        for name in key:
            _, backend, model, prompt = cfg[name]
            pinned.append({"role": role, "config": name, "backend": backend, "model": model,
                           "prompt_version": prompt, "combine": "AND" if len(key) > 1 else ""})
    if asr:
        pinned.append({"role": "asr", "config": asr["winner"], "backend": "", "model": asr["winner"],
                       "prompt_version": "", "combine": ""})
    PINNED.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(PINNED, pinned, fieldnames=["role", "config", "backend", "model", "prompt_version",
                                          "combine"])
    esc = html.escape
    trs = []
    for key, m in sorted(dec["table"].items(), key=lambda kv: -kv[1]["recall"]):
        mark = ("WINNER" if key == dec["winner"] else "fallback" if key == dec["runner_up"] else "")
        ok = m["false_reject"] <= dec["fr_cap"] and m["flip_rate"] <= MAX_FLIP_RATE
        trs.append(f"<tr><td>{esc(' AND '.join(key))}</td><td>{m['recall']:.0%}</td>"
                   f"<td>{m['false_reject']:.0%}</td><td>{m['flip_rate']:.0%}</td>"
                   f"<td>{m['b_agreement']:.0%}</td><td>{'yes' if ok else 'no'}</td>"
                   f"<td><b>{mark}</b></td></tr>")
    asr_rows = "".join(
        f"<tr><td>{esc(mname)}</td><td>{v['pass_rate']:.1%}</td><td>{v['n']}</td><td>{v['errors']}</td>"
        f"<td>{v.get('agree_with_incumbent', 1):.1%}</td></tr>"
        for mname, v in (asr or {}).get("models", {}).items())
    css = """:root{--bg:#0f1115;--panel:#181b22;--border:#2c3140;--text:#e6e9ef;--muted:#9098a6}
    body{background:var(--bg);color:var(--text);margin:0;padding:24px 16px;
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    .wrap{max-width:1000px;margin:0 auto} table{border-collapse:collapse;width:100%;font-size:13px}
    td,th{padding:5px 9px;text-align:left;border-bottom:1px solid var(--border)}
    th{color:var(--muted);font-weight:500} .sub{color:var(--muted)} .scroll{overflow-x:auto}"""
    n_bad = sum(c["label"] == "bad" for c in clips)
    n_ok = sum(c["label"] == "ok" for c in clips)
    doc = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Judge selection</title><style>{css}</style></head><body><div class='wrap'>"
           f"<h1>Stage 19 — judge &amp; ASR selection</h1><p class='sub'>{len(clips)} labeled clips "
           f"({n_bad} bad, {n_ok} ok); recall = bad clips flagged; false reject = ok clips flagged; "
           f"cap = {dec['fr_cap']:.0%}; flip cap {MAX_FLIP_RATE:.0%}. Available: "
           f"{esc(', '.join(dec['available']))}</p>"
           f"<div class='scroll'><table><tr><th>config</th><th>recall</th><th>false reject</th>"
           f"<th>flip</th><th>B agreement</th><th>eligible</th><th></th></tr>{''.join(trs)}</table></div>"
           f"<h2>ASR</h2><table><tr><th>model</th><th>pass rate</th><th>n</th><th>errors</th>"
           f"<th>agree w/ incumbent</th></tr>{asr_rows}</table>"
           f"<p class='sub'>ASR winner: {esc((asr or {}).get('winner', '—'))}</p></div></body></html>")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--configs", default="", help="comma-separated subset of config names")
    ap.add_argument("--skip-asr", action="store_true")
    args = ap.parse_args()
    clips = load_clips()
    configs = [c for c in CONFIGS if not args.configs or c[0] in args.configs.split(",")]
    n_calls = len(clips) * len(configs) * REPS
    print(f"=== Stage 19.2 — model selection === {len(clips)} labeled clips × {len(configs)} configs "
          f"× {REPS} reps = {n_calls} judge calls (est. ≤ ${n_calls * 0.004:.0f}) + "
          f"{0 if args.skip_asr else 300 * len(ASR_MODELS)} ASR calls (est. ≤ $2)")
    if args.dry_run:
        return 0
    if not args.yes:
        raise SelectionError("pass --yes to run (see the estimate above)")
    verdicts = judge_all(clips, configs)
    dec = decide(clips, verdicts, configs)
    asr = None if args.skip_asr else asr_check()
    write_outputs(clips, verdicts, configs, dec, asr)
    for key, m in sorted(dec["table"].items(), key=lambda kv: -kv[1]["recall"])[:12]:
        print(f"  {' AND '.join(key):<28} recall {m['recall']:.0%}  false-reject "
              f"{m['false_reject']:.0%}  flip {m['flip_rate']:.0%}  B {m['b_agreement']:.0%}")
    print(f"  winner: {dec['winner']}  fallback: {dec['runner_up']}  (false-reject cap "
          f"{dec['fr_cap']:.0%})")
    if asr:
        print(f"  ASR: {json.dumps(asr['models'])} → {asr['winner']}")
    print(f"  wrote {REPORT.relative_to(REPO_ROOT)}, {PINNED.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
