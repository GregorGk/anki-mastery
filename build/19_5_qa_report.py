"""Stage 19 / Step 6 — QA report for the eleven_v4 re-render + optional residue page.

Reads (read-only):
    data/_audio_manifest.tsv               current rows (after 19_4)
    data/_audio_manifest.tsv.bak_pre_stage19   v3 state before the run
    data/_audio_render_provenance.tsv      one row per accepted v4 upload
    data/_v4_unresolved.tsv                clips that kept their v3 row
    data/_v4_takes.tsv                     every take (pilot + full run)
Writes:
    reports/19_5_v4_qa.html                coverage, ladder depth, variants, swaps,
                                           judge/ASR failure reasons, credits
    reports/19_5_residue.html              (--residue) ≤ 10 unresolved clips, earliest
                                           studied first: v3 vs the best v4 take, your
                                           choice exported as data/_v4_residue_choices.tsv
Usage:
    uv run python build/19_5_qa_report.py [--residue]
"""
from __future__ import annotations

import argparse
import base64
import collections
import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import stage19_data as D  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
MANIFEST = DATA / "_audio_manifest.tsv"
BACKUP = DATA / "_audio_manifest.tsv.bak_pre_stage19"
PROVENANCE = DATA / "_audio_render_provenance.tsv"
UNRESOLVED = DATA / "_v4_unresolved.tsv"
TAKES = DATA / "_v4_takes.tsv"
REPORT = REPO_ROOT / "reports" / "19_5_v4_qa.html"
RESIDUE = REPO_ROOT / "reports" / "19_5_residue.html"
RESIDUE_MAX = 10

CSS = """:root{--bg:#0f1115;--panel:#181b22;--border:#2c3140;--text:#e6e9ef;--muted:#9098a6;
  --ok:#4ade80;--bad:#ff6b6b;--accent:#5aa9ff}
*{box-sizing:border-box}body{background:var(--bg);color:var(--text);margin:0;padding:24px 16px;
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:1000px;margin:0 auto}h1{font-size:21px;margin:0 0 4px}h2{font-size:16px;margin:22px 0 8px}
.sub{color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:5px 9px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}
th{color:var(--muted);font-weight:500}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.card b{display:block;font-size:20px}.scroll{overflow-x:auto}audio{height:30px;max-width:100%}
button{background:#232833;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{background:var(--accent);color:#0b0d10}"""


def _table(head: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f"<div class='scroll'><table><tr>{th}</tr>{body}</table></div>"


def report() -> None:
    man = read_tsv(MANIFEST)
    bp = [r for r in man if r["clip_type"] in ("word", "example")]
    on_v4 = [r for r in bp if r["tts_model"] == "eleven_v4"]
    prov = read_tsv(PROVENANCE)
    unres = read_tsv(UNRESOLVED)
    takes = read_tsv(TAKES)
    by_model = collections.Counter((r["clip_type"], r["tts_model"]) for r in bp)
    depth = collections.Counter(int(r["selected_from"] or 0) for r in prov)
    variants = collections.Counter(r["variant"] for r in prov)
    backup = {(r["sense_id"], r["clip_type"]): r for r in read_tsv(BACKUP)}
    swaps = sum(1 for r in prov if (r["sense_id"], r["clip_type"]) in backup
                and backup[(r["sense_id"], r["clip_type"])]["voice_id"] != r["voice_id"])
    fail = collections.Counter()
    for t in takes:
        if t["error"]:
            fail["error"] += 1
        elif t["sanity"]:
            fail["sanity:" + t["sanity"].split("_")[0]] += 1
        elif t["loudness"]:
            fail["loudness"] += 1
        elif t["asr_pass"] != "1":
            fail["asr"] += 1
        elif t["gate_pass"] != "1":
            for k, v in json.loads(t["judges"] or "{}").items():
                if v.get("verdict") != "bp_ok":
                    fail[f"judge:{k}:{v.get('verdict')}/{v.get('drift', '')}"] += 1
    credits = sum(int(float(t["credits"] or 0)) for t in takes)
    names = D.voice_names()
    per_voice = collections.Counter(names.get(r["voice_id"], r["voice_id"]) for r in on_v4)
    cards = [("BP clips on eleven_v4", f"{len(on_v4):,} / {len(bp):,}"),
             ("accepted uploads", f"{len(prov):,}"), ("unresolved (v3 kept)", f"{len(unres):,}"),
             ("voice swaps", f"{swaps:,}"), ("takes rendered", f"{len(takes):,}"),
             ("credits (header sum)", f"{credits:,}")]
    doc = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>v4 QA report</title><style>{CSS}</style></head><body><div class='wrap'>"
           f"<h1>Stage 19 — eleven_v4 QA</h1><div class='grid'>"
           + "".join(f"<div class='card'><b>{v}</b>{html.escape(k)}</div>" for k, v in cards)
           + "</div><h2>Clips by model</h2>"
           + _table(["clip", "model", "n"], [[k[0], k[1], n] for k, n in sorted(by_model.items())])
           + "<h2>Takes needed before acceptance</h2>"
           + _table(["takes rendered for the clip", "clips"], sorted(depth.items()))
           + "<h2>Accepted variant</h2>" + _table(["variant", "clips"], variants.most_common())
           + "<h2>v4 clips per voice</h2>" + _table(["voice", "clips"], per_voice.most_common())
           + "<h2>Why takes failed</h2>" + _table(["reason", "takes"], fail.most_common(40))
           + "<h2>Unresolved (v3 kept)</h2>"
           + _table(["sense", "clip", "attempts", "reasons"],
                    [[r["sense_id"], r["clip_type"], r["attempts"], r["reasons"]] for r in unres])
           + "</div></body></html>")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(doc, encoding="utf-8")
    print(f"  {len(on_v4):,}/{len(bp):,} BP clips on v4, {len(unres):,} unresolved, "
          f"{credits:,} credits → {REPORT.relative_to(REPO_ROOT)}")


def residue() -> None:
    jobs = D.load_jobs()
    takes = collections.defaultdict(list)
    for t in read_tsv(TAKES):
        if t["path"] and Path(t["path"]).is_file():
            takes[(t["sense_id"], t["clip_type"])].append(t)
    unres = sorted(read_tsv(UNRESOLVED), key=lambda r: jobs[(r["sense_id"], r["clip_type"])].order)
    cards = []
    for n, r in enumerate(unres[:RESIDUE_MAX], 1):
        j = jobs[(r["sense_id"], r["clip_type"])]
        cands = sorted(takes[j.key], key=lambda t: (int(t["asr_pass"] == "1"),
                                                     float(t["asr_similarity"] or 0)),
                       reverse=True)[:2]
        opts = [("keep_v3", str(j.v3_path))] + [(f"take:{t['take_id']}", t["path"]) for t in cands]
        players = "".join(
            f"<div><button data-c='{html.escape(c)}'>{'Keep v3' if c == 'keep_v3' else f'v4 option {i}'}"
            f"</button> <audio controls preload='none' src='data:audio/mpeg;base64,"
            f"{base64.b64encode(Path(p).read_bytes()).decode()}'></audio></div>"
            for i, (c, p) in enumerate(opts))
        cards.append(f"<div class='card' data-sid='{j.sense_id}' data-ct='{j.clip_type}'>"
                     f"<b>{n}. {html.escape(j.display_text)}</b><span class='sub'>{j.clip_type} · "
                     f"{html.escape(r['reasons'])}</span>{players}</div>")
    js = """<script>const st={};document.querySelectorAll('.card').forEach(c=>{
c.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
st[c.dataset.sid+'|'+c.dataset.ct]=b.dataset.c;c.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b))}))});
document.getElementById('exp').addEventListener('click',()=>{const L=['sense_id\\tclip_type\\tchoice'];
for(const k in st){const [s,t]=k.split('|');L.push([s,t,st[k]].join('\\t'))}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([L.join('\\n')+'\\n']));
a.download='_v4_residue_choices.tsv';a.click()})</script>"""
    RESIDUE.write_text(
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>v4 residue</title><style>{CSS}"
        f".card{{margin:10px 0}}</style></head><body><div class='wrap'><h1>Clips v4 couldn't fix</h1>"
        f"<p class='sub'>Optional. Pick per clip, then export as data/_v4_residue_choices.tsv.</p>"
        f"<button id='exp'>Export TSV</button>{''.join(cards)}</div>{js}</body></html>",
        encoding="utf-8")
    print(f"  residue page: {len(cards)} clips → {RESIDUE.relative_to(REPO_ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--residue", action="store_true")
    args = ap.parse_args()
    print("=== Stage 19.5 — v4 QA report ===")
    report()
    if args.residue:
        residue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
