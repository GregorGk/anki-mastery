"""Stage 18 / Step 3 — Render the learner-flag repair queue as HTML.

Reads:   data/_learner_flags.tsv             (from 18_2; may be absent/empty)
         data/07-anki-listening-general.tsv   (enrichment by sense_id)
         data/07-anki-media-manifest.tsv      (basename → R2 url, for playback)
Writes:  reports/18_learner_flags.html

Each flagged card becomes a row with the audio (streamed from R2 so you
can re-listen), the three IPA-relevant fields, risk/usage notes, the
inferred issue scope, and your status/notes. This is the queue a later
regeneration stage works through.

Run AFTER build/18_2_pull_anki_flags.py (degrades gracefully if no flags
have been pulled yet).

Usage:
    .venv/bin/python build/18_3_flags_html.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.anki_pilot import read_anki_export  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
FLAGS_TSV = DATA / "_learner_flags.tsv"
ANKI_TSV = DATA / "07-anki-listening-general.tsv"
MANIFEST = DATA / "07-anki-media-manifest.tsv"
REPORT = REPO_ROOT / "reports" / "18_learner_flags.html"

CARD_TONE = {"listening": "accent", "recall": "ok"}

CSS = """
:root{--bg:#0f1115;--panel:#181b22;--panel-2:#20242d;--border:#2c3140;
  --text:#e6e9ef;--muted:#9098a6;--accent:#5aa9ff;--ok:#4ade80;
  --bad:#ff6b6b;--warn:#fbbf24;--mono:ui-monospace,Menlo,monospace;}
*{box-sizing:border-box;}
body{background:var(--bg);color:var(--text);margin:0;padding:24px;
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
.wrap{max-width:1080px;margin:0 auto;}
h1{font-size:21px;margin:0 0 4px;} .sub{color:var(--muted);margin:0 0 18px;}
.empty{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:24px;text-align:center;color:var(--muted);}
code{background:var(--panel-2);padding:1px 5px;border-radius:3px;font-size:12px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:12px 16px;margin-bottom:10px;display:grid;
  grid-template-columns:1fr 280px;gap:16px;}
.head{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px;}
.head .pt{font-size:18px;font-weight:600;} .head .id{font-family:var(--mono);
  font-size:12px;color:var(--muted);} .head .en{color:var(--muted);font-size:13px;}
.badge{font-size:11px;padding:2px 8px;border-radius:3px;font-weight:600;
  text-transform:uppercase;letter-spacing:.3px;}
.badge.accent{background:rgba(90,169,255,.18);color:var(--accent);}
.badge.ok{background:rgba(74,222,128,.18);color:var(--ok);}
.badge.muted{background:var(--panel-2);color:var(--muted);}
.badge.red{background:rgba(255,107,107,.2);color:var(--bad);}
.ex{font-size:13px;margin:4px 0;} .ex .pt-ex{font-style:italic;}
.meta{font-size:12px;color:var(--muted);margin-top:6px;}
.meta b{color:var(--text);font-weight:600;}
.risk{color:var(--bad);} .usage{font-style:italic;}
.right{display:flex;flex-direction:column;gap:6px;}
.right audio{width:100%;height:30px;}
.alab{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;}
.notes{font-family:var(--mono);font-size:12px;color:var(--warn);}
table.summary{border-collapse:collapse;margin:0 0 16px;font-size:13px;}
table.summary td,table.summary th{padding:4px 12px;border-bottom:1px solid var(--border);text-align:left;}
"""


def _esc(v) -> str:
    return html.escape(str(v if v is not None else ""))


def _sound_basename(v: str) -> str:
    v = v or ""
    if v.startswith("[sound:") and v.endswith("]"):
        return v[len("[sound:"):-1]
    return ""


def main() -> int:
    flags = read_tsv(FLAGS_TSV)
    enrich = {r["sense_id"]: r for r in
              (read_anki_export(ANKI_TSV) if ANKI_TSV.exists() else [])}
    url_by_base = {r["media_filename"]: r.get("url", "")
                   for r in read_tsv(MANIFEST)}

    p: list[str] = []
    p.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    p.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    p.append("<title>Stage 18 — learner flag repair queue</title>")
    p.append(f"<style>{CSS}</style></head><body><div class='wrap'>")
    p.append("<h1>Stage 18 — learner flag repair queue</h1>")

    if not flags:
        p.append(
            "<p class='sub'>The repair queue from cards you red-flagged "
            "while learning.</p>")
        p.append(
            "<div class='empty'>No flags pulled yet.<br><br>Flag bad cards "
            "red in Anki, then run "
            "<code>build/18_2_pull_anki_flags.py --ankiconnect</code> "
            "(or <code>--from-export</code>), then re-run this report.</div>")
        p.append("</div></body></html>")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("".join(p), encoding="utf-8")
        print(f"=== Stage 18.3 — flags report ===\n  no flags yet → "
              f"{REPORT.relative_to(REPO_ROOT)}")
        return 0

    from collections import Counter
    by_ct = Counter(f.get("card_type") or "?" for f in flags)
    by_scope = Counter(f.get("issue_scope") or "?" for f in flags)
    p.append(f"<p class='sub'>{len(flags)} flagged cards · "
             f"newest pull {_esc(max((f.get('pulled_at','') for f in flags), default=''))}</p>")
    p.append("<table class='summary'><tr><th>by card type</th><th>by issue scope</th></tr><tr><td>")
    p.append("<br>".join(f"{_esc(k)}: {v}" for k, v in by_ct.most_common()))
    p.append("</td><td>")
    p.append("<br>".join(f"{_esc(k)}: {v}" for k, v in by_scope.most_common()))
    p.append("</td></tr></table>")

    for f in flags:
        sid = f["sense_id"]
        e = enrich.get(sid, {})
        ct = f.get("card_type", "")
        tone = CARD_TONE.get(ct, "muted")
        pt = e.get("pt_display_safe") or f.get("pt", "") or e.get("pt", "")
        en = e.get("en_primary") or f.get("en_primary", "")
        risk = e.get("risk_note", "")
        usage = e.get("usage_hint", "")
        players = []
        for col, lab in (("audio_word", "word"), ("audio_example", "example")):
            base = _sound_basename(e.get(col, ""))
            url = url_by_base.get(base, "")
            if url:
                players.append(
                    f"<div><span class='alab'>{lab}</span>"
                    f"<audio controls preload='none' src='{_esc(url)}'></audio></div>")
        p.append("<div class='card'><div class='left'>")
        p.append(
            f"<div class='head'><span class='pt'>{_esc(pt)}</span>"
            f"<span class='id'>{_esc(sid)}</span>"
            f"<span class='en'>— {_esc(en)}</span>"
            f"<span class='badge {tone}'>{_esc(ct or '?')}</span>"
            f"<span class='badge red'>{_esc(f.get('flag_color','red'))}</span></div>")
        if e.get("example_pt"):
            p.append(f"<div class='ex'><span class='pt-ex'>{_esc(e['example_pt'])}</span>"
                     f"<br><span class='muted'>{_esc(e.get('example_en',''))}</span></div>")
        if risk:
            p.append(f"<div class='meta risk'>⚠ {_esc(risk)}</div>")
        if usage:
            p.append(f"<div class='meta usage'>{_esc(usage)}</div>")
        p.append(
            f"<div class='meta'>scope <b>{_esc(f.get('issue_scope','') or '?')}</b>"
            f" · status <b>{_esc(f.get('status','') or 'open')}</b>"
            f" · deck {_esc(f.get('deck','') or '—')}</div>")
        if f.get("notes"):
            p.append(f"<div class='meta notes'>{_esc(f['notes'])}</div>")
        p.append("</div><div class='right'>")
        p.append("".join(players) if players else "<span class='alab'>no audio</span>")
        p.append("</div></div>")

    p.append("</div></body></html>")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(p), encoding="utf-8")
    print(f"=== Stage 18.3 — flags report ===")
    print(f"  flags: {len(flags)}")
    print(f"  wrote: {REPORT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
