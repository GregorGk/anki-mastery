"""Stage 12 / Step 1 — Render audit/12_0_usage_hint_pilot.html (v2).

Joins data/_usage_hints.tsv with data/06-final.tsv. Shows hint_priority
(essential / useful / omit), usage_hint, confidence, reason, plus full
row context. Filter buttons: all / essential / useful / omit / low-conf.
"""
from __future__ import annotations

import csv
import html as _html
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIDECAR = REPO / "data" / "_usage_hints.tsv"
FINAL = REPO / "data" / "06-final.tsv"
OUT = REPO / "audit" / "12_0_usage_hint_pilot.html"


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> int:
    if not SIDECAR.exists():
        print(f"Missing {SIDECAR}", file=sys.stderr)
        return 1
    sidecar = read_tsv(SIDECAR)
    final_by_sid = {r["sense_id"]: r for r in read_tsv(FINAL)}

    pri_counts: dict[str, int] = {"essential": 0, "useful": 0, "omit": 0}
    conf_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    n_risk = 0
    rows_html: list[str] = []
    for s in sidecar:
        sid = s["sense_id"]
        f = final_by_sid.get(sid, {})
        hint = s.get("usage_hint", "").strip()
        priority = s.get("hint_priority", "")
        conf = s.get("confidence", "")
        reason = s.get("reason", "")
        risk = s.get("risk_note", "").strip()
        pos = s.get("pos", "")
        pt = s.get("pt", "")

        if risk:
            n_risk += 1

        if priority in pri_counts:
            pri_counts[priority] += 1
        if conf in conf_counts:
            conf_counts[conf] += 1

        pri_badge = {
            "essential": '<span class="badge ess">essential</span>',
            "useful":    '<span class="badge use">useful</span>',
            "omit":      '<span class="badge omt">omit</span>',
        }.get(priority, _html.escape(priority))

        conf_badge_cls = (
            "ok" if conf == "high"
            else "warn" if conf == "medium"
            else "no" if conf == "low"
            else ""
        )

        row_cls = priority or "unknown"
        rows_html.append(f"""
        <tr class="{row_cls}"
            data-priority="{_html.escape(priority)}"
            data-conf="{_html.escape(conf)}">
          <td><code>{_html.escape(sid)}</code><br>
              <span class="muted small">rank {_html.escape(f.get("rank","?"))}</span></td>
          <td><strong>{_html.escape(pt)}</strong><br>
              <span class="muted small">{_html.escape(pos)}</span></td>
          <td><span class="muted small">{_html.escape(f.get("en_primary",""))}</span><br>
              <span class="muted xs">{_html.escape(f.get("en_all",""))[:120]}</span></td>
          <td>{_html.escape(f.get("example_pt",""))}<br>
              <em>{_html.escape(f.get("example_en",""))}</em><br>
              <span class="muted xs">target: <code>{_html.escape(f.get("target_word_used",""))}</code></span></td>
          <td class="hint-cell">{'<strong>' + _html.escape(hint) + '</strong>' if hint else '<span class="muted">(empty)</span>'}{('<br><span class="risk">⚠ ' + _html.escape(risk) + '</span>') if risk else ''}</td>
          <td>{pri_badge}</td>
          <td><span class="badge {conf_badge_cls}">{_html.escape(conf)}</span></td>
          <td class="reason">{_html.escape(reason)}</td>
        </tr>""")

    n = len(sidecar)
    n_hint = pri_counts["essential"] + pri_counts["useful"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 12.0 v2 — usage_hint ({n} candidates)</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1750px; margin: 0 auto; padding: 0 1em; line-height: 1.4; color: #222; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
.toolbar {{ position: sticky; top: 0; background: #ffd; padding: 0.7em 1em;
  border: 2px solid #ba0; border-radius: 4px; margin: 0.5em 0;
  display: flex; align-items: center; gap: 1.5em; flex-wrap: wrap; z-index: 100;
  font-size: 0.92em; }}
.metric {{ display: inline-block; }}
.metric strong {{ font-size: 1.1em; color: #06598f; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; }}
th, td {{ border: 1px solid #aaa; padding: 0.4em 0.6em; text-align: left; vertical-align: top; }}
th {{ background: #e6ecf2; position: sticky; top: 3em; z-index: 5; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.92em; }}
.muted {{ color: #888; }}
.small {{ font-size: 0.85em; }}
.xs {{ font-size: 0.75em; }}
em {{ color: #06598f; font-style: italic; }}
tr.essential .hint-cell {{ background: #d4edda; }}
tr.useful    .hint-cell {{ background: #e8f4ff; }}
tr.omit      .hint-cell {{ background: #fafafa; }}
.badge {{ display: inline-block; padding: 0.12em 0.5em; border-radius: 3px;
  font-size: 0.78em; font-weight: bold; white-space: nowrap; }}
.badge.ess {{ background: #198754; color: #fff; }}
.badge.use {{ background: #0d6efd; color: #fff; }}
.badge.omt {{ background: #eee; color: #555; }}
.badge.ok {{ background: #d4edda; color: #155724; }}
.badge.warn {{ background: #fff3cd; color: #664d03; }}
.badge.no {{ background: #fde2e2; color: #842029; }}
.reason {{ font-size: 0.85em; color: #444; max-width: 280px; }}
.risk {{ display: inline-block; margin-top: 0.3em; padding: 0.1em 0.4em;
  border-radius: 3px; background: #fde2e2; color: #842029; font-size: 0.82em; }}
button {{ font-size: 0.92em; padding: 0.3em 0.7em; cursor: pointer; }}
button.active {{ background: #06598f; color: white; }}
</style>
</head>
<body>

<h1>Stage 12.0 v2 — usage_hint candidates ({n} rows, filtered)</h1>

<div class="toolbar">
  <span class="metric">Candidates: <strong>{n}</strong></span>
  <span class="metric"><strong>essential</strong> {pri_counts['essential']} · <strong>useful</strong> {pri_counts['useful']} · <strong>omit</strong> {pri_counts['omit']}</span>
  <span class="metric">non-empty hints: <strong>{n_hint}</strong> ({100*n_hint//max(n,1)}%)</span>
  <span class="metric">risk_note: <strong>{n_risk}</strong></span>
  <span class="metric">conf <strong>high</strong> {conf_counts['high']} · <strong>med</strong> {conf_counts['medium']} · <strong>low</strong> {conf_counts['low']}</span>
  <span style="margin-left: auto;">filter:</span>
  <button onclick="filt('all', this)" class="active">all</button>
  <button onclick="filt('essential', this)">essential</button>
  <button onclick="filt('useful', this)">useful</button>
  <button onclick="filt('omit', this)">omit</button>
  <button onclick="filt('low', this)">low conf</button>
</div>

<table>
  <thead>
    <tr>
      <th>sense_id</th>
      <th>lemma · pos</th>
      <th>EN gloss</th>
      <th>example (pt / en)</th>
      <th>usage_hint</th>
      <th>priority</th>
      <th>conf</th>
      <th>reason</th>
    </tr>
  </thead>
  <tbody>{"".join(rows_html)}
  </tbody>
</table>

<script>
function filt(mode, btn) {{
  document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('tbody tr').forEach(r => {{
    const pri = r.getAttribute('data-priority');
    const conf = r.getAttribute('data-conf');
    let show = true;
    if (mode === 'essential') show = (pri === 'essential');
    else if (mode === 'useful') show = (pri === 'useful');
    else if (mode === 'omit') show = (pri === 'omit');
    else if (mode === 'low') show = (conf === 'low');
    r.style.display = show ? '' : 'none';
  }});
}}
</script>
</body>
</html>
""", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  total: {n}  | essential: {pri_counts['essential']}  "
          f"useful: {pri_counts['useful']}  omit: {pri_counts['omit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
