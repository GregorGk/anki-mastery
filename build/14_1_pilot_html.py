"""Stage 14 / Step 1 — Generate the pilot review HTML.

Reads `data/_topic_tags.pilot.tsv` and writes
`reports/14_topic_tags_pilot.html`. Self-contained (inline CSS, no external
assets). Reviewers open the file directly in a browser.

Sections (from plan §"Pilot HTML"):
  - Top-line metrics: total rows, deterministic vs LLM, low-conf rate,
    largest bucket, topic-collapse warnings
  - Topic distribution table (50 rows, counts + bars)
  - Confidence distribution
  - All low-confidence rows (table)
  - All 45 numerals (deterministic check)
  - 50 random rows (seed-stable)
  - Top 20 #topic-daily-routines rows (verb catchall)
  - Top 20 #topic-objects-tools rows (noun catchall)
  - Top 30 #topic-character-qualities rows (generic-adj fallback, largest
    drift surface)

Usage:
  .venv/bin/python build/14_1_pilot_html.py
"""
from __future__ import annotations

import argparse
import html
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.topic_tag_rules import ALLOWED_TOPIC_TAGS  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
PILOT_TSV = DATA_DIR / "_topic_tags.pilot.tsv"
PILOT_HTML = REPORTS_DIR / "14_topic_tags_pilot.html"

# Soft-warning thresholds (must mirror verify_all.py)
COVERAGE_DAILY_ROUTINES_CAP = 0.15
COVERAGE_OBJECTS_TOOLS_CAP = 0.15
COVERAGE_WORK_JOBS_CAP = 0.15
COVERAGE_CHARACTER_QUALITIES_CAP = 0.15
COVERAGE_MEASUREMENT_CAP = 0.12
COVERAGE_ANY_TOPIC_CAP = 0.25
LOW_CONF_CAP = 0.05


CSS = """
body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI",
       Helvetica, Arial, sans-serif; max-width: 1200px; margin: 1.5em auto;
       padding: 0 1em; color: #222; }
h1 { font-size: 1.6em; margin: 0.2em 0 0.4em 0; }
h2 { font-size: 1.2em; margin: 1.4em 0 0.3em 0; border-bottom: 1px solid #ddd;
     padding-bottom: 0.15em; }
.kpi { display: inline-block; padding: 0.4em 0.7em; margin-right: 0.6em;
       background: #f3f4f6; border-radius: 4px; font-size: 0.95em; }
.warn { background: #fef3c7; }
.fail { background: #fee2e2; }
.ok   { background: #d1fae5; }
table { border-collapse: collapse; width: 100%; margin: 0.3em 0; }
th, td { padding: 4px 8px; border-bottom: 1px solid #eee; text-align: left;
         vertical-align: top; }
th { background: #f8f8f8; font-weight: 600; }
tr.det td { background: #fafafa; }
tr.low_conf td { background: #fef9c3; }
tr.missing_topic td { background: #fee2e2; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px;
       font-size: 0.92em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.bar { display: inline-block; height: 10px; background: #6366f1;
       vertical-align: middle; margin-right: 4px; }
.confidence-high   { color: #047857; font-weight: 600; }
.confidence-medium { color: #b45309; font-weight: 600; }
.confidence-low    { color: #b91c1c; font-weight: 600; }
.example { color: #6b7280; font-style: italic; font-size: 0.92em; }
nav { margin: 1em 0 1.6em 0; padding: 0.5em 0.8em; background: #f3f4f6;
      border-radius: 4px; }
nav a { margin-right: 0.8em; color: #4338ca; text-decoration: none; }
nav a:hover { text-decoration: underline; }
"""


def _esc(s: str) -> str:
    return html.escape(s or "")


def _bar_html(count: int, max_count: int, width_px: int = 120) -> str:
    if max_count <= 0:
        return ""
    width = max(1, int(count / max_count * width_px))
    return f'<span class="bar" style="width: {width}px"></span>'


def _conf_class(conf: str) -> str:
    return f"confidence-{conf}" if conf in {"high", "medium", "low"} else ""


def _row_class(r: dict, missing_topic_sids: set[str]) -> str:
    classes = []
    if r["sense_id"] in missing_topic_sids:
        classes.append("missing_topic")
    elif (r.get("source") or "").strip() == "deterministic":
        classes.append("det")
    elif (r.get("confidence") or "").strip().lower() != "high":
        classes.append("low_conf")
    return " ".join(classes)


def _row_html(r: dict, missing_topic_sids: set[str]) -> str:
    cls = _row_class(r, missing_topic_sids)
    conf = (r.get("confidence") or "").strip().lower()
    example_pt = (r.get("example_pt") or "").strip()
    example_en = (r.get("example_en") or "").strip()
    ex_html = ""
    if example_pt or example_en:
        ex_html = (
            f'<div class="example"><b>pt:</b> {_esc(example_pt)}'
            f'<br><b>en:</b> {_esc(example_en)}</div>'
        )
    return (
        f'<tr class="{cls}">'
        f'<td>{_esc(r["sense_id"])}</td>'
        f'<td>{_esc(r.get("rank", ""))}</td>'
        f'<td><b>{_esc(r.get("pt", ""))}</b></td>'
        f'<td>{_esc(r.get("pos", ""))}</td>'
        f'<td>{_esc(r.get("en_primary", ""))}</td>'
        f'<td><code>{_esc(r.get("topic_primary", ""))}</code></td>'
        f'<td class="{_conf_class(conf)}">{_esc(conf)}</td>'
        f'<td>{_esc(r.get("source", ""))}</td>'
        f'<td>{_esc(r.get("reason", ""))}{ex_html}</td>'
        f'</tr>'
    )


def _table(rows: list[dict], missing_topic_sids: set[str], caption: str = "") -> str:
    if not rows:
        return '<p><em>(no rows in this section)</em></p>'
    head = (
        '<table><thead><tr>'
        '<th>sense_id</th><th>rank</th><th>pt</th><th>pos</th>'
        '<th>en_primary</th><th>topic</th><th>conf</th><th>src</th>'
        '<th>reason / example</th>'
        '</tr></thead><tbody>'
    )
    body = "\n".join(_row_html(r, missing_topic_sids) for r in rows)
    return f'{head}{body}</tbody></table>'


def _build_report(rows: list[dict]) -> str:
    total = len(rows)
    if total == 0:
        return "<html><body><h1>Pilot report</h1><p>No rows in sidecar.</p></body></html>"

    # Classify each row by section.
    missing_topic_sids = {r["sense_id"] for r in rows
                          if not (r.get("topic_primary") or "").strip()}

    by_topic = Counter()
    by_conf = Counter()
    by_source = Counter()
    for r in rows:
        by_topic[(r.get("topic_primary") or "").strip()] += 1
        by_conf[(r.get("confidence") or "").strip().lower()] += 1
        by_source[(r.get("source") or "").strip()] += 1

    low_conf_rows = sorted(
        [r for r in rows if (r.get("confidence") or "").strip().lower() != "high"],
        key=lambda r: r["sense_id"],
    )

    numerals = sorted(
        [r for r in rows if (r.get("pos") or "").strip() == "num"],
        key=lambda r: int(r.get("rank") or 0),
    )

    rng = random.Random(14)
    sampled = rng.sample(rows, min(50, total))
    sampled.sort(key=lambda r: r["sense_id"])

    def top_n(topic: str, n: int) -> list[dict]:
        return sorted(
            [r for r in rows if (r.get("topic_primary") or "").strip() == topic],
            key=lambda r: int(r.get("rank") or 0),
        )[:n]

    top_daily = top_n("#topic-daily-routines", 20)
    top_objects = top_n("#topic-objects-tools", 20)
    top_charq = top_n("#topic-character-qualities", 30)

    # Topic distribution counts (incl. zero-coverage tags).
    max_topic_count = max(by_topic.values()) if by_topic else 1
    dist_rows = []
    for tag in ALLOWED_TOPIC_TAGS:
        c = by_topic.get(tag, 0)
        pct = (c / total) * 100 if total else 0
        warn = ""
        if c == 0:
            warn = ' <span class="kpi warn">zero-coverage</span>'
        elif (tag == "#topic-daily-routines"
              and c / total > COVERAGE_DAILY_ROUTINES_CAP):
            warn = ' <span class="kpi warn">over daily-routines cap</span>'
        elif (tag == "#topic-objects-tools"
              and c / total > COVERAGE_OBJECTS_TOOLS_CAP):
            warn = ' <span class="kpi warn">over objects-tools cap</span>'
        elif (tag == "#topic-work-jobs"
              and c / total > COVERAGE_WORK_JOBS_CAP):
            warn = ' <span class="kpi warn">over work-jobs cap</span>'
        elif (tag == "#topic-character-qualities"
              and c / total > COVERAGE_CHARACTER_QUALITIES_CAP):
            warn = ' <span class="kpi warn">over character-qualities cap</span>'
        elif (tag == "#topic-measurement"
              and c / total > COVERAGE_MEASUREMENT_CAP):
            warn = ' <span class="kpi warn">over measurement cap</span>'
        elif c / total > COVERAGE_ANY_TOPIC_CAP:
            warn = ' <span class="kpi fail">over 25% anti-collapse cap</span>'
        dist_rows.append(
            f'<tr><td><code>{_esc(tag)}</code></td>'
            f'<td style="text-align:right">{c}</td>'
            f'<td style="text-align:right">{pct:.1f}%</td>'
            f'<td>{_bar_html(c, max_topic_count)}</td>'
            f'<td>{warn}</td></tr>'
        )

    # KPI line — top-level acceptance signals.
    # `low_conf` counts ONLY confidence == "low" (the "guessing" signal).
    # `medium_conf` is informational — it's where the prompt's "broadest
    # honest topic" fallback fired on generic verbs/adjectives.
    low_n = by_conf.get("low", 0)
    medium_n = by_conf.get("medium", 0)
    low_pct = low_n / total if total else 0
    medium_pct = medium_n / total if total else 0
    invalid = [r for r in rows if (r.get("topic_primary") or "").strip()
               and (r.get("topic_primary") or "").strip() not in set(ALLOWED_TOPIC_TAGS)]

    kpi_low_conf = (
        f'<span class="kpi {"fail" if low_pct > LOW_CONF_CAP else "ok"}">'
        f'low_conf: {low_n} ({low_pct:.1%})</span>'
    )
    kpi_medium_conf = (
        f'<span class="kpi">medium_conf: {medium_n} ({medium_pct:.1%}) — informational</span>'
    )
    kpi_invalid = f'<span class="kpi {"fail" if invalid else "ok"}">'\
                  f'invalid_tags: {len(invalid)}</span>'
    kpi_missing = f'<span class="kpi {"fail" if missing_topic_sids else "ok"}">'\
                  f'missing_topic: {len(missing_topic_sids)}</span>'
    kpi_total = f'<span class="kpi">total: {total}</span>'
    kpi_det = f'<span class="kpi">deterministic: {by_source.get("deterministic", 0)}</span>'
    kpi_llm = f'<span class="kpi">llm: {by_source.get("llm", 0)}</span>'
    kpi_manual = (
        f'<span class="kpi">manual: {by_source.get("manual", 0)}</span>'
        if by_source.get("manual", 0) else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stage 14 — Topic Tag Pilot Review</title>
<style>{CSS}</style>
</head>
<body>
<h1>Stage 14 — Topic Tag Pilot Review</h1>

<div>
{kpi_total} {kpi_det} {kpi_llm} {kpi_manual} {kpi_low_conf} {kpi_medium_conf} {kpi_invalid} {kpi_missing}
</div>

<nav>
<a href="#distribution">Distribution</a>
<a href="#numerals">Numerals (45)</a>
<a href="#low-conf">Low-conf rows</a>
<a href="#random">50 random</a>
<a href="#daily-routines">Top daily-routines</a>
<a href="#objects-tools">Top objects-tools</a>
<a href="#character-qualities">Top character-qualities</a>
</nav>

<h2 id="distribution">Topic distribution (50 buckets)</h2>
<table>
<thead><tr><th>tag</th><th>count</th><th>%</th><th></th><th>warn</th></tr></thead>
<tbody>{"".join(dist_rows)}</tbody>
</table>

<h2 id="numerals">All numerals (deterministic pos=num — must all be <code>#topic-numbers</code>)</h2>
{_table(numerals, missing_topic_sids)}

<h2 id="low-conf">All low-confidence rows</h2>
{_table(low_conf_rows, missing_topic_sids)}

<h2 id="random">50 random rows (seed-stable)</h2>
{_table(sampled, missing_topic_sids)}

<h2 id="daily-routines">Top 20 #topic-daily-routines rows (verb catchall)</h2>
{_table(top_daily, missing_topic_sids)}

<h2 id="objects-tools">Top 20 #topic-objects-tools rows (noun catchall)</h2>
{_table(top_objects, missing_topic_sids)}

<h2 id="character-qualities">Top 30 #topic-character-qualities rows (generic-adj fallback)</h2>
{_table(top_charq, missing_topic_sids)}

</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=PILOT_TSV)
    ap.add_argument("--output", type=Path, default=PILOT_HTML)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: pilot sidecar missing at {args.input}", file=sys.stderr)
        print("Run build/14_0_topic_classifier.py --pilot 500 --seed 14 first.",
              file=sys.stderr)
        return 1

    rows = read_tsv(args.input)
    html_doc = _build_report(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows, "
          f"{args.output.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
