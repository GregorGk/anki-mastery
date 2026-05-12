"""Stage 15 / Step 1 — Pilot review HTML.

Reads `data/_risk_register.pilot.tsv` and writes
`reports/15_risk_register_pilot.html`. Self-contained (inline CSS).

Sections:
  - KPI line: total / manual / llm / llm_audit_only / low_conf / medium_conf
  - bp_validity distribution
  - register distribution
  - risk_flags distribution (split per flag)
  - All Stage-12 carry-forward risk_note rows (must not be empty)
  - All bp_status=nsfw rows (spot-check: register + flags)
  - All bp_status=false_friend rows (spot-check: false_friend flag)
  - All 4 manual-override prefill rows (verify values shipped as expected)
  - Random-normal audit subset: 100 LLM-classified normals (should be
    standard/neutral/none/"")
  - All low-confidence rows
"""
from __future__ import annotations

import argparse
import html
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.risk_register_rules import (  # noqa: E402
    ALLOWED_BP_VALIDITY,
    ALLOWED_REGISTER,
    ALLOWED_RISK_FLAGS,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
PILOT_TSV = DATA_DIR / "_risk_register.pilot.tsv"
PILOT_HTML = REPORTS_DIR / "15_risk_register_pilot.html"

# Pilot acceptance thresholds (from plan §"Pilot acceptance").
LOW_CONF_CAP = 0.05
UNCERTAIN_VALIDITY_CAP = 0.01
UNCERTAIN_REGISTER_CAP = 0.01

# Sense_ids prefilled in data/_manual_risk_register.tsv (see plan).
MANUAL_PREFILL_SIDS = ("1600.00.01", "2124.00.01", "2124.00.02", "2751.00.01")

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
tr.manual td { background: #e0e7ff; }
tr.audit td { background: #fef9c3; }
tr.flagged td { background: #fee2e2; }
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


def _bar_html(count: int, max_count: int, width_px: int = 200) -> str:
    if max_count <= 0:
        return ""
    width = max(1, int(count / max_count * width_px))
    return f'<span class="bar" style="width: {width}px"></span>'


def _conf_class(conf: str) -> str:
    return f"confidence-{conf}" if conf in {"high", "medium", "low"} else ""


def _row_class(r: dict) -> str:
    src = (r.get("source") or "").strip()
    if src == "manual":
        return "manual"
    if src == "llm_audit_only":
        return "audit"
    note = (r.get("risk_note") or "").strip()
    flags = (r.get("risk_flags") or "").strip()
    if note and flags == "none":
        return "flagged"  # verifier H8 violation
    return ""


def _row_html(r: dict) -> str:
    cls = _row_class(r)
    conf = (r.get("confidence") or "").strip().lower()
    ex_pt = (r.get("example_pt") or "").strip()
    ex_en = (r.get("example_en") or "").strip()
    ex_html = ""
    if ex_pt or ex_en:
        ex_html = (
            f'<div class="example"><b>pt:</b> {_esc(ex_pt)}'
            f'<br><b>en:</b> {_esc(ex_en)}</div>'
        )
    return (
        f'<tr class="{cls}">'
        f'<td>{_esc(r["sense_id"])}</td>'
        f'<td>{_esc(r.get("pt", ""))}</td>'
        f'<td>{_esc(r.get("pos", ""))}</td>'
        f'<td>{_esc(r.get("en_primary", ""))}</td>'
        f'<td>{_esc(r.get("bp_status", ""))}</td>'
        f'<td><code>{_esc(r.get("bp_validity", ""))}</code></td>'
        f'<td><code>{_esc(r.get("register", ""))}</code></td>'
        f'<td><code>{_esc(r.get("risk_flags", ""))}</code></td>'
        f'<td>{_esc(r.get("risk_note", ""))}</td>'
        f'<td class="{_conf_class(conf)}">{_esc(conf)}</td>'
        f'<td>{_esc(r.get("source", ""))}</td>'
        f'<td>{_esc(r.get("reason", ""))[:120]}{ex_html}</td>'
        f'</tr>'
    )


def _table(rows: list[dict], caption: str = "") -> str:
    if not rows:
        return f'<p><em>(no rows in this section)</em></p>'
    head = (
        '<table><thead><tr>'
        '<th>sense_id</th><th>pt</th><th>pos</th><th>en_primary</th>'
        '<th>bp_status</th><th>bp_validity</th><th>register</th>'
        '<th>risk_flags</th><th>risk_note</th><th>conf</th>'
        '<th>src</th><th>reason / example</th>'
        '</tr></thead><tbody>'
    )
    body = "\n".join(_row_html(r) for r in rows)
    return f'{head}{body}</tbody></table>'


def _distribution_table(
    counts: Counter, allowed: tuple[str, ...], total: int, name: str
) -> str:
    max_count = max(counts.values()) if counts else 1
    rows: list[str] = []
    used_values = set(counts.keys())
    # Show allowed values first (in order), then any unexpected ones.
    for v in allowed:
        n = counts.get(v, 0)
        pct = n / total * 100 if total else 0
        rows.append(
            f'<tr><td><code>{_esc(v)}</code></td>'
            f'<td style="text-align:right">{n}</td>'
            f'<td style="text-align:right">{pct:.1f}%</td>'
            f'<td>{_bar_html(n, max_count)}</td></tr>'
        )
    extras = used_values - set(allowed)
    for v in sorted(extras):
        n = counts.get(v, 0)
        pct = n / total * 100 if total else 0
        rows.append(
            f'<tr class="flagged"><td><code>{_esc(v)}</code> ⚠ UNKNOWN</td>'
            f'<td style="text-align:right">{n}</td>'
            f'<td style="text-align:right">{pct:.1f}%</td>'
            f'<td>{_bar_html(n, max_count)}</td></tr>'
        )
    return (
        f'<h2>{_esc(name)} distribution</h2>'
        '<table><thead><tr><th>value</th><th>count</th><th>%</th>'
        '<th></th></tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table>'
    )


def _risk_flag_distribution(rows: list[dict], total: int) -> str:
    """Split pipe-joined risk_flags into per-flag counts."""
    flag_counts: Counter = Counter()
    for r in rows:
        raw = (r.get("risk_flags") or "").strip()
        if not raw:
            continue
        for f in raw.split("|"):
            f = f.strip()
            if f:
                flag_counts[f] += 1
    return _distribution_table(flag_counts, ALLOWED_RISK_FLAGS, total, "risk_flags")


def _build_report(rows: list[dict]) -> str:
    total = len(rows)
    if total == 0:
        return "<html><body><h1>Pilot v0 report</h1><p>No rows.</p></body></html>"

    by_validity = Counter((r.get("bp_validity") or "").strip() for r in rows)
    by_register = Counter((r.get("register") or "").strip() for r in rows)
    by_conf = Counter((r.get("confidence") or "").strip().lower() for r in rows)
    by_source = Counter((r.get("source") or "").strip() for r in rows)

    # KPI line.
    low_n = by_conf.get("low", 0)
    medium_n = by_conf.get("medium", 0)
    low_pct = low_n / total if total else 0
    uncertain_v = by_validity.get("uncertain", 0)
    uncertain_r = by_register.get("uncertain", 0)

    kpi_total = f'<span class="kpi">total: {total}</span>'
    kpi_manual = f'<span class="kpi">manual: {by_source.get("manual", 0)}</span>'
    kpi_llm = f'<span class="kpi">llm: {by_source.get("llm", 0)}</span>'
    kpi_audit = (
        f'<span class="kpi">audit_only: {by_source.get("llm_audit_only", 0)}</span>'
    )
    kpi_def = (
        f'<span class="kpi">deterministic_default: '
        f'{by_source.get("deterministic_default", 0)}</span>'
    )
    kpi_low = (
        f'<span class="kpi {"fail" if low_pct > LOW_CONF_CAP else "ok"}">'
        f'low_conf: {low_n} ({low_pct:.1%})</span>'
    )
    kpi_med = f'<span class="kpi">medium_conf: {medium_n} ({medium_n/total:.1%}) — informational</span>'
    kpi_uncv = (
        f'<span class="kpi {"fail" if uncertain_v / total > UNCERTAIN_VALIDITY_CAP else "ok"}">'
        f'uncertain_validity: {uncertain_v}</span>'
    )
    kpi_uncr = (
        f'<span class="kpi {"fail" if uncertain_r / total > UNCERTAIN_REGISTER_CAP else "ok"}">'
        f'uncertain_register: {uncertain_r}</span>'
    )

    # Section: Stage-12 carry-forward.
    carry_rows = [r for r in rows
                  if (r.get("old_risk_note") or "").strip()]

    # Section: bp_status=nsfw rows.
    nsfw_rows = [r for r in rows if r.get("bp_status") == "nsfw"]
    nsfw_rows.sort(key=lambda r: r["sense_id"])

    # Section: bp_status=false_friend rows.
    ff_rows = [r for r in rows if r.get("bp_status") == "false_friend"]
    ff_rows.sort(key=lambda r: r["sense_id"])

    # Section: manual override prefill rows.
    manual_rows = [r for r in rows if r["sense_id"] in MANUAL_PREFILL_SIDS]
    manual_rows.sort(key=lambda r: r["sense_id"])

    # Section: audit-only random normals.
    audit_rows = [r for r in rows if r.get("source") == "llm_audit_only"]
    audit_rows.sort(key=lambda r: r["sense_id"])
    # Audit acceptance: ≥ 95 % land at standard/neutral/none/""
    audit_normal = sum(
        1 for r in audit_rows
        if r.get("bp_validity") == "standard"
        and r.get("register") == "neutral"
        and r.get("risk_flags") == "none"
        and not (r.get("risk_note") or "").strip()
    )
    audit_pct = audit_normal / len(audit_rows) if audit_rows else 0
    audit_pass = audit_pct >= 0.95

    # Section: low-confidence rows.
    low_rows = [r for r in rows
                if (r.get("confidence") or "").strip().lower() == "low"]
    low_rows.sort(key=lambda r: r["sense_id"])

    # Section: rows where risk_note non-empty but risk_flags == none (H8 violation).
    h8_violations = [
        r for r in rows
        if (r.get("risk_note") or "").strip()
        and (r.get("risk_flags") or "").strip() == "none"
    ]
    h8_violations.sort(key=lambda r: r["sense_id"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stage 15 — Risk / Register Pilot Review</title>
<style>{CSS}</style>
</head>
<body>
<h1>Stage 15 — Risk / Register Pilot Review</h1>

<div>
{kpi_total} {kpi_manual} {kpi_llm} {kpi_audit} {kpi_def}
{kpi_low} {kpi_med} {kpi_uncv} {kpi_uncr}
</div>

<p>
  <strong>Audit-only random-normals subset</strong>:
  {len(audit_rows)} rows; {audit_normal} ({audit_pct:.1%}) returned
  <code>standard / neutral / none / ""</code>
  ({"✓ ≥ 95% pass" if audit_pass else "⚠ FAIL — investigate"}).
</p>

<p>
  <strong>H8 violations</strong> (non-empty risk_note + risk_flags=none):
  {len(h8_violations)} {"✓" if not h8_violations else "⚠ MUST FIX"}.
</p>

<nav>
<a href="#bp-validity">bp_validity</a>
<a href="#register">register</a>
<a href="#risk-flags">risk_flags</a>
<a href="#carry">Stage-12 carry</a>
<a href="#nsfw">bp_status=nsfw</a>
<a href="#ff">bp_status=false_friend</a>
<a href="#manual">manual prefill</a>
<a href="#audit">audit-only normals</a>
<a href="#low">low_conf rows</a>
<a href="#h8">H8 violations</a>
</nav>

<section id="bp-validity">
{_distribution_table(by_validity, ALLOWED_BP_VALIDITY, total, "bp_validity")}
</section>

<section id="register">
{_distribution_table(by_register, ALLOWED_REGISTER, total, "register")}
</section>

<section id="risk-flags">
{_risk_flag_distribution(rows, total)}
</section>

<section id="carry">
<h2>Stage-12 risk_note carry-forward ({len(carry_rows)} rows)</h2>
<p>Every row must ship a non-empty risk_note and risk_flags != none (verifier H9 + H10).</p>
{_table(carry_rows)}
</section>

<section id="nsfw">
<h2>bp_status=nsfw ({len(nsfw_rows)} rows)</h2>
<p>Expect: register ∈ {{vulgar, taboo, slang, archaic}} AND risk_flags includes one of sexual / vulgar / offensive / slur / racial_sensitive.</p>
{_table(nsfw_rows)}
</section>

<section id="ff">
<h2>bp_status=false_friend ({len(ff_rows)} rows)</h2>
<p>Expect: risk_flags includes <code>false_friend</code>.</p>
{_table(ff_rows)}
</section>

<section id="manual">
<h2>Manual-override prefill ({len(manual_rows)} rows)</h2>
<p>Must ship the manual values exactly — never LLM-overwritten.</p>
{_table(manual_rows)}
</section>

<section id="audit">
<h2>Audit-only random-normal subset ({len(audit_rows)} rows)</h2>
<p>These rows were classified by the LLM but won't change the production candidate set. Expect ≥ 95% to be standard/neutral/none/""; the rest indicate filter blind spots.</p>
{_table(audit_rows)}
</section>

<section id="low">
<h2>Low-confidence rows ({len(low_rows)} rows)</h2>
{_table(low_rows)}
</section>

<section id="h8">
<h2>H8 violations: risk_note non-empty but risk_flags=none ({len(h8_violations)})</h2>
<p>Verifier hard check — must be zero before full run.</p>
{_table(h8_violations)}
</section>

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
        print("Run build/15_0_risk_register_classifier.py --pilot first.",
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
