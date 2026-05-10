"""Stage 8 / Step 2 — Stratified calibration HTML.

Generates `audit/08_2_calibration.html` with ~100 stratified clips for
human listening, plus an embedded reviewer guide. User listens, marks
each clip OK/MISPRONOUNCED/UNCLEAR, exports labels (TSV via JS clipboard).

After listening, run `--apply-calibration` to:
  - read `data/_audio_calibration_labels.tsv` (user-pasted)
  - apply the auto-confirm rule (judge + priority + label)
  - emit `data/_audio_mispronunciation_confirmed.tsv` (input to 08_3)
  - emit `data/_audio_calibration_unclear.tsv` (manual queue)

The HTML embeds `docs/reviewer_guide.md` rendered inline (custom mini-
markdown converter to avoid an external dep).

Usage:
    # Build HTML
    .venv/bin/python build/08_2_calibration.py
    open audit/08_2_calibration.html

    # After listening + pasting labels:
    .venv/bin/python build/08_2_calibration.py --apply-calibration
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
DOCS_DIR = REPO_ROOT / "docs"

RISK_PATH = DATA_DIR / "_audio_risk_classification.tsv"
MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
VERDICTS_PATH = DATA_DIR / "_audio_judge_verdicts.tsv"
USER_REPORTED_PATH = DATA_DIR / "_audio_user_reported_failures.tsv"
LABELS_PATH = DATA_DIR / "_audio_calibration_labels.tsv"
CONFIRMED_PATH = DATA_DIR / "_audio_mispronunciation_confirmed.tsv"
UNCLEAR_PATH = DATA_DIR / "_audio_calibration_unclear.tsv"
HTML_PATH = AUDIT_DIR / "08_2_calibration.html"
REVIEWER_GUIDE_PATH = DOCS_DIR / "reviewer_guide.md"

CALIBRATION_SEED = 42  # deterministic sample selection


# ---------------------------------------------------------------------------
# Mini markdown → HTML (handles the subset used in reviewer_guide.md)
# ---------------------------------------------------------------------------
def _md_inline(s: str) -> str:
    """Inline markdown: **bold**, *italic*, `code`."""
    s = html.escape(s)
    s = re.sub(r"\*\*([^*]+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+?)`", r"<code>\1</code>", s)
    return s


def _md_table(lines: list[str]) -> str:
    """Convert a contiguous block of `|...|` lines to an HTML table."""
    if len(lines) < 2:
        return ""
    rows: list[list[str]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        # Skip alignment row (|---|---|)
        if re.match(r"^\|[\s:|-]+\|$", ln):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    out = ['<table class="md-table">']
    out.append("<thead><tr>" + "".join(f"<th>{_md_inline(c)}</th>" for c in rows[0]) + "</tr></thead>")
    out.append("<tbody>")
    for r in rows[1:]:
        out.append("<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def md_to_html(md: str) -> str:
    """Minimal markdown → HTML for the reviewer guide."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        # Tables: collect contiguous | lines
        if ln.strip().startswith("|"):
            tbl_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            out.append(_md_table(tbl_lines))
            continue
        # Headings
        m = re.match(r"^(#{1,6})\s+(.+)$", ln)
        if m:
            level = len(m.group(1))
            text = _md_inline(m.group(2))
            out.append(f"<h{level}>{text}</h{level}>")
            i += 1
            continue
        # List items (collect contiguous - or *)
        if re.match(r"^\s*[-*]\s+", ln):
            list_items: list[str] = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                m = re.match(r"^\s*[-*]\s+(.+)$", lines[i])
                if m:
                    list_items.append(_md_inline(m.group(1)))
                i += 1
            out.append("<ul>" + "".join(f"<li>{li}</li>" for li in list_items) + "</ul>")
            continue
        # Numbered list
        if re.match(r"^\s*\d+\.\s+", ln):
            ol_items: list[str] = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                m = re.match(r"^\s*\d+\.\s+(.+)$", lines[i])
                if m:
                    ol_items.append(_md_inline(m.group(1)))
                i += 1
            out.append("<ol>" + "".join(f"<li>{li}</li>" for li in ol_items) + "</ol>")
            continue
        # Blank line → paragraph break
        if not ln.strip():
            i += 1
            continue
        # Plain paragraph: collect contiguous non-blank, non-special lines
        para: list[str] = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|\s*[-*]\s|\s*\d+\.\s|\|)", lines[i]):
            para.append(lines[i])
            i += 1
        if para:
            text = _md_inline(" ".join(para))
            out.append(f"<p>{text}</p>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Calibration sample builder
# ---------------------------------------------------------------------------
def _build_sample(rng: random.Random) -> list[dict]:
    """Stratified ~100 clips across risk families."""
    risk_rows = read_tsv(RISK_PATH)
    manifest_rows = read_tsv(MANIFEST_PATH)
    verdicts = read_tsv(VERDICTS_PATH) if VERDICTS_PATH.exists() else []

    # Index manifest word clips
    manifest_word = {
        r["sense_id"]: r for r in manifest_rows
        if r.get("clip_type") == "word" and r.get("status") == "uploaded"
    }
    verdict_by_sid = {v["sense_id"]: v for v in verdicts}
    risk_by_sid = {r["sense_id"]: r for r in risk_rows}

    # Resolve item shape
    def make_item(sid: str) -> dict | None:
        m = manifest_word.get(sid)
        if not m:
            return None
        r = risk_by_sid.get(sid, {})
        v = verdict_by_sid.get(sid, {})
        return {
            "sense_id": sid,
            "pt": m.get("text_input") or r.get("pt", ""),
            "voice_id": m.get("voice_id", ""),
            "url": m.get("url", ""),
            "risk_patterns": r.get("risk_patterns", "none"),
            "priority": r.get("priority", "—"),
            "judge_verdict": v.get("pronunciation_verdict", ""),
            "judge_drift": v.get("drift", ""),
            "judge_severity": v.get("severity", ""),
            "judge_confidence": v.get("confidence", ""),
            "judge_evidence": v.get("evidence", ""),
        }

    def by_pattern(pat: str) -> list[dict]:
        return [
            i for r in risk_rows
            if pat in r.get("risk_patterns", "")
            for i in [make_item(r["sense_id"])]
            if i is not None
        ]

    def random_pick(items: list[dict], n: int) -> list[dict]:
        if len(items) <= n:
            return items
        return rng.sample(items, n)

    sample: list[dict] = []
    seen: set[str] = set()

    def add(items: list[dict]) -> None:
        for i in items:
            if i["sense_id"] not in seen:
                sample.append(i)
                seen.add(i["sense_id"])

    # Bucket 1: ALL user-reported failures
    user_reported_items = [i for i in by_pattern("user_reported")]
    add(user_reported_items)

    # Bucket 2: voice_high_risk_critical non-bp (currently empty post-swap)
    crit_nonbp = [
        i for i in by_pattern("voice_high_risk_critical")
        if i["judge_verdict"] == "non_bp"
    ]
    add(random_pick(crit_nonbp, 10))

    # Bucket 3: voice_high_risk_elevated non-bp (currently empty post-swap)
    elev_nonbp = [
        i for i in by_pattern("voice_high_risk_elevated")
        if i["judge_verdict"] == "non_bp"
    ]
    add(random_pick(elev_nonbp, 5))

    # Bucket 4: final_l_vocalization (20)
    add(random_pick(by_pattern("final_l_vocalization"), 20))

    # Bucket 5: de_te_palatalization (15)
    add(random_pick(by_pattern("de_te_palatalization"), 15))

    # Bucket 6: initial_r_or_rr (10)
    add(random_pick(by_pattern("initial_r_or_rr"), 10))

    # Bucket 7: coda_s_ep_risk + final_unstressed_e (10 combined)
    coda_s = by_pattern("coda_s_ep_risk")
    final_e = by_pattern("final_unstressed_e")
    combined = list({i["sense_id"]: i for i in coda_s + final_e}.values())
    add(random_pick(combined, 10))

    # Bucket 8: english_loanword + spanish_collision + french_loanword (10 combined)
    en = by_pattern("english_loanword")
    es = by_pattern("spanish_collision")
    fr = by_pattern("french_loanword")
    combined = list({i["sense_id"]: i for i in en + es + fr}.values())
    add(random_pick(combined, 10))

    # Bucket 9: audio-judge bp_ok controls (10)
    bp_ok = [
        make_item(v["sense_id"])
        for v in verdicts
        if v.get("pronunciation_verdict") == "bp_ok"
    ]
    bp_ok = [i for i in bp_ok if i is not None and i["sense_id"] not in seen]
    add(random_pick(bp_ok, 10))

    # Bucket 10: voice-coverage minimum (≥1 clip per voice in pool)
    voices_in_sample = {i["voice_id"] for i in sample}
    voices_in_manifest = {m["voice_id"] for m in manifest_word.values() if m.get("voice_id")}
    missing_voices = voices_in_manifest - voices_in_sample
    for v in missing_voices:
        candidates = [
            make_item(sid) for sid, m in manifest_word.items()
            if m.get("voice_id") == v and sid not in seen
        ]
        candidates = [c for c in candidates if c is not None]
        if candidates:
            add([rng.choice(candidates)])

    return sample


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 8 — Calibration listening checklist</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 920px; margin: 2em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1, h2, h3, h4 {{ color: #111; margin-top: 1.6em; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
.guide {{ background: #f5f8fb; border: 1px solid #cdd; border-radius: 6px;
  padding: 1em 1.5em; margin-bottom: 2em; font-size: 0.95em; }}
.md-table {{ border-collapse: collapse; margin: 0.6em 0; font-size: 0.92em; }}
.md-table th, .md-table td {{ border: 1px solid #aaa; padding: 0.3em 0.7em; text-align: left; vertical-align: top; }}
.md-table th {{ background: #e6ecf2; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
.clip {{ border-left: 4px solid #ddd; padding: 0.6em 1em; margin: 1em 0; background: #fafafa; }}
.clip.priority-P0 {{ border-left-color: #d00; }}
.clip.priority-P1 {{ border-left-color: #f80; }}
.clip.priority-P2 {{ border-left-color: #aa0; }}
.clip.priority-P3 {{ border-left-color: #088; }}
.clip header {{ font-weight: bold; font-size: 1.1em; }}
.meta {{ font-size: 0.85em; color: #555; margin: 0.3em 0; }}
.meta code {{ font-size: 0.85em; }}
.judge {{ font-style: italic; margin: 0.4em 0; color: #344; }}
audio {{ width: 100%; margin: 0.4em 0; }}
.radio-row {{ display: flex; gap: 1em; margin-top: 0.4em; }}
.radio-row label {{ cursor: pointer; }}
.export {{ position: sticky; top: 0; background: #ffd; padding: 1em; border: 2px solid #ba0;
  border-radius: 4px; margin: 1em 0; z-index: 10; }}
button {{ font-size: 1em; padding: 0.5em 1em; cursor: pointer; }}
#export-output {{ width: 100%; height: 200px; font-family: ui-monospace, monospace; }}
</style>
</head>
<body>

<h1>Stage 8 — Calibration listening checklist</h1>

<p>{n_clips} clips queued. Listen, mark each <strong>OK</strong>,
<strong>MISPRONOUNCED</strong>, or <strong>UNCLEAR</strong>. When done,
click <em>Export labels</em> at the top to copy the TSV to your clipboard.
Then paste into <code>data/_audio_calibration_labels.tsv</code> and run
<code>build/08_2_calibration.py --apply-calibration</code>.</p>

<div class="export">
  <button onclick="exportLabels()">Export labels (copy TSV to clipboard)</button>
  <span id="export-status" style="margin-left: 1em; color: #060;"></span>
  <details style="margin-top: 0.6em;">
    <summary>Show raw TSV</summary>
    <textarea id="export-output" readonly></textarea>
  </details>
</div>

<div class="guide">
{reviewer_guide_html}
</div>

<h2>Clips ({n_clips})</h2>
{clip_html}

<script>
function exportLabels() {{
  const rows = ["sense_id\\tpt\\tvoice_id\\tlabel"];
  document.querySelectorAll('.clip').forEach(c => {{
    const sid = c.dataset.senseId;
    const pt = c.dataset.pt;
    const vid = c.dataset.voiceId;
    const sel = c.querySelector('input[type=radio]:checked');
    const label = sel ? sel.value : "UNANSWERED";
    rows.push([sid, pt, vid, label].join('\\t'));
  }});
  const tsv = rows.join('\\n');
  document.getElementById('export-output').value = tsv;
  navigator.clipboard.writeText(tsv).then(() => {{
    document.getElementById('export-status').textContent = '✓ Copied to clipboard.';
  }}, () => {{
    document.getElementById('export-status').textContent = 'Copy failed — copy manually from the textarea below.';
  }});
}}
</script>

</body>
</html>
"""


def _clip_block(idx: int, item: dict) -> str:
    sid = html.escape(item["sense_id"])
    pt = html.escape(item["pt"])
    vid = html.escape(item["voice_id"])
    url = html.escape(item["url"])
    priority = html.escape(item["priority"] or "—")
    patterns = html.escape(item["risk_patterns"] or "none")
    judge = ""
    if item["judge_verdict"]:
        judge = (
            f"<div class=\"judge\">audio judge: <strong>{html.escape(item['judge_verdict'])}</strong>"
            f" (drift={html.escape(item['judge_drift'])}, sev={html.escape(item['judge_severity'])},"
            f" conf={html.escape(item['judge_confidence'])}) — {html.escape(item['judge_evidence'])}</div>"
        )
    radio_name = f"label_{idx}"
    return f"""
<section class="clip priority-{priority}" data-sense-id="{sid}" data-pt="{pt}" data-voice-id="{vid}">
  <header>{idx + 1}. <code>{sid}</code> — <em>{pt}</em></header>
  <div class="meta">priority: <strong>{priority}</strong> · patterns: <code>{patterns}</code> · voice: <code>{vid}</code></div>
  {judge}
  <audio controls preload="none" src="{url}"></audio>
  <div class="radio-row">
    <label><input type="radio" name="{radio_name}" value="OK"> OK (sounds clearly BP)</label>
    <label><input type="radio" name="{radio_name}" value="MISPRONOUNCED"> MISPRONOUNCED</label>
    <label><input type="radio" name="{radio_name}" value="UNCLEAR"> UNCLEAR</label>
  </div>
</section>
"""


def build_html() -> Path:
    rng = random.Random(CALIBRATION_SEED)
    sample = _build_sample(rng)

    # Sort by priority then sense_id for stable display
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "—": 4}
    sample.sort(key=lambda i: (priority_order.get(i["priority"], 5), i["sense_id"]))

    if REVIEWER_GUIDE_PATH.exists():
        guide_md = REVIEWER_GUIDE_PATH.read_text(encoding="utf-8")
        guide_html = md_to_html(guide_md)
    else:
        guide_html = "<p><em>(reviewer guide not found)</em></p>"

    clip_blocks = "\n".join(_clip_block(i, item) for i, item in enumerate(sample))
    html_doc = HTML_TEMPLATE.format(
        n_clips=len(sample),
        reviewer_guide_html=guide_html,
        clip_html=clip_blocks,
    )

    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(html_doc, encoding="utf-8")

    # Print stratification summary
    by_prio = defaultdict(int)
    for item in sample:
        by_prio[item["priority"]] += 1
    print(f"Wrote {len(sample)} clips to {HTML_PATH}")
    for prio in ("P0", "P1", "P2", "P3", "—"):
        print(f"  {prio:<3}  {by_prio[prio]}")
    return HTML_PATH


# ---------------------------------------------------------------------------
# --apply-calibration: read labels, derive confirmed + unclear sets
# ---------------------------------------------------------------------------
def apply_calibration() -> int:
    if not LABELS_PATH.exists():
        print(
            f"ERROR: labels file not found at {LABELS_PATH}. "
            "Paste the exported TSV from the HTML form into that path first.",
            file=sys.stderr,
        )
        return 1
    labels = read_tsv(LABELS_PATH)
    risk_rows = read_tsv(RISK_PATH)
    verdicts = read_tsv(VERDICTS_PATH) if VERDICTS_PATH.exists() else []

    risk_by_sid = {r["sense_id"]: r for r in risk_rows}
    verdict_by_sid = {v["sense_id"]: v for v in verdicts}

    confirmed: list[dict] = []
    unclear: list[dict] = []

    for row in labels:
        sid = row.get("sense_id", "").strip()
        if not sid:
            continue
        label = row.get("label", "").strip()
        v = verdict_by_sid.get(sid, {})
        r = risk_by_sid.get(sid, {})
        priority = r.get("priority", "—")
        verdict = v.get("pronunciation_verdict", "")
        confidence = v.get("confidence", "")

        # Auto-confirm rule
        confirmed_for_alias = False
        if label == "MISPRONOUNCED":
            confirmed_for_alias = True
        elif verdict == "non_bp" and confidence == "high":
            confirmed_for_alias = True
        elif verdict == "non_bp" and confidence == "medium" and priority in ("P0", "P1"):
            confirmed_for_alias = True

        out_row = {
            "sense_id": sid,
            "pt": row.get("pt", "") or r.get("pt", ""),
            "voice_id": row.get("voice_id", "") or "",
            "user_label": label,
            "judge_verdict": verdict,
            "judge_confidence": confidence,
            "priority": priority,
            "risk_patterns": r.get("risk_patterns", "none"),
        }
        if label == "UNCLEAR":
            unclear.append(out_row)
        elif confirmed_for_alias:
            confirmed.append(out_row)

    fns = ["sense_id", "pt", "voice_id", "user_label", "judge_verdict", "judge_confidence", "priority", "risk_patterns"]
    write_tsv(CONFIRMED_PATH, confirmed, fieldnames=fns)
    write_tsv(UNCLEAR_PATH, unclear, fieldnames=fns)

    # Per-family MISPRONOUNCED rate (for adaptive expansion gate)
    family_counts = defaultdict(lambda: {"total": 0, "mispronounced": 0})
    for row in labels:
        sid = row.get("sense_id", "")
        label = row.get("label", "")
        r = risk_by_sid.get(sid, {})
        patterns = r.get("risk_patterns", "").split(",")
        for pat in patterns:
            pat = pat.strip()
            if pat:
                family_counts[pat]["total"] += 1
                if label == "MISPRONOUNCED":
                    family_counts[pat]["mispronounced"] += 1

    high_failure_families: list[str] = []
    print()
    print("Per-family MISPRONOUNCED rate:")
    for fam, d in sorted(family_counts.items(), key=lambda x: -x[1]["mispronounced"]):
        if d["total"] == 0:
            continue
        rate = 100 * d["mispronounced"] / d["total"]
        flag = " *EXPAND*" if rate > 10 and d["total"] >= 3 else ""
        print(f"  {fam:<30} {d['mispronounced']}/{d['total']} ({rate:.0f}%){flag}")
        if rate > 10 and d["total"] >= 3:
            high_failure_families.append(fam)

    print()
    print(f"  confirmed_for_alias:  {len(confirmed)}  → {CONFIRMED_PATH}")
    print(f"  unclear:              {len(unclear)}  → {UNCLEAR_PATH}")
    if high_failure_families:
        print()
        print(f"⚠ Families with >10% MISPRONOUNCED (consider re-running with expanded sample):")
        for f in high_failure_families:
            print(f"  - {f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply-calibration", action="store_true",
                        help="Read labels TSV, derive confirmed + unclear sets.")
    args = parser.parse_args()

    if args.apply_calibration:
        return apply_calibration()

    build_html()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
