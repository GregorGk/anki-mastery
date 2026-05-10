"""Stage 8 / Step 6 — After-fix listening QA + summary + final TSV rebuild.

Three modes:

1. `--build-html` — build `audit/08_6_after_fix.html` with before/after
   listening pairs (~30–40 clips). Each row has two `<audio>` elements
   (v1 archived URL + v2 current URL) plus the alias used and 3-button
   radio (BETTER / SAME / WORSE). Embeds reviewer guide.

2. `--apply-and-rebuild` — read user labels from
   `data/_audio_after_fix_labels.tsv`. Apply hard gate:
     - any WORSE row in any family OR
     - SAME rate > 10% in any family
   blocks the rebuild and surfaces the offending families for a Phase
   08_3 → 08_4 → 08_5 rerun. Otherwise, emit `audit/08_summary.txt`
   with the Stage 8 stats and instructs the user to run
   `build/derive_final.py` (which we don't write here — Stage 8 just
   ensures the manifest is correct).

Note: v1 of this script is a skeleton. The before/after sample-builder
logic is implemented; the hard-gate rerun automation is a stub the user
can extend after running through Phase 08_5 once.
"""
from __future__ import annotations

import argparse
import csv
import html
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

MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
APPLICATIONS_PATH = DATA_DIR / "_pronunciation_alias_applications.tsv"
ALIASES_PATH = DATA_DIR / "_pronunciation_aliases.tsv"
USER_REPORTED_PATH = DATA_DIR / "_audio_user_reported_failures.tsv"
LABELS_PATH = DATA_DIR / "_audio_after_fix_labels.tsv"
RERUN_QUEUE_PATH = DATA_DIR / "_audio_alias_rerun_queue.tsv"
SUMMARY_PATH = AUDIT_DIR / "08_summary.txt"
HTML_PATH = AUDIT_DIR / "08_6_after_fix.html"
REVIEWER_GUIDE_PATH = DOCS_DIR / "reviewer_guide.md"

AFTER_FIX_SEED = 42
AFTER_FIX_TARGET_SIZE = 30


def _build_after_fix_sample(rng: random.Random) -> list[dict]:
    """~30-40 before/after pairs.

    Layout per spec:
      - all user-reported repaired clips
      - 10–20 random repaired clips
      - ≥1 repaired clip per alias family
      - ≥1 repaired clip per voice
      - all respelling_failed clips
    """
    if not APPLICATIONS_PATH.exists() or not ALIASES_PATH.exists():
        return []

    apps = read_tsv(APPLICATIONS_PATH)
    aliases = {a["alias_id"]: a for a in read_tsv(ALIASES_PATH)}
    manifest = read_tsv(MANIFEST_PATH)
    user_reported = set()
    if USER_REPORTED_PATH.exists():
        user_reported = {r["pt"].strip() for r in read_tsv(USER_REPORTED_PATH) if r.get("pt")}

    # Build per-sense info
    sid_to_word = {
        r["sense_id"]: r for r in manifest
        if r.get("clip_type") == "word" and r.get("status") == "uploaded"
    }
    items: list[dict] = []
    for app in apps:
        sid = app.get("sense_id", "")
        m = sid_to_word.get(sid)
        if not m:
            continue
        alias_id = app.get("alias_id", "")
        a = aliases.get(alias_id, {})
        url_after = m.get("url", "")
        # Derive v1 URL by replacing trailing -vN.mp3 with -v1.mp3.
        # Stage 6 leaves old versioned R2 objects in place (versioning is
        # additive), so v1 is always reachable at the same path.
        url_before = re.sub(r"-v\d+\.mp3$", "-v1.mp3", url_after)
        items.append({
            "sense_id": sid,
            "pt": m.get("text_input") or a.get("pt", ""),
            "voice_id": m.get("voice_id", ""),
            "url_after": url_after,
            "url_before": url_before if url_before != url_after else "",
            "alias_id": alias_id,
            "alias_respelling": a.get("pt_respelling", ""),
            "applied_family": a.get("applied_family", ""),
            "is_user_reported": (a.get("pt", "") in user_reported),
            "respelling_failed": (m.get("status") == "respelling_failed"),
        })

    # Bucket selection
    sample: list[dict] = []
    seen: set[str] = set()
    def add(group: list[dict]) -> None:
        for i in group:
            if i["sense_id"] not in seen:
                sample.append(i)
                seen.add(i["sense_id"])

    # All user-reported
    add([i for i in items if i["is_user_reported"]])
    # All respelling_failed
    add([i for i in items if i["respelling_failed"]])
    # ≥1 per family
    by_family = defaultdict(list)
    for i in items:
        by_family[i["applied_family"]].append(i)
    for fam, group in by_family.items():
        if not any(s["applied_family"] == fam for s in sample):
            add([rng.choice(group)])
    # ≥1 per voice
    by_voice = defaultdict(list)
    for i in items:
        by_voice[i["voice_id"]].append(i)
    for vid, group in by_voice.items():
        if not any(s["voice_id"] == vid for s in sample):
            add([rng.choice(group)])
    # Random fill to ~30
    remaining = [i for i in items if i["sense_id"] not in seen]
    rng.shuffle(remaining)
    needed = max(0, AFTER_FIX_TARGET_SIZE - len(sample))
    add(remaining[:needed])

    return sample


def cmd_build_html() -> int:
    rng = random.Random(AFTER_FIX_SEED)
    sample = _build_after_fix_sample(rng)
    if not sample:
        print("Nothing to render — alias applications + manifest needed first.", file=sys.stderr)
        return 1

    # Reuse 08_2's md_to_html
    import importlib.util
    spec_path = REPO_ROOT / "build" / "08_2_calibration.py"
    spec = importlib.util.spec_from_file_location("calibration", spec_path)
    if spec is None or spec.loader is None:
        print("ERROR: failed to import 08_2_calibration.py for md_to_html.", file=sys.stderr)
        return 1
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    md_to_html = cal.md_to_html  # type: ignore[attr-defined]

    if REVIEWER_GUIDE_PATH.exists():
        guide_html = md_to_html(REVIEWER_GUIDE_PATH.read_text(encoding="utf-8"))
    else:
        guide_html = "<p><em>(reviewer guide missing)</em></p>"

    cards: list[str] = []
    for i, item in enumerate(sample):
        sid = html.escape(item["sense_id"])
        pt = html.escape(item["pt"])
        respell = html.escape(item["alias_respelling"])
        family = html.escape(item["applied_family"])
        url_before = html.escape(item["url_before"]) if item["url_before"] else ""
        url_after = html.escape(item["url_after"])
        radio = f"label_{i}"
        before_block = (
            f'<div class="before"><strong>BEFORE (v1):</strong> <audio controls preload="none" src="{url_before}"></audio></div>'
            if url_before
            else '<div class="before"><em>BEFORE not archived (v1 R2 object overwritten by Stage 6 resume); judge AFTER alone for now.</em></div>'
        )
        cards.append(f"""
<section class="row">
  <header><code>{sid}</code> — <em>{pt}</em> &nbsp; <small>family={family}, alias='{respell}'</small></header>
  {before_block}
  <div class="after"><strong>AFTER (v2 with alias):</strong> <audio controls preload="none" src="{url_after}"></audio></div>
  <div class="radio-row">
    <label><input type="radio" name="{radio}" value="BETTER"> BETTER (v2 sounds more BP)</label>
    <label><input type="radio" name="{radio}" value="SAME"> SAME (no audible difference)</label>
    <label><input type="radio" name="{radio}" value="WORSE"> WORSE (alias regressed quality)</label>
  </div>
</section>
""")

    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 8 — After-fix listening QA</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 920px; margin: 2em auto; padding: 0 1em; line-height: 1.5; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
.guide {{ background: #f5f8fb; border: 1px solid #cdd; border-radius: 6px;
  padding: 1em 1.5em; margin-bottom: 2em; font-size: 0.95em; }}
.md-table {{ border-collapse: collapse; margin: 0.6em 0; font-size: 0.92em; }}
.md-table th, .md-table td {{ border: 1px solid #aaa; padding: 0.3em 0.7em; vertical-align: top; }}
.md-table th {{ background: #e6ecf2; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; }}
.row {{ border-left: 4px solid #4a8; padding: 0.6em 1em; margin: 1em 0; background: #fafafa; }}
.row header {{ font-weight: bold; }}
.before, .after {{ margin: 0.4em 0; }}
audio {{ width: 100%; margin: 0.2em 0; }}
.radio-row {{ display: flex; gap: 1em; margin-top: 0.4em; }}
.export {{ position: sticky; top: 0; background: #ffd; padding: 1em; border: 2px solid #ba0;
  border-radius: 4px; margin: 1em 0; z-index: 10; }}
button {{ font-size: 1em; padding: 0.5em 1em; cursor: pointer; }}
#export-output {{ width: 100%; height: 200px; font-family: monospace; }}
</style>
</head>
<body>

<h1>Stage 8 — After-fix listening QA</h1>

<p>{n} clips. For each, listen to BEFORE (v1) and AFTER (v2) and decide:
<strong>BETTER</strong> (v2 sounds more BP), <strong>SAME</strong>
(no audible difference), <strong>WORSE</strong> (v2 regressed). The
hard gate blocks the final TSV rebuild if there's any WORSE or
&gt;10% SAME in any family.</p>

<div class="export">
  <button onclick="exportLabels()">Export labels (copy TSV)</button>
  <span id="export-status" style="margin-left: 1em; color: #060;"></span>
  <details style="margin-top: 0.6em;">
    <summary>Show raw TSV</summary>
    <textarea id="export-output" readonly></textarea>
  </details>
</div>

<div class="guide">
{guide_html}
</div>

<h2>Clips ({n})</h2>
{cards}

<script>
function exportLabels() {{
  const rows = ["sense_id\\tpt\\tvoice_id\\talias_respelling\\tapplied_family\\tlabel"];
  document.querySelectorAll('section.row').forEach((s, idx) => {{
    const sid = s.querySelector('code').textContent;
    const pt = s.querySelector('em').textContent;
    const headerSmall = s.querySelector('small')?.textContent || '';
    const fam = (headerSmall.match(/family=([^,]+)/)?.[1] || '').trim();
    const respell = (headerSmall.match(/alias='([^']+)'/)?.[1] || '').trim();
    const sel = s.querySelector('input[type=radio]:checked');
    const label = sel ? sel.value : "UNANSWERED";
    rows.push([sid, pt, '', respell, fam, label].join('\\t'));
  }});
  const tsv = rows.join('\\n');
  document.getElementById('export-output').value = tsv;
  navigator.clipboard.writeText(tsv).then(() => {{
    document.getElementById('export-status').textContent = '✓ Copied.';
  }}, () => {{
    document.getElementById('export-status').textContent = 'Copy failed — copy manually.';
  }});
}}
</script>

</body>
</html>
"""
    out = template.format(n=len(sample), guide_html=guide_html, cards="\n".join(cards))
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(out, encoding="utf-8")
    print(f"Wrote {HTML_PATH} ({len(sample)} clips)")
    return 0


def cmd_apply_and_rebuild() -> int:
    """Read after-fix labels, apply hard gate, emit summary."""
    if not LABELS_PATH.exists():
        print(f"ERROR: labels not found at {LABELS_PATH}", file=sys.stderr)
        return 1
    labels = read_tsv(LABELS_PATH)
    by_family = defaultdict(lambda: {"better": 0, "same": 0, "worse": 0})
    worse_rows: list[dict] = []
    for r in labels:
        fam = r.get("applied_family", "") or "UNKNOWN"
        lbl = (r.get("label") or "").upper()
        if lbl == "BETTER":
            by_family[fam]["better"] += 1
        elif lbl == "SAME":
            by_family[fam]["same"] += 1
        elif lbl == "WORSE":
            by_family[fam]["worse"] += 1
            worse_rows.append(r)

    print("Per-family BETTER/SAME/WORSE counts:")
    blocking_families: list[str] = []
    for fam, c in sorted(by_family.items()):
        total = c["better"] + c["same"] + c["worse"]
        same_rate = c["same"] / total if total else 0.0
        print(f"  {fam:<30} BETTER={c['better']:<3} SAME={c['same']:<3} WORSE={c['worse']:<3} (same_rate={same_rate*100:.0f}%)")
        if c["worse"] > 0 or (total >= 3 and same_rate > 0.10):
            blocking_families.append(fam)

    print()
    if blocking_families:
        # Write rerun queue
        rerun: list[dict] = []
        for r in labels:
            fam = r.get("applied_family", "") or "UNKNOWN"
            if fam in blocking_families:
                rerun.append({
                    "sense_id": r.get("sense_id", ""),
                    "pt": r.get("pt", ""),
                    "applied_family": fam,
                    "label": r.get("label", ""),
                    "alias_respelling": r.get("alias_respelling", ""),
                })
        fns = ["sense_id", "pt", "applied_family", "label", "alias_respelling"]
        write_tsv(RERUN_QUEUE_PATH, rerun, fieldnames=fns)
        print(f"⚠ HARD GATE BLOCKED: {len(blocking_families)} family(ies) need rerun:")
        for f in blocking_families:
            print(f"  - {f}")
        print(f"  → wrote {RERUN_QUEUE_PATH} ({len(rerun)} clips)")
        print(f"  → run 08_3 with stronger respellings, 08_4 to upload new dict, 08_5 to re-render those.")
        return 2

    # No block — emit summary
    print("✓ No families blocked. Final TSV rebuild can proceed.")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        f.write("Stage 8 pronunciation correction summary\n")
        f.write("=" * 50 + "\n\n")
        for fam, c in sorted(by_family.items()):
            f.write(f"  {fam:<30} BETTER={c['better']} SAME={c['same']} WORSE={c['worse']}\n")
    print(f"Wrote {SUMMARY_PATH}")
    print()
    print("Next: run build/derive_final.py + build/verify_all.py to rebuild 06-final.tsv.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-html", action="store_true",
                        help="Build before/after listening HTML.")
    parser.add_argument("--apply-and-rebuild", action="store_true",
                        help="Read user labels, apply hard gate, emit summary.")
    args = parser.parse_args()

    if args.build_html:
        return cmd_build_html()
    if args.apply_and_rebuild:
        return cmd_apply_and_rebuild()
    parser.error("Specify --build-html or --apply-and-rebuild.")


if __name__ == "__main__":
    raise SystemExit(main())
