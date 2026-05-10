"""Stage 8 / Step 7 — Final flag HTML for v4 improvement queue.

Lists senses for a binary user pass: each clip is OK (default) or FLAG
(needs v4 improvement). Output TSV is the worklist for a future v4
alias pass.

Default batch (no flags): 27 alias-affected + 23 non-alias non_bp = 50.
Subsequent batches: pass --target-total N (e.g. 100). The script reads
the existing user-saved `_audio_v4_flag_list.tsv` and excludes those
sense_ids automatically — so batches don't overlap.

Audio uses local cache file:// URLs (matches 08_6 working pattern).
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"

MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
APPLICATIONS_PATH = DATA_DIR / "_pronunciation_alias_applications.tsv"
ALIASES_PATH = DATA_DIR / "_pronunciation_aliases.tsv"
VERDICTS_PATH = DATA_DIR / "_audio_judge_verdicts.tsv"
EXISTING_FLAGS_PATH = DATA_DIR / "_audio_v4_flag_list.tsv"
HTML_OUT = AUDIT_DIR / "08_7_final_flag.html"

DEFAULT_TARGET_TOTAL = 50


def _local_cache_url(sense_id: str, clip_type: str) -> str:
    """Walk down from v10 picking first existing local cache file."""
    for v in range(10, 0, -1):
        cache_path = AUDIO_CACHE / f"{sense_id}-{clip_type}-v{v}.mp3"
        if cache_path.exists():
            return f"file://{cache_path}"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL,
                        help="How many clips to include in this HTML (default 50).")
    args = parser.parse_args()
    target_total = args.target_total

    # Read previously-saved flag list to avoid re-showing those senses
    already_shown: set[str] = set()
    if EXISTING_FLAGS_PATH.exists():
        prev = read_tsv(EXISTING_FLAGS_PATH)
        already_shown = {r["sense_id"] for r in prev if r.get("sense_id")}
        print(f"Found {len(already_shown)} sense_ids in {EXISTING_FLAGS_PATH.name}; excluding them.")

    # Load alias applications (sense_ids that received an alias word clip)
    apps = read_tsv(APPLICATIONS_PATH) if APPLICATIONS_PATH.exists() else []
    alias_sense_ids = sorted({
        r["sense_id"] for r in apps
        if r.get("clip_type") == "word" and r.get("sense_id")
        and r["sense_id"] not in already_shown
    })

    # Build sense_id → respelling lookup via alias_id
    aliases_rows = read_tsv(ALIASES_PATH) if ALIASES_PATH.exists() else []
    alias_lookup = {a["alias_id"]: a for a in aliases_rows}
    sid_to_respelling: dict[str, str] = {}
    for r in apps:
        sid = r.get("sense_id", "")
        aid = r.get("alias_id", "")
        if sid and aid in alias_lookup:
            sid_to_respelling[sid] = alias_lookup[aid].get("pt_respelling", "")

    # Load verdicts (drift info + non-alias non_bp pool)
    verdicts = read_tsv(VERDICTS_PATH) if VERDICTS_PATH.exists() else []
    # Dedupe: first verdict per sense_id wins
    sid_to_verdict: dict[str, dict] = {}
    for v in verdicts:
        sid = v.get("sense_id", "")
        if sid and sid not in sid_to_verdict:
            sid_to_verdict[sid] = v

    alias_sid_set = set(alias_sense_ids)
    non_bp_non_alias = [
        v for sid, v in sid_to_verdict.items()
        if v.get("pronunciation_verdict", "") != "bp_ok"
        and sid not in alias_sid_set
        and sid not in already_shown
    ]
    # Sort by rank ascending (sense_id format RRRR.EE.SS sorts to rank order)
    non_bp_non_alias.sort(key=lambda v: v["sense_id"])

    n_to_fill = max(0, target_total - len(alias_sense_ids))
    non_bp_selected = [v["sense_id"] for v in non_bp_non_alias[:n_to_fill]]

    selected_sids = list(alias_sense_ids) + non_bp_selected

    # Manifest lookup for pt + voice_id
    manifest = read_tsv(MANIFEST_PATH)
    sid_to_manifest: dict[tuple[str, str], dict] = {
        (r.get("sense_id", ""), r.get("clip_type", "")): r for r in manifest
    }

    # Build per-row data
    rows: list[dict] = []
    for sid in selected_sids:
        m = sid_to_manifest.get((sid, "word"), {})
        pt = m.get("text_input", "")
        respelling = sid_to_respelling.get(sid, "")
        drift = sid_to_verdict.get(sid, {}).get("drift", "") or ""
        verdict = sid_to_verdict.get(sid, {}).get("pronunciation_verdict", "") or ""
        audio_url = _local_cache_url(sid, "word")
        rows.append({
            "sense_id": sid,
            "pt": pt,
            "respelling": respelling,
            "drift": drift if drift not in ("none", "") else "",
            "verdict": verdict,
            "audio": audio_url,
            "is_alias": sid in set(alias_sense_ids),
        })

    # Build HTML rows
    table_rows: list[str] = []
    for r in rows:
        radio_name = f"flag_{r['sense_id']}"
        respelling_cell = (
            f"<code class='alias'>{html.escape(r['respelling'])}</code>"
            if r['respelling'] else "<span class='muted'>—</span>"
        )
        drift_cell = (
            f"<span class='drift drift-{html.escape(r['drift'].lower() or 'none')}'>"
            f"{html.escape(r['drift'])}</span>"
            if r['drift'] else "<span class='muted'>—</span>"
        )
        audio_cell = (
            f"<audio controls preload='none' src='{html.escape(r['audio'])}'></audio>"
            if r['audio'] else "<span class='muted'>(no cache)</span>"
        )
        row_class = "alias-row" if r['is_alias'] else "other-row"
        table_rows.append(f"""
<tr class="{row_class}">
  <td><code>{html.escape(r['sense_id'])}</code></td>
  <td><strong>{html.escape(r['pt'])}</strong></td>
  <td>{respelling_cell}</td>
  <td>{drift_cell}</td>
  <td>{audio_cell}</td>
  <td class="flag-cell">
    <label><input type="radio" name="{radio_name}" value="OK" checked> OK</label>
    <label><input type="radio" name="{radio_name}" value="FLAG"> FLAG</label>
  </td>
</tr>
""")

    n_alias = len(alias_sense_ids)
    n_other = len(non_bp_selected)
    n_total = len(selected_sids)

    template = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 8 — Final flag pass</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1200px; margin: 2em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; }}
th, td {{ border: 1px solid #aaa; padding: 0.4em 0.6em; text-align: left; vertical-align: middle; }}
th {{ background: #e6ecf2; position: sticky; top: 60px; z-index: 5; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
code.alias {{ background: #d4edda; color: #155724; }}
audio {{ width: 240px; }}
.muted {{ color: #999; }}
.drift {{ font-size: 0.85em; padding: 0.1em 0.4em; border-radius: 3px; }}
.drift-en {{ background: #fde2e2; color: #842029; }}
.drift-ep {{ background: #fff3cd; color: #664d03; }}
.drift-es {{ background: #cfe2ff; color: #084298; }}
.drift-fr {{ background: #f3e5f5; color: #4a148c; }}
.drift-other {{ background: #e0e0e0; color: #333; }}
.flag-cell label {{ margin-right: 0.8em; white-space: nowrap; }}
.alias-row {{ background: #fafffa; }}
.other-row {{ background: #fffafa; }}
.export {{ position: sticky; top: 0; background: #ffd; padding: 0.8em 1em; border: 2px solid #ba0;
  border-radius: 4px; margin: 0 0 1em 0; z-index: 10; display: flex; align-items: center; gap: 1em; }}
button {{ font-size: 1em; padding: 0.5em 1em; cursor: pointer; }}
.intro {{ background: #f5f8fb; border: 1px solid #cdd; border-radius: 6px;
  padding: 1em 1.5em; margin: 1em 0 1.5em; font-size: 0.95em; }}
#export-output {{ width: 100%; height: 200px; font-family: ui-monospace, monospace; }}
.flag-count {{ font-weight: bold; color: #c00; }}
</style>
</head>
<body>

<h1>Stage 8 — Final flag pass</h1>

<div class="intro">
  <p><strong>What this is:</strong> {n_total} clips ({n_alias}
  alias-affected with green respelling cells + {n_other} other audio-judge non_bp
  with red drift labels). For each clip, decide: <code>OK</code> (good
  enough, leave alone) or <code>FLAG</code> (needs v4 improvement).
  Default is <code>OK</code>.</p>
  <p><strong>How to flag:</strong> Click the <code>FLAG</code> radio
  for clips you want revisited. Most clips will stay <code>OK</code> —
  only flag what genuinely bothers you. When done, click Export →
  paste TSV into <code>data/_audio_v4_flag_list.tsv</code>.</p>
  <p><strong>Future v4 pass:</strong> only the FLAG-marked clips get
  re-rendered. The OK ones stay as-is, avoiding TTS-non-determinism
  drift.</p>
</div>

<div class="export">
  <button onclick="exportFlags()">Export flags (copy TSV to clipboard)</button>
  <span id="export-status" style="color: #060;"></span>
  <span id="flag-counter" class="flag-count"></span>
</div>

<details style="margin-bottom: 1em;">
  <summary>Show raw TSV</summary>
  <textarea id="export-output" readonly></textarea>
</details>

<table>
<tr>
  <th>sense_id</th>
  <th>pt</th>
  <th>alias respelling</th>
  <th>drift</th>
  <th>audio</th>
  <th>flag</th>
</tr>
{''.join(table_rows)}
</table>

<script>
function updateCounter() {{
  const flagged = document.querySelectorAll('input[type=radio][value=FLAG]:checked').length;
  document.getElementById('flag-counter').textContent =
    flagged > 0 ? `${{flagged}} flagged` : '';
}}
document.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', updateCounter));

function exportFlags() {{
  const rows = ["sense_id\\tpt\\talias_respelling\\tdrift\\tflag"];
  document.querySelectorAll('table tr').forEach(tr => {{
    const cells = tr.querySelectorAll('td');
    if (cells.length === 0) return;
    const sid = cells[0].textContent.trim();
    const pt = cells[1].textContent.trim();
    const respelling = cells[2].textContent.trim();
    const drift = cells[3].textContent.trim();
    const sel = tr.querySelector('input[type=radio]:checked');
    const flag = sel ? sel.value : 'OK';
    rows.push([sid, pt, respelling === '—' ? '' : respelling,
               drift === '—' ? '' : drift, flag].join('\\t'));
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

    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(template, encoding="utf-8")
    print(f"Wrote {HTML_OUT}")
    print(f"  {n_total} clips: {n_alias} alias-affected + {n_other} non-alias non_bp")
    print(f"  open with:  open {HTML_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
