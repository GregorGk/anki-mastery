"""Stage 8 / Step 9 — Pilot test on the 11 phase-7 residual example sentences.

These are the example clips that audio judge flagged as non_bp on
Multilingual v2 (EP/ES drift on 7–10 word sentences). Render each
via Flash v2.5 + no dict, build a side-by-side HTML to verify Flash
fixes them on long sentences.
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"

PHASE7_RESIDUAL_PATH = DATA_DIR / "_audio_phase7_residual.tsv"
MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
HTML_OUT = AUDIT_DIR / "08_9_pilot_flash_examples.html"

FLASH_MODEL_ID = "eleven_flash_v2_5"


def _local_cache_url(sense_id: str, clip_type: str) -> str:
    for v in range(10, 0, -1):
        cache_path = AUDIO_CACHE / f"{sense_id}-{clip_type}-v{v}.mp3"
        if cache_path.exists():
            return f"file://{cache_path}"
    return ""


def main() -> int:
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set", file=sys.stderr)
        return 1

    residuals = read_tsv(PHASE7_RESIDUAL_PATH)
    print(f"Found {len(residuals)} phase-7 residual examples to test on Flash v2.5")

    # Manifest lookup for voice_gender (informational) — voice_id is in residuals
    manifest = read_tsv(MANIFEST_PATH)
    sid_to_manifest = {(r.get("sense_id", ""), r.get("clip_type", "")): r for r in manifest}

    flash_client = ElevenLabsClient(model_id=FLASH_MODEL_ID)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rendered_count = cached_count = failed_count = 0
    rows: list[dict] = []

    for r in residuals:
        sid = r.get("sense_id", "")
        text = r.get("pt_or_text", "")
        voice_id = r.get("voice_id", "")
        drift = r.get("drift", "")
        evidence = r.get("evidence", "")
        if not sid or not text or not voice_id:
            continue

        out = AUDIT_DIR / f"08_9_pilot_flash_ex_{sid}.mp3"
        if out.exists():
            cached_count += 1
        else:
            try:
                res = flash_client.generate_pcm(
                    text=text, voice_id=voice_id,
                    sense_id=f"pilot-flash-ex-{sid}",
                    clip_type="example", version=1,
                )
                norm = normalize_pcm_to_mp3_verified(res.audio_pcm)
                out.write_bytes(norm.mp3_bytes)
                rendered_count += 1
                print(f"  rendered {sid} ({len(text)} chars, voice={voice_id[:10]}...)")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {sid}: {exc}", file=sys.stderr)
                failed_count += 1
                continue

        m = sid_to_manifest.get((sid, "example"), {})
        voice_name_lookup = {
            "4r3G9XKliGgVZLKMgjik": "Lair",
            "AaeZyyi87RCxtFnHPS3e": "Prof. Campanholi",
            "Rw38T6bn0lTNOb1aUevR": "Beatriz",
            "m151rjrbWXbBqyq56tly": "Carla",
            "ny3E2DZImeZm00WLGZi9": "José Paulo",
            "qPfM2laM0pRL4rrZtBGl": "Sandro Dutra",
            "wxoDdfPKBuna5KnUEotz": "Eduardo",
            "xNGAXaCH8MaasNuo7Hr7": "Beto",
        }
        rows.append({
            "sense_id": sid, "text": text, "voice_id": voice_id,
            "voice_name": voice_name_lookup.get(voice_id, voice_id[:10]),
            "drift": drift, "severity": r.get("severity", ""),
            "evidence": evidence,
            "current_url": _local_cache_url(sid, "example"),
            "flash_url": f"file://{out}",
        })

    print(f"\nRendered: {rendered_count} new, {cached_count} cached, {failed_count} failed")

    # Build HTML
    table_rows: list[str] = []
    for r in rows:
        radio_name = f"vote_{r['sense_id']}"
        drift_html = (
            f"<span class='drift drift-{html.escape(r['drift'].lower())}'>{html.escape(r['drift'])}</span>"
            if r['drift'] else "<span class='muted'>—</span>"
        )
        current_audio = (
            f"<audio controls preload='none' src='{html.escape(r['current_url'])}'></audio>"
            if r['current_url'] else "<span class='muted'>(no cache)</span>"
        )
        flash_audio = f"<audio controls preload='none' src='{html.escape(r['flash_url'])}'></audio>"
        table_rows.append(f"""
<tr>
  <td><code>{html.escape(r['sense_id'])}</code></td>
  <td><strong>{html.escape(r['text'])}</strong><br><small class='muted'>voice: {html.escape(r['voice_name'])}</small></td>
  <td>{drift_html}</td>
  <td><small>{html.escape(r['evidence'])}</small></td>
  <td>{current_audio}</td>
  <td>{flash_audio}</td>
  <td class='vote-cell'>
    <label><input type='radio' name='{radio_name}' value='SAME' checked> SAME</label>
    <label><input type='radio' name='{radio_name}' value='FLASH_BETTER'> FLASH</label>
    <label><input type='radio' name='{radio_name}' value='MULTILINGUAL_BETTER'> CURRENT</label>
    <label><input type='radio' name='{radio_name}' value='BOTH_BAD'> BOTH_BAD</label>
  </td>
</tr>
""")

    n_total = len(rows)
    template = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 8 — Flash v2.5 example sentences pilot</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1500px; margin: 1em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
th, td {{ border: 1px solid #aaa; padding: 0.5em 0.7em; text-align: left; vertical-align: top; }}
th {{ background: #e6ecf2; position: sticky; top: 70px; z-index: 5; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
audio {{ width: 220px; }}
.muted {{ color: #999; }}
.drift {{ font-size: 0.85em; padding: 0.1em 0.4em; border-radius: 3px; }}
.drift-en {{ background: #fde2e2; color: #842029; }}
.drift-ep {{ background: #fff3cd; color: #664d03; }}
.drift-es {{ background: #cfe2ff; color: #084298; }}
.vote-cell label {{ margin-right: 0.5em; white-space: nowrap; display: inline-block; font-size: 0.82em; }}
.export {{ position: sticky; top: 0; background: #ffd; padding: 0.6em 1em; border: 2px solid #ba0;
  border-radius: 4px; margin: 0 0 1em 0; z-index: 10; display: flex; align-items: center; gap: 1em; flex-wrap: wrap; }}
button {{ font-size: 1em; padding: 0.5em 1em; cursor: pointer; }}
.intro {{ background: #f5f8fb; border: 1px solid #cdd; border-radius: 6px;
  padding: 1em 1.5em; margin: 1em 0 1em; font-size: 0.95em; }}
#export-output {{ width: 100%; height: 200px; font-family: ui-monospace, monospace; }}
.tally {{ font-weight: bold; }}
.tally span {{ display: inline-block; padding: 0.1em 0.5em; border-radius: 3px; margin: 0 0.2em; }}
.tally-flash {{ background: #d4edda; color: #155724; }}
.tally-current {{ background: #cfe2ff; color: #084298; }}
.tally-same {{ background: #f0f0f0; color: #333; }}
.tally-bad {{ background: #fde2e2; color: #842029; }}
</style>
</head>
<body>

<h1>Stage 8 — Flash v2.5 vs Multilingual v2 (example sentences)</h1>

<div class="intro">
  <p><strong>What this is:</strong> {n_total} example sentences that
  audio judge flagged as non_bp on Multilingual v2 (EP or ES drift).
  Each rendered both ways: <code>CURRENT</code> = Multilingual v2 from
  production; <code>FLASH</code> = Flash v2.5 fresh, no dict.</p>
  <p><strong>Goal:</strong> verify Flash fixes the drift on long
  sentences (7-10 words) too, not just isolated words. If yes, the
  migration covers these cases. If no, we keep them on Multilingual or
  manual-override.</p>
  <p><strong>Vote:</strong> per row, pick the better-sounding audio.
  <code>SAME</code> default.</p>
</div>

<div class="export">
  <button onclick="exportVotes()">Export decisions (copy TSV to clipboard)</button>
  <span id="export-status" style="color: #060;"></span>
  <span class="tally" id="tally"></span>
</div>

<details style="margin-bottom: 1em;">
  <summary>Show raw TSV</summary>
  <textarea id="export-output" readonly></textarea>
</details>

<table>
<tr>
  <th>sense_id</th>
  <th>sentence</th>
  <th>drift</th>
  <th>evidence (audio judge)</th>
  <th>current (Multilingual v2)</th>
  <th>Flash v2.5 (no dict)</th>
  <th>vote</th>
</tr>
{''.join(table_rows)}
</table>

<script>
function updateTally() {{
  let flash = 0, current = 0, same = 0, bad = 0;
  document.querySelectorAll('input[type=radio]:checked').forEach(r => {{
    if (r.value === 'FLASH_BETTER') flash++;
    else if (r.value === 'MULTILINGUAL_BETTER') current++;
    else if (r.value === 'SAME') same++;
    else if (r.value === 'BOTH_BAD') bad++;
  }});
  document.getElementById('tally').innerHTML =
    `<span class="tally-flash">FLASH: ${{flash}}</span>` +
    `<span class="tally-current">CURRENT: ${{current}}</span>` +
    `<span class="tally-same">SAME: ${{same}}</span>` +
    `<span class="tally-bad">BOTH_BAD: ${{bad}}</span>`;
}}
document.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', updateTally));
updateTally();

function exportVotes() {{
  const rows = ["sense_id\\ttext\\tdrift\\tdecision"];
  document.querySelectorAll('table tr').forEach(tr => {{
    const cells = tr.querySelectorAll('td');
    if (cells.length === 0) return;
    const sid = cells[0].textContent.trim();
    const text = cells[1].textContent.trim().split('\\nvoice:')[0].trim();
    const drift = cells[2].textContent.trim();
    const sel = tr.querySelector('input[type=radio]:checked');
    const decision = sel ? sel.value : 'SAME';
    rows.push([sid, text, drift === '—' ? '' : drift, decision].join('\\t'));
  }});
  const tsv = rows.join('\\n');
  document.getElementById('export-output').value = tsv;
  navigator.clipboard.writeText(tsv).then(() => {{
    document.getElementById('export-status').textContent = '✓ Copied to clipboard.';
  }}, () => {{
    document.getElementById('export-status').textContent = 'Copy failed — copy manually from textarea.';
  }});
}}
</script>

</body>
</html>
"""

    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(template, encoding="utf-8")
    print(f"\nWrote {HTML_OUT}")
    print(f"  open: open {HTML_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
