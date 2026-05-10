"""Stage 8 / Step 8 — Pilot test: Flash v2.5 vs Multilingual v2.

Render ~127 risky senses (27 alias-affected + 100 non-alias non_bp from
audio judge) via eleven_flash_v2_5 + no dict, build side-by-side HTML
against current production audio.

Voice diversity is guaranteed by the audio_judge non_bp pool covering
all 11 production voices proportionally to their drift rates.
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"

MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
APPLICATIONS_PATH = DATA_DIR / "_pronunciation_alias_applications.tsv"
VERDICTS_PATH = DATA_DIR / "_audio_judge_verdicts.tsv"
V4_FLAG_LIST_PATH = DATA_DIR / "_audio_v4_flag_list.tsv"
HTML_OUT = AUDIT_DIR / "08_8_pilot_flash.html"

PILOT_NON_ALIAS_COUNT = 100
FLASH_MODEL_ID = "eleven_flash_v2_5"


def _local_cache_url(sense_id: str, clip_type: str) -> str:
    """Walk down from v10 picking first existing local cache file."""
    for v in range(10, 0, -1):
        cache_path = AUDIO_CACHE / f"{sense_id}-{clip_type}-v{v}.mp3"
        if cache_path.exists():
            return f"file://{cache_path}"
    return ""


def _check_voice_flash_compat(voice_id: str, api_key: str) -> bool:
    try:
        resp = httpx.get(
            f"https://api.elevenlabs.io/v1/voices/{voice_id}",
            headers={"xi-api-key": api_key},
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return FLASH_MODEL_ID in (data.get("high_quality_base_model_ids") or [])
    except Exception as exc:  # noqa: BLE001
        print(f"  voice compat check failed for {voice_id[:10]}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set", file=sys.stderr)
        return 1

    # Pool 1: alias-affected sense_ids (word clips)
    apps = read_tsv(APPLICATIONS_PATH)
    alias_sense_ids = sorted({
        r["sense_id"] for r in apps
        if r.get("clip_type") == "word" and r.get("sense_id")
    })

    # Pool 2: non-alias non_bp from audio judge
    verdicts = read_tsv(VERDICTS_PATH)
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
    ]
    non_bp_non_alias.sort(key=lambda v: v["sense_id"])
    non_alias_selected = [v["sense_id"] for v in non_bp_non_alias[:PILOT_NON_ALIAS_COUNT]]

    selected_sids = list(alias_sense_ids) + non_alias_selected
    n_alias = len(alias_sense_ids)
    n_non_alias = len(non_alias_selected)
    print(f"Pilot pool: {len(selected_sids)} senses ({n_alias} alias + {n_non_alias} non-alias non_bp)")

    # Manifest lookup
    manifest = read_tsv(MANIFEST_PATH)
    sid_to_manifest = {(r.get("sense_id", ""), r.get("clip_type", "")): r for r in manifest}

    # V4 flag context (from prior listening pass)
    sid_to_flag: dict[str, str] = {}
    if V4_FLAG_LIST_PATH.exists():
        for r in read_tsv(V4_FLAG_LIST_PATH):
            sid = r.get("sense_id", "")
            if sid:
                sid_to_flag[sid] = r.get("flag", "")

    # Voice compat check (one API call per unique voice)
    unique_voices = {
        sid_to_manifest.get((sid, "word"), {}).get("voice_id", "")
        for sid in selected_sids
    }
    unique_voices.discard("")
    print(f"Checking Flash v2.5 compatibility for {len(unique_voices)} unique voices...")
    voice_compat: dict[str, bool] = {}
    for vid in sorted(unique_voices):
        compat = _check_voice_flash_compat(vid, api_key)
        voice_compat[vid] = compat
        print(f"  {vid[:10]}: {'✓ compatible' if compat else '✗ INCOMPATIBLE'}")
    n_incompat = sum(1 for c in voice_compat.values() if not c)
    if n_incompat:
        print(f"WARNING: {n_incompat} voice(s) incompatible — those senses will be skipped.")

    # Render Flash v2.5 audio for each compatible sense
    flash_client = ElevenLabsClient(model_id=FLASH_MODEL_ID)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rendered_count = cached_count = skipped_count = 0
    rows: list[dict] = []

    for i, sid in enumerate(selected_sids, start=1):
        m = sid_to_manifest.get((sid, "word"), {})
        pt = m.get("text_input", "")
        vid = m.get("voice_id", "")
        if not pt or not vid or not voice_compat.get(vid, False):
            skipped_count += 1
            continue

        out = AUDIT_DIR / f"08_8_pilot_flash_{sid}.mp3"
        if out.exists():
            cached_count += 1
        else:
            try:
                res = flash_client.generate_pcm(
                    text=pt, voice_id=vid,
                    sense_id=f"pilot-flash-{sid}",
                    clip_type="word", version=1,
                )
                norm = normalize_pcm_to_mp3_verified(res.audio_pcm)
                out.write_bytes(norm.mp3_bytes)
                rendered_count += 1
                if rendered_count % 10 == 0:
                    print(f"  rendered {rendered_count}/{len(selected_sids)} (sid={sid})")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {sid} ({pt!r}): {exc}", file=sys.stderr)
                skipped_count += 1
                continue

        flash_url = f"file://{out}"
        current_url = _local_cache_url(sid, "word")
        v4_flag = sid_to_flag.get(sid, "")
        is_alias = sid in alias_sid_set
        drift = (sid_to_verdict.get(sid, {}).get("drift", "") or "").strip()
        if drift == "none":
            drift = ""
        rows.append({
            "sense_id": sid, "pt": pt, "voice_id": vid,
            "is_alias": is_alias, "v4_flag": v4_flag, "drift": drift,
            "current_url": current_url, "flash_url": flash_url,
        })

    print(f"\nRendered: {rendered_count} new, {cached_count} cached, {skipped_count} skipped")

    # Build HTML
    table_rows: list[str] = []
    for r in rows:
        radio_name = f"vote_{r['sense_id']}"
        if r['v4_flag'] == "FLAG":
            v4_html = "<span class='flag flag-yes'>FLAG</span>"
        elif r['v4_flag'] == "OK":
            v4_html = "<span class='flag flag-ok'>OK</span>"
        else:
            v4_html = "<span class='muted'>—</span>"
        drift_html = (
            f"<span class='drift drift-{html.escape(r['drift'].lower())}'>{html.escape(r['drift'])}</span>"
            if r['drift'] else "<span class='muted'>—</span>"
        )
        alias_marker = "<span class='alias-marker'>ALIAS</span>" if r['is_alias'] else ""
        current_audio = (
            f"<audio controls preload='none' src='{html.escape(r['current_url'])}'></audio>"
            if r['current_url'] else "<span class='muted'>(no cache)</span>"
        )
        flash_audio = (
            f"<audio controls preload='none' src='{html.escape(r['flash_url'])}'></audio>"
        )
        table_rows.append(f"""
<tr>
  <td><code>{html.escape(r['sense_id'])}</code>{alias_marker}</td>
  <td><strong>{html.escape(r['pt'])}</strong></td>
  <td>{v4_html}</td>
  <td>{drift_html}</td>
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
<title>Stage 8 — Flash v2.5 pilot</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1500px; margin: 1em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; }}
th, td {{ border: 1px solid #aaa; padding: 0.4em 0.6em; text-align: left; vertical-align: middle; }}
th {{ background: #e6ecf2; position: sticky; top: 70px; z-index: 5; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
audio {{ width: 200px; }}
.muted {{ color: #999; }}
.flag {{ font-size: 0.85em; padding: 0.1em 0.4em; border-radius: 3px; font-weight: bold; }}
.flag-yes {{ background: #fde2e2; color: #842029; }}
.flag-ok {{ background: #d4edda; color: #155724; }}
.drift {{ font-size: 0.85em; padding: 0.1em 0.4em; border-radius: 3px; }}
.drift-en {{ background: #fde2e2; color: #842029; }}
.drift-ep {{ background: #fff3cd; color: #664d03; }}
.drift-es {{ background: #cfe2ff; color: #084298; }}
.drift-fr {{ background: #f3e5f5; color: #4a148c; }}
.drift-other {{ background: #e0e0e0; color: #333; }}
.alias-marker {{ background: #cfe2ff; color: #084298; padding: 0.1em 0.3em; border-radius: 3px;
  font-size: 0.7em; font-weight: bold; margin-left: 0.4em; }}
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

<h1>Stage 8 — Flash v2.5 vs Multilingual v2 pilot</h1>

<div class="intro">
  <p><strong>What this is:</strong> {n_total} risky clips rendered both
  ways. <code>CURRENT</code> column is your existing production audio
  (Multilingual v2, with v3 alias dict applied where applicable —
  <span class='alias-marker'>ALIAS</span> tag in column 1).
  <code>FLASH</code> column is a fresh Flash v2.5 render with NO dict
  (testing native BP defaults).</p>
  <p><strong>How to vote:</strong> Per row, pick which sounds more
  BP-correct: <code>FLASH</code>, <code>CURRENT</code>,
  <code>SAME</code> (default — no audible difference / tie), or
  <code>BOTH_BAD</code>. Live tally shows in the export bar.</p>
  <p><strong>Decision threshold:</strong> If <code>FLASH</code> dominates
  by ~2× over <code>CURRENT</code>, switching the deck makes sense
  (drops alias complexity, ~50% cheaper renders). Otherwise stay on
  Multilingual v2.</p>
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
  <th>pt</th>
  <th>v4 flag</th>
  <th>drift</th>
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
  const rows = ["sense_id\\tpt\\tv4_flag\\tdrift\\tis_alias\\tdecision"];
  document.querySelectorAll('table tbody tr, table tr').forEach(tr => {{
    const cells = tr.querySelectorAll('td');
    if (cells.length === 0) return;
    const sidText = cells[0].textContent.trim();
    const sid = sidText.split(/\\s+/)[0];
    const isAlias = sidText.includes('ALIAS') ? 'true' : 'false';
    const pt = cells[1].textContent.trim();
    const v4 = cells[2].textContent.trim();
    const drift = cells[3].textContent.trim();
    const sel = tr.querySelector('input[type=radio]:checked');
    const decision = sel ? sel.value : 'SAME';
    rows.push([sid, pt, v4 === '—' ? '' : v4, drift === '—' ? '' : drift, isAlias, decision].join('\\t'));
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
