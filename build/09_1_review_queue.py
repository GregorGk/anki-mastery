"""Stage 9 / Step 1 — Review queue HTML for the 99 ASR-mismatch Flash clips.

For each entry in `_audio_human_review.tsv` whose URL contains
`flash_v2` (i.e. came from the Stage-9 migration), build a row in
`audit/09_1_review_queue.html` with:
  - the current Flash audio (file:// from local cache)
  - sense_id, pt, ASR transcript, sim score, voice name + gender
  - three decision radios:
      LEAVE              — audio is fine, just an ASR artifact
      REGEN_VOICE_SWAP   — re-render with a different voice
      TAG_ALIAS          — needs an alias rule to fix pronunciation
  - a same-gender voice dropdown (used iff REGEN_VOICE_SWAP)
  - an alias respelling text input (used iff TAG_ALIAS)
  - sticky Export TSV button

User exports decisions → `data/_audio_review_queue_decisions.tsv` which
drives the next bulk action: swap voices in manifest + re-render, or
add alias rules + re-render only those senses.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"
CONFIG_DIR = REPO_ROOT / "config"

MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
REVIEW_PATH = DATA_DIR / "_audio_human_review.tsv"
VOICES_PATH = CONFIG_DIR / "voices.tsv"
HTML_OUT = AUDIT_DIR / "09_1_review_queue.html"

# Friendly voice names — keep this in sync with config/voices.tsv
VOICE_NAMES: dict[str, str] = {
    "Rw38T6bn0lTNOb1aUevR": "Beatriz",
    "m151rjrbWXbBqyq56tly": "Carla - Institutional",
    "wxoDdfPKBuna5KnUEotz": "Eduardo",
    "qPfM2laM0pRL4rrZtBGl": "Sandro Dutra",
    "ny3E2DZImeZm00WLGZi9": "José Paulo",
    "AaeZyyi87RCxtFnHPS3e": "Prof. Campanholi",
    "xNGAXaCH8MaasNuo7Hr7": "Beto",
    "4r3G9XKliGgVZLKMgjik": "Lair",
}


def _local_cache_url(url: str) -> str:
    """Derive file:// URL from manifest URL's trailing filename. Returns
    empty string if no local cache file matches."""
    if not url:
        return ""
    filename = url.rsplit("/", 1)[-1]
    cache_path = AUDIO_CACHE / filename
    if cache_path.exists():
        return f"file://{cache_path}"
    return ""


def main() -> int:
    # Voice gender map
    voice_gender: dict[str, str] = {}
    for r in read_tsv(VOICES_PATH):
        voice_gender[r["voice_id"]] = r.get("gender", "")
    female_voices = [(v, VOICE_NAMES.get(v, v[:10])) for v, g in voice_gender.items() if g == "female"]
    male_voices = [(v, VOICE_NAMES.get(v, v[:10])) for v, g in voice_gender.items() if g == "male"]

    # Manifest index for additional context
    manifest = read_tsv(MANIFEST_PATH)
    manifest_idx = {(r["sense_id"], r["clip_type"]): r for r in manifest}

    # Filter review queue to Flash-only entries
    queue = [
        r for r in read_tsv(REVIEW_PATH)
        if "flash_v2" in (r.get("url", "") or "")
    ]
    # Sort by similarity ascending (worst first → most needs decision)
    def _sim(r: dict) -> float:
        try:
            return float(r.get("similarity", "1.0") or "1.0")
        except ValueError:
            return 1.0
    queue.sort(key=_sim)

    if not queue:
        print("No Flash entries in human review queue. Nothing to do.")
        return 0

    print(f"Found {len(queue)} Flash entries in review queue. Building HTML...")

    # Build voice dropdown HTML once (reused per row)
    def _voice_options(current_vid: str, allowed_gender: str) -> str:
        """Build <option> tags. Default selected = current voice; same-gender
        voices appear in their own optgroup, opposite-gender in another."""
        same: list[tuple[str, str]] = female_voices if allowed_gender == "female" else male_voices
        opposite: list[tuple[str, str]] = male_voices if allowed_gender == "female" else female_voices
        parts = [f'<option value="" selected>(current: {html.escape(VOICE_NAMES.get(current_vid, current_vid[:10]))})</option>']
        parts.append(f'<optgroup label="Same gender ({allowed_gender})">')
        for vid, name in same:
            sel = ""  # don't pre-select; current is the default empty option
            if vid == current_vid:
                continue  # don't re-list current
            parts.append(f'<option value="{html.escape(vid)}" {sel}>{html.escape(name)}</option>')
        parts.append('</optgroup>')
        parts.append(f'<optgroup label="Opposite gender (cross-gender swap)">')
        for vid, name in opposite:
            parts.append(f'<option value="{html.escape(vid)}">{html.escape(name)}</option>')
        parts.append('</optgroup>')
        return "".join(parts)

    table_rows: list[str] = []
    for i, r in enumerate(queue):
        sid = r.get("sense_id", "")
        clip_type = r.get("clip_type", "word")
        voice_id = r.get("voice_id", "")
        gender = voice_gender.get(voice_id, "?")
        voice_name = VOICE_NAMES.get(voice_id, voice_id[:10])
        text_input = r.get("input_text", "")
        asr = r.get("asr_transcript", "")
        sim = r.get("similarity", "")
        url = r.get("url", "")
        local_url = _local_cache_url(url)
        radio_name = f"d_{sid}_{clip_type}"
        voice_select_name = f"v_{sid}_{clip_type}"
        alias_input_name = f"a_{sid}_{clip_type}"

        try:
            sim_float = float(sim)
        except ValueError:
            sim_float = 1.0
        sim_class = "low" if sim_float < 0.5 else ("mid" if sim_float < 0.7 else "high")

        audio_el = (
            f"<audio controls preload='none' src='{html.escape(local_url)}'></audio>"
            if local_url else "<span class='muted'>(no cache)</span>"
        )

        table_rows.append(f"""
<tr>
  <td><code>{html.escape(sid)}</code><br><small class='muted'>{html.escape(clip_type)}</small></td>
  <td><strong>{html.escape(text_input)}</strong></td>
  <td>{html.escape(voice_name)}<br><small class='muted'>{html.escape(gender)}</small></td>
  <td>ASR: <em>{html.escape(asr)}</em><br><span class='sim sim-{sim_class}'>sim {html.escape(sim)}</span></td>
  <td>{audio_el}</td>
  <td class='decision-cell'>
    <label><input type='radio' name='{radio_name}' value='LEAVE' checked> LEAVE</label><br>
    <label><input type='radio' name='{radio_name}' value='REGEN_VOICE_SWAP'> REGEN (swap voice)</label>
    <select name='{voice_select_name}' class='voice-select'>{_voice_options(voice_id, gender)}</select><br>
    <label><input type='radio' name='{radio_name}' value='TAG_ALIAS'> TAG_ALIAS</label>
    <input type='text' name='{alias_input_name}' class='alias-input' placeholder='respelling (e.g. ospitau)'>
  </td>
</tr>
""")

    template = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 9 — Flash review queue decisions</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1500px; margin: 1em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
th, td {{ border: 1px solid #aaa; padding: 0.5em 0.7em; text-align: left; vertical-align: top; }}
th {{ background: #e6ecf2; position: sticky; top: 75px; z-index: 5; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
audio {{ width: 220px; }}
.muted {{ color: #999; font-size: 0.9em; }}
.sim {{ font-weight: bold; }}
.sim-low {{ color: #842029; }}
.sim-mid {{ color: #664d03; }}
.sim-high {{ color: #155724; }}
.decision-cell label {{ display: inline-block; margin-right: 0.4em; white-space: nowrap; font-size: 0.85em; }}
.decision-cell .voice-select {{ margin-left: 0.4em; font-size: 0.85em; }}
.decision-cell .alias-input {{ margin-left: 0.4em; font-size: 0.85em; width: 180px; }}
.export {{ position: sticky; top: 0; background: #ffd; padding: 0.6em 1em; border: 2px solid #ba0;
  border-radius: 4px; margin: 0 0 1em 0; z-index: 10; display: flex; align-items: center; gap: 1em; flex-wrap: wrap; }}
button {{ font-size: 1em; padding: 0.5em 1em; cursor: pointer; }}
.intro {{ background: #f5f8fb; border: 1px solid #cdd; border-radius: 6px;
  padding: 1em 1.5em; margin: 1em 0 1em; font-size: 0.95em; }}
#export-output {{ width: 100%; height: 200px; font-family: ui-monospace, monospace; }}
.tally {{ font-weight: bold; }}
.tally span {{ display: inline-block; padding: 0.1em 0.5em; border-radius: 3px; margin: 0 0.2em; }}
.tally-leave {{ background: #d4edda; color: #155724; }}
.tally-swap {{ background: #fff3cd; color: #664d03; }}
.tally-alias {{ background: #cfe2ff; color: #084298; }}
</style>
</head>
<body>

<h1>Stage 9 — Flash ASR-mismatch review queue ({len(queue)} clips)</h1>

<div class="intro">
  <p><strong>What this is:</strong> {len(queue)} clips where Flash v2.5
  produced BP audio that gpt-4o-transcribe couldn't decode back to the
  source text (sim &lt; 0.92 ASR threshold). Audio is already in R2;
  these aren't broken — just ASR can't transcribe.</p>
  <p><strong>Common failure modes:</strong> English loanwords
  (<code>rock, gene, chance, render</code>), short BP words that sound
  similar (<code>meio</code> vs <code>meu</code>), silent BP <code>h</code>
  words (<code>humor</code> → ASR hears "o mar").</p>
  <p><strong>Per row, pick one:</strong></p>
  <ul>
    <li><strong>LEAVE</strong> — audio sounds fine; ASR artifact only.</li>
    <li><strong>REGEN (swap voice)</strong> — try a different voice from
        the dropdown. Use same gender to preserve speaker_gender
        assignment; cross-gender only if you really want to change.</li>
    <li><strong>TAG_ALIAS</strong> — add an alias rule. Type the
        respelling (e.g. <code>ospitau</code> for <code>hospital</code>).
        We'll add it to the dict and re-render only that sense.</li>
  </ul>
</div>

<div class="export">
  <button onclick="exportDecisions()">Export decisions (copy TSV to clipboard)</button>
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
  <th>voice</th>
  <th>ASR mismatch</th>
  <th>Flash audio</th>
  <th>decision</th>
</tr>
{''.join(table_rows)}
</table>

<script>
function updateTally() {{
  let leave = 0, swap = 0, alias = 0;
  document.querySelectorAll('input[type=radio]:checked').forEach(r => {{
    if (r.value === 'LEAVE') leave++;
    else if (r.value === 'REGEN_VOICE_SWAP') swap++;
    else if (r.value === 'TAG_ALIAS') alias++;
  }});
  document.getElementById('tally').innerHTML =
    `<span class="tally-leave">LEAVE: ${{leave}}</span>` +
    `<span class="tally-swap">REGEN_SWAP: ${{swap}}</span>` +
    `<span class="tally-alias">TAG_ALIAS: ${{alias}}</span>`;
}}
document.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', updateTally));
updateTally();

function exportDecisions() {{
  const rows = ["sense_id\\tclip_type\\tpt\\tcurrent_voice_id\\tasr_transcript\\tsimilarity\\tdecision\\tnew_voice_id\\talias_respelling"];
  document.querySelectorAll('table tr').forEach(tr => {{
    const cells = tr.querySelectorAll('td');
    if (cells.length === 0) return;
    const sidCell = cells[0].textContent.trim().split(/\\s+/);
    const sid = sidCell[0];
    const clipType = sidCell[1] || '';
    const pt = cells[1].textContent.trim();
    const voiceCell = cells[2].textContent.trim();
    const voiceName = voiceCell.split(/\\s+/).slice(0, -1).join(' '); // strip trailing gender
    // Find the current voice_id by matching name → use a map
    const voiceMap = {{
      'Beatriz': 'Rw38T6bn0lTNOb1aUevR',
      'Carla - Institutional': 'm151rjrbWXbBqyq56tly',
      'Eduardo': 'wxoDdfPKBuna5KnUEotz',
      'Sandro Dutra': 'qPfM2laM0pRL4rrZtBGl',
      'José Paulo': 'ny3E2DZImeZm00WLGZi9',
      'Prof. Campanholi': 'AaeZyyi87RCxtFnHPS3e',
      'Beto': 'xNGAXaCH8MaasNuo7Hr7',
      'Lair': '4r3G9XKliGgVZLKMgjik',
    }};
    const currentVoiceId = voiceMap[voiceName] || '';
    const asrLine = cells[3].textContent.trim();
    const asrMatch = asrLine.match(/ASR:\\s*(.+?)sim\\s+(\\S+)/);
    const asrTranscript = asrMatch ? asrMatch[1].trim() : '';
    const sim = asrMatch ? asrMatch[2] : '';
    const sel = tr.querySelector('input[type=radio]:checked');
    const decision = sel ? sel.value : 'LEAVE';
    const newVoice = tr.querySelector('select.voice-select');
    const newVoiceId = (decision === 'REGEN_VOICE_SWAP' && newVoice) ? newVoice.value : '';
    const aliasInput = tr.querySelector('input.alias-input');
    const aliasResp = (decision === 'TAG_ALIAS' && aliasInput) ? aliasInput.value.trim() : '';
    rows.push([sid, clipType, pt, currentVoiceId, asrTranscript, sim, decision, newVoiceId, aliasResp].join('\\t'));
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
    print(f"Wrote {HTML_OUT}")
    print(f"  open: open {HTML_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
