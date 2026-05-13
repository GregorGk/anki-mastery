"""Stage 16 / Step 9 — Render the Gemini drift watchlist as a review HTML.

Reads:
    data/_gemini_drift_watchlist.tsv     (222 production-deck drift rows)
    data/_audio_manifest.tsv             (for audio URL per row)
    data/06-final.tsv                    (for English gloss + Stage 15 metadata)
    data/_ep_cues.tsv                    (optional per-text BP/EP listening cues)

Writes:
    audit/16_8_drift_watchlist.html

The HTML lets the user:
  - Read the watchlist with audio playback, Gemini's evidence, and tips.
  - Filter by voice / severity.
  - Mark each entry agree / disagree / skip — exports a TSV of human
    overrides (e.g. for senses where Gemini called drift but it actually
    sounds fine).

Usage:
    .venv/bin/python build/16_9_drift_watchlist_html.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"
CONFIG = REPO_ROOT / "config"

WATCHLIST = DATA / "_gemini_drift_watchlist.tsv"
MANIFEST = DATA / "_audio_manifest.tsv"
FINAL_TSV = DATA / "06-final.tsv"
CUES_TSV = DATA / "_ep_cues.tsv"
VOICES_TSV = CONFIG / "voices.tsv"
OUT_HTML = AUDIT / "16_8_drift_watchlist.html"


def _sev_rank(s: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(s, 3)


def _sev_class(s: str) -> str:
    return {"high": "sev-high", "medium": "sev-med", "low": "sev-low"}.get(s, "")


def _conf_class(s: str) -> str:
    return {"high": "conf-high", "medium": "conf-med", "low": "conf-low"}.get(s, "")


def main() -> int:
    if not WATCHLIST.exists():
        print(f"ERROR: {WATCHLIST} missing. Run build/16_8_gemini_qa_gate.py first.",
              file=sys.stderr)
        return 1

    watch = list(csv.DictReader(WATCHLIST.open(encoding="utf-8"), dialect="excel-tab"))

    # Manifest: (sense_id, clip_type, voice_id) → row (for url + object_key).
    manifest_lookup: dict[tuple[str, str, str], dict] = {}
    for r in csv.DictReader(MANIFEST.open(encoding="utf-8"), dialect="excel-tab"):
        manifest_lookup[(r["sense_id"], r["clip_type"], r["voice_id"])] = r

    # Final TSV: sense_id → row (for en_word + en_example + bp_validity etc.).
    final_lookup: dict[str, dict] = {}
    for r in csv.DictReader(FINAL_TSV.open(encoding="utf-8"), dialect="excel-tab"):
        final_lookup[r["sense_id"]] = r

    # EP cues: (text, clip_type) → row.
    cues_lookup: dict[tuple[str, str], dict] = {}
    if CUES_TSV.exists():
        for r in csv.DictReader(CUES_TSV.open(encoding="utf-8"), dialect="excel-tab"):
            cues_lookup[(r["text"], r["clip_type"])] = r

    # Sort by severity, confidence, voice, sense_id.
    watch.sort(key=lambda r: (
        _sev_rank(r.get("severity", "")),
        _sev_rank(r.get("confidence", "")),  # same rank table works
        r.get("voice_name", ""),
        r.get("sense_id", ""),
    ))

    # ── Header stats ───────────────────────────────────────────────
    voice_counts = Counter(r["voice_name"] for r in watch)
    sev_counts = Counter(r["severity"] for r in watch)
    clip_counts = Counter(r["clip_type"] for r in watch)

    # Deck-wide denominators per voice (BP v3 word+example, status=uploaded).
    voice_id_to_name: dict[str, str] = {}
    active_voice_ids: set[str] = set()
    for r in csv.DictReader(VOICES_TSV.open(encoding="utf-8"), dialect="excel-tab"):
        voice_id_to_name[r["voice_id"]] = r.get("bp_name", "")
        if r.get("status", "") == "active":
            active_voice_ids.add(r["voice_id"])
    voice_totals: Counter = Counter()
    for r in csv.DictReader(MANIFEST.open(encoding="utf-8"), dialect="excel-tab"):
        if (r["clip_type"] in ("word", "example")
                and r["tts_model"] == "eleven_v3"
                and r["status"] == "uploaded"
                and r["voice_id"] in active_voice_ids):
            voice_totals[voice_id_to_name[r["voice_id"]]] += 1
    deck_total = sum(voice_totals.values())

    # ── Render ─────────────────────────────────────────────────────
    head_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 16.8 — Gemini drift watchlist</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #181b22;
    --panel-2: #20242d;
    --border: #2c3140;
    --text: #e6e9ef;
    --muted: #9098a6;
    --accent: #5aa9ff;
    --sev-high: #ff6b6b;
    --sev-med:  #ffa94d;
    --sev-low:  #ffe66d;
    --ok: #4ade80;
    --warn: #fbbf24;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font: 14px/1.45
         -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 20px; }
  h1, h2, h3 { margin: 0 0 12px; }
  h1 { font-size: 22px; }
  h2 { font-size: 18px; color: var(--muted); }
  .container { max-width: 1280px; margin: 0 auto; }
  .stats { display: grid; grid-template-columns: 2fr 3fr; gap: 16px;
           margin-bottom: 18px; }
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 14px 16px; }
  .panel h3 { font-size: 13px; color: var(--muted); text-transform: uppercase;
              letter-spacing: 0.5px; margin-bottom: 8px; }
  table.summary { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.summary td, table.summary th { padding: 4px 8px; text-align: left;
                                       border-bottom: 1px solid var(--border); }
  table.summary th { color: var(--muted); font-weight: 500; }
  .pct { color: var(--muted); }
  .pct.bad { color: var(--sev-high); }
  .pct.warn { color: var(--sev-med); }

  details.tips { background: var(--panel); border: 1px solid var(--border);
                 border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }
  details.tips summary { cursor: pointer; font-weight: 600; color: var(--accent); }
  details.tips ul { margin: 10px 0 0 18px; padding: 0; }
  details.tips li { margin: 4px 0; }
  details.tips code { background: var(--panel-2); padding: 1px 5px;
                      border-radius: 3px; font-size: 12px; }

  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
              margin-bottom: 14px; padding: 12px 14px; background: var(--panel);
              border: 1px solid var(--border); border-radius: 8px;
              position: sticky; top: 0; z-index: 5; }
  .controls label { color: var(--muted); font-size: 12px; }
  .controls select, .controls input { background: var(--panel-2);
              color: var(--text); border: 1px solid var(--border);
              border-radius: 4px; padding: 5px 8px; font-size: 13px; }
  .controls .grow { flex: 1; }
  button { background: var(--accent); color: #fff; border: 0; padding: 7px 14px;
           border-radius: 5px; cursor: pointer; font-weight: 600; font-size: 13px; }
  button.secondary { background: var(--panel-2); color: var(--text);
                     border: 1px solid var(--border); }
  button:hover { filter: brightness(1.1); }
  #count { color: var(--muted); font-size: 12px; }

  .entry { background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
           display: grid; grid-template-columns: 1fr 320px; gap: 16px; }
  .entry.hidden { display: none; }
  .e-left { min-width: 0; }
  .e-meta { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
            margin-bottom: 8px; font-size: 12px; }
  .e-meta .id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                color: var(--muted); }
  .e-meta .voice { background: var(--panel-2); padding: 2px 8px;
                   border-radius: 3px; }
  .badge { padding: 2px 8px; border-radius: 3px; font-weight: 600;
           font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }
  .sev-high { background: rgba(255,107,107,0.18); color: var(--sev-high); }
  .sev-med  { background: rgba(255,169,77,0.18);  color: var(--sev-med); }
  .sev-low  { background: rgba(255,230,109,0.18); color: var(--sev-low); }
  .conf-high { background: rgba(74,222,128,0.18); color: var(--ok); }
  .conf-med  { background: rgba(251,191,36,0.18); color: var(--warn); }
  .conf-low  { background: rgba(144,152,166,0.20); color: var(--muted); }

  .e-text { font-size: 18px; line-height: 1.3; margin: 4px 0 8px;
            color: var(--text); }
  .e-en   { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
  .e-evidence { background: var(--panel-2); border-left: 3px solid var(--accent);
                padding: 8px 12px; border-radius: 0 4px 4px 0;
                font-size: 13px; line-height: 1.5; margin: 8px 0; }
  .e-evidence .label { color: var(--accent); font-weight: 600; margin-right: 6px; }
  .e-cues { background: var(--panel-2); padding: 8px 12px; border-radius: 4px;
            font-size: 12px; margin-top: 8px; }
  .e-cues h4 { margin: 0 0 4px; font-size: 11px; color: var(--muted);
               text-transform: uppercase; letter-spacing: 0.5px; }
  .e-cues .ipa { font-family: ui-monospace, SFMono-Regular, monospace; }
  .e-cues ul { margin: 4px 0 0 16px; padding: 0; }

  .e-right { display: flex; flex-direction: column; gap: 8px; }
  audio { width: 100%; }
  .review { display: flex; gap: 4px; }
  .review label { flex: 1; background: var(--panel-2); border: 1px solid var(--border);
                  padding: 6px 0; border-radius: 4px; text-align: center;
                  font-size: 12px; cursor: pointer; user-select: none;
                  transition: all 0.1s; }
  .review label:hover { border-color: var(--accent); }
  .review input { display: none; }
  .review input:checked + span { color: #fff; font-weight: 600; }
  .review label.agree input:checked ~ span,
  .review label.agree:has(input:checked) { background: rgba(255,107,107,0.25);
                                            border-color: var(--sev-high); }
  .review label.disagree:has(input:checked) { background: rgba(74,222,128,0.25);
                                              border-color: var(--ok); }
  .review label.skip:has(input:checked) { background: var(--panel-2);
                                          border-color: var(--muted); }
  .e-stage15 { font-size: 11px; color: var(--muted); margin-top: 4px; }
  .e-stage15 .flag { background: rgba(251,191,36,0.18); color: var(--warn);
                     padding: 1px 6px; border-radius: 3px; margin-right: 4px; }
</style>
</head>
<body>
<div class="container">
"""

    parts: list[str] = [head_html]
    parts.append('<h1>Stage 16.8 — Gemini drift watchlist</h1>')
    parts.append(f'<h2>{len(watch)} production-deck drifts out of {deck_total:,} '
                 f'BP v3 word+example clips '
                 f'({100*len(watch)/max(deck_total,1):.1f}%)</h2>')

    # Stats grid: per-voice + per-severity.
    parts.append('<div class="stats">')
    parts.append('<div class="panel"><h3>Per-voice drift rate</h3>'
                 '<table class="summary"><tr>'
                 '<th>voice</th><th style="text-align:right">drift</th>'
                 '<th style="text-align:right">total</th>'
                 '<th style="text-align:right">%</th></tr>')
    for vname in sorted(voice_counts.keys(), key=lambda v: -voice_counts[v]):
        n = voice_counts[vname]
        denom = voice_totals.get(vname, 0)
        pct = 100 * n / denom if denom else 0
        cls = "bad" if pct >= 5 else ("warn" if pct >= 2 else "")
        parts.append(f'<tr><td>{escape(vname)}</td>'
                     f'<td style="text-align:right">{n}</td>'
                     f'<td style="text-align:right">{denom}</td>'
                     f'<td style="text-align:right" class="pct {cls}">'
                     f'{pct:.1f}%</td></tr>')
    parts.append('</table></div>')

    parts.append('<div class="panel"><h3>By severity × clip type</h3>'
                 '<table class="summary"><tr><th>severity</th>'
                 '<th style="text-align:right">word</th>'
                 '<th style="text-align:right">example</th>'
                 '<th style="text-align:right">total</th></tr>')
    sev_clip = Counter((r["severity"], r["clip_type"]) for r in watch)
    for sev in ("high", "medium", "low"):
        w_ = sev_clip.get((sev, "word"), 0)
        e_ = sev_clip.get((sev, "example"), 0)
        if not (w_ or e_):
            continue
        parts.append(f'<tr><td><span class="badge {_sev_class(sev)}">'
                     f'{sev}</span></td>'
                     f'<td style="text-align:right">{w_}</td>'
                     f'<td style="text-align:right">{e_}</td>'
                     f'<td style="text-align:right"><b>{w_+e_}</b></td></tr>')
    parts.append('</table></div></div>')  # close stats

    # Tips section.
    parts.append('''
<details class="tips" open>
<summary>🎧 What to listen for — BP vs EP audio cues</summary>
<ul>
  <li><b>Final unstressed /e/</b> — BP palatalizes to <code>[dʒi]</code>
      / <code>[tʃi]</code> in words ending in <i>-de</i>, <i>-te</i>
      (e.g. <i>verde</i> = "verji", <i>noite</i> = "noytʃi"). EP keeps
      a clean <code>[dɨ]</code> / <code>[tɨ]</code> or near-silent
      schwa.</li>
  <li><b>Unstressed /e/ everywhere else</b> — EP reduces heavily to
      <code>[ɨ]</code> or omits it (<i>pequeno</i> → "pquenu"). BP
      keeps it as a clear <code>[e]</code> or <code>[i]</code>.</li>
  <li><b>Coda /s/ before consonant or pause</b> — EP <code>[ʃ]</code>
      ("ash"), BP <code>[s]</code> ("ass"). <i>fez</i>: BP
      <code>[fes]</code>, EP <code>[feʃ]</code>.</li>
  <li><b>/r/ at end of syllable</b> — BP often retracted/velar/uvular
      (varies by region); EP usually a clean alveolar tap or trill.</li>
  <li><b>/v/</b> — should be labiodental <code>[v]</code> in both.
      A bilabial <code>[β]</code> (Spanish-like) is a sign of foreign
      training data leaking into the model.</li>
  <li><b>Lexical EP markers</b> — <i>sítio</i> (= place, in EP),
      <i>aldeia</i>, <i>concelho</i>, <i>comboio</i>, <i>autocarro</i>,
      <i>telemóvel</i>. BP would use <i>local/lugar</i>, <i>vila</i>,
      <i>município</i>, <i>trem</i>, <i>ônibus</i>, <i>celular</i>.</li>
  <li><b>Click 🎓 inside an entry</b> to reveal per-text IPA cues
      (only available for AB-pool cards).</li>
</ul>
</details>
''')

    # Controls.
    voice_options = ['<option value="">All voices</option>']
    for v in sorted(voice_counts.keys()):
        voice_options.append(f'<option value="{escape(v)}">{escape(v)} '
                             f'({voice_counts[v]})</option>')
    parts.append('<div class="controls">')
    parts.append(f'<label>Voice <select id="filter-voice">'
                 f'{"".join(voice_options)}</select></label>')
    parts.append('<label>Severity <select id="filter-sev">'
                 '<option value="">all</option>'
                 '<option value="high">high</option>'
                 '<option value="medium">medium</option>'
                 '<option value="low">low</option></select></label>')
    parts.append('<label>Clip type <select id="filter-clip">'
                 '<option value="">all</option>'
                 '<option value="word">word</option>'
                 '<option value="example">example</option></select></label>')
    parts.append('<label>Review <select id="filter-review">'
                 '<option value="">all</option>'
                 '<option value="unreviewed">unreviewed</option>'
                 '<option value="agree">agree</option>'
                 '<option value="disagree">disagree</option>'
                 '<option value="skip">skip</option></select></label>')
    parts.append('<span class="grow"></span>')
    parts.append('<span id="count"></span>')
    parts.append('<button id="export-btn">Export labels TSV</button>')
    parts.append('</div>')

    # Entries.
    parts.append('<div id="entries">')
    for i, r in enumerate(watch, 1):
        sid = r["sense_id"]
        clip = r["clip_type"]
        vid = r["voice_id"]
        vname = r["voice_name"]
        text = r["text"]
        sev = r["severity"]
        conf = r["confidence"]
        evidence = r.get("evidence", "")
        drift = r.get("drift", "")

        m = manifest_lookup.get((sid, clip, vid), {})
        url = m.get("url", "")
        fr = final_lookup.get(sid, {})
        en_glosses = []
        if clip == "word" and fr.get("en_word"):
            en_glosses.append(escape(fr["en_word"]))
        if clip == "example" and fr.get("en_example"):
            en_glosses.append(escape(fr["en_example"]))
        if not en_glosses:
            if fr.get("en_word"): en_glosses.append(escape(fr["en_word"]))
        bp_validity = fr.get("bp_validity", "")
        risk_flags = fr.get("risk_flags", "")

        cue = cues_lookup.get((text, clip))

        parts.append(f'<div class="entry" data-voice="{escape(vname)}" '
                     f'data-sev="{sev}" data-clip="{clip}" '
                     f'data-review="" data-sense="{sid}" data-vid="{vid}">')

        # left column
        parts.append('<div class="e-left">')
        parts.append('<div class="e-meta">')
        parts.append(f'<span class="id">#{i:03d} · {escape(sid)} · {escape(clip)}</span>')
        parts.append(f'<span class="voice">{escape(vname)}</span>')
        parts.append(f'<span class="badge {_sev_class(sev)}">sev: {escape(sev)}</span>')
        parts.append(f'<span class="badge {_conf_class(conf)}">conf: {escape(conf)}</span>')
        if drift and drift != sev:
            parts.append(f'<span class="badge" style="background:var(--panel-2);'
                         f'color:var(--muted)">{escape(drift)}</span>')
        parts.append('</div>')
        parts.append(f'<div class="e-text">{escape(text)}</div>')
        if en_glosses:
            parts.append(f'<div class="e-en">↪ {" / ".join(en_glosses)}</div>')
        if evidence:
            parts.append('<div class="e-evidence">'
                         f'<span class="label">Gemini:</span>{escape(evidence)}</div>')
        if cue:
            parts.append('<div class="e-cues">')
            parts.append('<h4>📚 BP vs EP listening cue</h4>')
            if cue.get("bp_ipa") or cue.get("ep_ipa"):
                parts.append(f'<div><span class="ipa">BP: {escape(cue.get("bp_ipa",""))}</span></div>')
                parts.append(f'<div><span class="ipa">EP: {escape(cue.get("ep_ipa",""))}</span></div>')
            if cue.get("headline_marker"):
                parts.append(f'<div style="margin-top:4px"><b>Headline:</b> '
                             f'{escape(cue["headline_marker"])}</div>')
            cue_items = [cue.get(k, "") for k in ("cue_1", "cue_2", "cue_3", "cue_4")]
            cue_items = [c for c in cue_items if c]
            if cue_items:
                parts.append('<ul>')
                for c in cue_items:
                    parts.append(f'<li>{escape(c)}</li>')
                parts.append('</ul>')
            parts.append('</div>')
        if bp_validity and bp_validity != "standard":
            stg15 = f'<span class="flag">Stage 15: {escape(bp_validity)}</span>'
            if risk_flags:
                stg15 += f'<span class="flag">{escape(risk_flags)}</span>'
            parts.append(f'<div class="e-stage15">{stg15}</div>')
        parts.append('</div>')  # /e-left

        # right column
        parts.append('<div class="e-right">')
        if url:
            parts.append(f'<audio controls preload="none" src="{escape(url)}"></audio>')
        else:
            parts.append('<div style="color:var(--muted)">(no audio url found)</div>')
        radio_name = f"rev-{i}"
        parts.append(f'<div class="review">'
                     f'<label class="agree"><input type="radio" name="{radio_name}" value="agree" data-i="{i}"><span>Agree (drift)</span></label>'
                     f'<label class="disagree"><input type="radio" name="{radio_name}" value="disagree" data-i="{i}"><span>Disagree (OK)</span></label>'
                     f'<label class="skip"><input type="radio" name="{radio_name}" value="skip" data-i="{i}"><span>Skip</span></label>'
                     f'</div>')
        parts.append('</div>')  # /e-right

        parts.append('</div>')  # /entry

    parts.append('</div>')  # /entries

    # JS for filter + export.
    parts.append('''
<script>
(() => {
  const entries = Array.from(document.querySelectorAll('.entry'));
  const fVoice = document.getElementById('filter-voice');
  const fSev   = document.getElementById('filter-sev');
  const fClip  = document.getElementById('filter-clip');
  const fRev   = document.getElementById('filter-review');
  const countEl = document.getElementById('count');

  function apply() {
    const v = fVoice.value, s = fSev.value, c = fClip.value, r = fRev.value;
    let shown = 0;
    for (const el of entries) {
      const okV = !v || el.dataset.voice === v;
      const okS = !s || el.dataset.sev === s;
      const okC = !c || el.dataset.clip === c;
      const review = el.dataset.review || '';
      const okR = !r || (r === 'unreviewed' ? review === '' : review === r);
      const show = okV && okS && okC && okR;
      el.classList.toggle('hidden', !show);
      if (show) shown++;
    }
    countEl.textContent = `${shown} / ${entries.length} shown`;
  }
  [fVoice, fSev, fClip, fRev].forEach(s => s.addEventListener('change', apply));

  document.addEventListener('change', e => {
    if (e.target.matches('input[type=radio][name^=rev-]')) {
      const entry = e.target.closest('.entry');
      entry.dataset.review = e.target.value;
      apply();
    }
  });

  // Keyboard: A = agree, D = disagree, S = skip — applies to first visible entry.
  document.addEventListener('keydown', e => {
    if (e.target.matches('input,select,textarea')) return;
    const map = { a: 'agree', d: 'disagree', s: 'skip' };
    const v = map[e.key.toLowerCase()];
    if (!v) return;
    const first = entries.find(el => !el.classList.contains('hidden')
                                    && !el.dataset.review);
    if (!first) return;
    const inp = first.querySelector(`input[value="${v}"]`);
    if (inp) { inp.checked = true; inp.dispatchEvent(new Event('change', {bubbles:true})); }
  });

  document.getElementById('export-btn').addEventListener('click', () => {
    const lines = ['sense_id\\tclip_type\\tvoice_id\\tvoice_name\\ttext\\tseverity\\thuman_review'];
    for (const el of entries) {
      const review = el.dataset.review || '';
      if (!review) continue;
      const sid = el.dataset.sense;
      const vid = el.dataset.vid;
      const vname = el.dataset.voice;
      const sev = el.dataset.sev;
      const clip = el.dataset.clip;
      const text = el.querySelector('.e-text').textContent;
      lines.push([sid, clip, vid, vname, text, sev, review].join('\\t'));
    }
    const blob = new Blob([lines.join('\\n')], {type: 'text/tab-separated-values'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '_drift_watchlist_human_review.tsv';
    a.click();
    URL.revokeObjectURL(url);
  });

  apply();
})();
</script>
</div></body></html>''')

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    print(f"  watchlist rows: {len(watch)}")
    print(f"  voices:         {len(voice_counts)}")
    print(f"  severities:     {dict(sev_counts.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
