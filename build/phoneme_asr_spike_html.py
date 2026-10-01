#!/usr/bin/env python3
"""
Render the phoneme-ASR spike as a self-contained HTML report -- so the
model's per-clip judgment can be checked by ear.

For each of the 21 clips: an embedded audio player (base64 -- no server, no
network needed), the Stage-5 reference IPA, the phones wav2vec2-espeak
recognized, and the BP-marker checks. Play the clip, compare what you hear to
both phone strings, and mark whether the model judged it right. The export
button dumps your judgments as TSV.

Reuses `build/phoneme_asr_spike.py`'s `run_batch()` -- one source of truth
for the recognition logic. Writes `reports/phoneme_asr_spike.html`.

Run:
    .venv/bin/python build/phoneme_asr_spike_html.py
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phoneme_asr_spike import (  # noqa: E402
    AUDIO_CACHE,
    GROUP_DESC,
    GROUP_ORDER,
    MODEL_ID,
    group_means,
    run_batch,
)

REPORT = Path(__file__).resolve().parent.parent / "reports" / "phoneme_asr_spike.html"

FEATURE_LABELS = {
    "dʒ": 'dʒ palatalization — d before i/e → "dji"',
    "tʃ": 'tʃ palatalization — t before i/e → "tchi"',
    "final-i": "final-i raising — unstressed final -e → -i",
}
GROUP_TONE = {
    "legacy_human_OK": "good",
    "legacy_human_BAD": "bad",
    "v3_gemini_BAD": "bad",
    "v3_contested": "warn",
}
SOURCE_TONE = {
    "human:OK": "good",
    "human:MISPRONOUNCED": "bad",
    "gemini:non_bp": "warn",
    "gemini:bp_ok/user:bad": "warn",
}

CSS = """
:root{
  --bg:#0f1115;--panel:#181b22;--panel-2:#20242d;--border:#2c3140;
  --text:#e6e9ef;--muted:#9098a6;--accent:#5aa9ff;
  --good:#4ade80;--bad:#ff6b6b;--warn:#fbbf24;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box;}
body{background:var(--bg);color:var(--text);margin:0;padding:24px;
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
.wrap{max-width:1100px;margin:0 auto;}
h1{font-size:22px;margin:0 0 6px;}
h2{font-size:16px;margin:26px 0 10px;}
.sub{color:var(--muted);margin:0 0 16px;}
code{background:var(--panel-2);padding:1px 5px;border-radius:3px;
  font-family:var(--mono);font-size:12px;}
.banner{border-radius:8px;padding:12px 16px;margin:0 0 18px;border:1px solid var(--border);}
.banner.bad{background:rgba(255,107,107,.12);border-color:rgba(255,107,107,.4);}
.banner.good{background:rgba(74,222,128,.12);border-color:rgba(74,222,128,.4);}
details.howto{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:12px 16px;margin:0 0 18px;}
details.howto summary{cursor:pointer;font-weight:600;color:var(--accent);}
.howto-body{margin-top:10px;}
.howto-body h4{margin:14px 0 4px;color:var(--muted);text-transform:uppercase;
  font-size:11px;letter-spacing:.5px;}
.howto-body ul,.howto-body ol{margin:4px 0;padding-left:20px;}
.howto-body li{margin:3px 0;}
.warn-text{background:rgba(251,191,36,.1);border-left:3px solid var(--warn);
  padding:8px 12px;border-radius:0 4px 4px 0;margin-top:12px;}
table.scorecard{border-collapse:collapse;width:100%;font-size:13px;}
table.scorecard th,table.scorecard td{padding:6px 10px;text-align:left;
  border-bottom:1px solid var(--border);}
table.scorecard th{color:var(--muted);font-weight:500;}
table.scorecard td.num{text-align:right;font-family:var(--mono);}
.bar{display:inline-block;height:9px;background:var(--accent);border-radius:2px;
  vertical-align:middle;margin-left:8px;}
tr.row-good td.grpname{color:var(--good);}
tr.row-bad td.grpname{color:var(--bad);}
tr.row-warn td.grpname{color:var(--warn);}
.verdict-note{color:var(--muted);font-size:13px;margin:8px 0 0;}
.toolbar{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;
  flex-wrap:wrap;background:var(--panel);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;margin:18px 0;}
.toolbar label{color:var(--muted);font-size:12px;}
.toolbar select{background:var(--panel-2);color:var(--text);border:1px solid var(--border);
  border-radius:4px;padding:5px 8px;font-size:13px;}
.toolbar .grow{flex:1;}
#count{color:var(--muted);font-size:12px;}
button{background:var(--accent);color:#fff;border:0;padding:7px 14px;border-radius:5px;
  cursor:pointer;font-weight:600;font-size:13px;}
button:hover{filter:brightness(1.1);}
.group-section.hidden,.card.hidden{display:none;}
.grp{font-size:15px;margin:22px 0 2px;}
.grp.good{color:var(--good);}
.grp.bad{color:var(--bad);}
.grp.warn{color:var(--warn);}
.grp-desc{color:var(--muted);font-size:12px;margin:0 0 10px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:12px 16px;margin-bottom:10px;}
.card-head{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px;}
.card-head .pt{font-size:18px;font-weight:600;}
.card-head .id{font-family:var(--mono);font-size:12px;color:var(--muted);}
.card-head .en{color:var(--muted);font-size:13px;}
.card-head .filename{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--muted);}
.badge{font-size:11px;padding:2px 8px;border-radius:3px;font-weight:600;}
.badge.good{background:rgba(74,222,128,.18);color:var(--good);}
.badge.bad{background:rgba(255,107,107,.18);color:var(--bad);}
.badge.warn{background:rgba(251,191,36,.18);color:var(--warn);}
.badge.score{background:var(--panel-2);color:var(--muted);}
.card-body{display:grid;grid-template-columns:1fr 300px;gap:16px;}
.phones{background:var(--panel-2);border-radius:6px;padding:8px 12px;}
.prow{display:flex;gap:10px;align-items:baseline;margin:3px 0;}
.plabel{color:var(--muted);font-size:11px;width:150px;flex-shrink:0;}
.pval{font-family:var(--mono);font-size:15px;}
.pval.ref{color:var(--text);}
.pval.rec{color:var(--accent);}
ul.markers{list-style:none;margin:10px 0 0;padding:0;}
ul.markers li{padding:3px 0;font-size:13px;}
ul.markers li.ok{color:var(--good);}
ul.markers li.miss{color:var(--bad);}
ul.markers .sym{font-weight:600;}
.sim{margin-top:8px;color:var(--muted);font-size:12px;}
.card-right{display:flex;flex-direction:column;gap:8px;}
.card-right audio{width:100%;}
.review{background:var(--panel-2);border-radius:6px;padding:8px 10px;}
.rlabel{color:var(--muted);font-size:12px;display:block;margin-bottom:6px;}
.rbtns{display:flex;gap:4px;}
.rbtns label{flex:1;text-align:center;font-size:12px;padding:6px 0;border-radius:4px;
  border:1px solid var(--border);background:var(--bg);cursor:pointer;}
.rbtns input{display:none;}
.rbtns label:hover{border-color:var(--accent);}
.rbtns .r-yes:has(input:checked){background:rgba(74,222,128,.22);border-color:var(--good);color:#fff;}
.rbtns .r-no:has(input:checked){background:rgba(255,107,107,.22);border-color:var(--bad);color:#fff;}
.rbtns .r-skip:has(input:checked){background:var(--panel);border-color:var(--muted);color:#fff;}
@media(max-width:720px){.card-body{grid-template-columns:1fr;}}
"""

JS = """
(() => {
  const cards = Array.from(document.querySelectorAll('.card'));
  const sections = Array.from(document.querySelectorAll('.group-section'));
  const fGroup = document.getElementById('f-group');
  const fReview = document.getElementById('f-review');
  const countEl = document.getElementById('count');
  function apply(){
    const g = fGroup.value, r = fReview.value;
    let reviewed = 0;
    for (const el of cards){
      if (el.dataset.review) reviewed++;
      const rev = el.dataset.review || '';
      const okR = !r || (r === 'unreviewed' ? rev === '' : rev === r);
      el.classList.toggle('hidden', !okR);
    }
    for (const sec of sections){
      const okG = !g || sec.dataset.group === g;
      const vis = sec.querySelectorAll('.card:not(.hidden)').length;
      sec.classList.toggle('hidden', !okG || vis === 0);
    }
    const shown = document.querySelectorAll('.group-section:not(.hidden) .card:not(.hidden)').length;
    countEl.textContent = reviewed + '/' + cards.length + ' reviewed \\u00b7 ' + shown + ' shown';
  }
  fGroup.addEventListener('change', apply);
  fReview.addEventListener('change', apply);
  document.addEventListener('change', e => {
    if (e.target.matches('input[type=radio]')){
      e.target.closest('.card').dataset.review = e.target.value;
      apply();
    }
  });
  document.getElementById('export-btn').addEventListener('click', () => {
    const cols = ['sense_id','pt','group','source','filename','ref_ipa',
                  'recognized','markers','sim','your_verdict'];
    const lines = [cols.join('\\t')];
    for (const el of cards){
      const checked = el.querySelector('input[type=radio]:checked');
      lines.push([el.dataset.sense, el.dataset.pt, el.dataset.group, el.dataset.source,
        el.dataset.filename, el.dataset.ref, el.dataset.rec, el.dataset.markers,
        el.dataset.sim, checked ? checked.value : ''].join('\\t'));
    }
    const blob = new Blob([lines.join('\\n')], {type:'text/tab-separated-values'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'phoneme_asr_spike_review.tsv';
    a.click();
    URL.revokeObjectURL(a.href);
  });
  apply();
})();
"""


def esc(s: object) -> str:
    return html.escape(str(s))


def attr(s: object) -> str:
    return html.escape(str(s), quote=True)


def audio_data_uri(path: Path) -> str:
    return "data:audio/mpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def card_html(r: dict) -> str:
    tone = SOURCE_TONE.get(r["source"], "score")
    n_ok = sum(1 for _, ok in r["feats"] if ok)
    n_tot = len(r["feats"])
    marker_lis = []
    for name, ok in r["feats"]:
        label = FEATURE_LABELS.get(name, name)
        cls = "ok" if ok else "miss"
        sym = "✓ heard" if ok else "✗ missed"
        marker_lis.append(f"<li class='{cls}'>{esc(label)} <span class='sym'>{sym}</span></li>")
    en = f"<span class='en'>— {esc(r['en'])}</span>" if r["en"] else ""
    rec_display = esc(r["recognized"]) or "<em>(nothing recognized)</em>"
    uri = audio_data_uri(AUDIO_CACHE / r["filename"])
    rev_name = f"rev-{attr(r['filename'])}"
    return (
        f"<div class='card' data-group='{attr(r['group'])}' data-review=''"
        f" data-sense='{attr(r['sense_id'])}' data-pt='{attr(r['pt'])}'"
        f" data-source='{attr(r['source'])}' data-filename='{attr(r['filename'])}'"
        f" data-ref='{attr(r['ref_ipa'])}' data-rec='{attr(r['recognized'])}'"
        f" data-markers='{n_ok}/{n_tot}' data-sim='{r['sim']:.2f}'>"
        f"<div class='card-head'>"
        f"<span class='pt'>{esc(r['pt'])}</span>"
        f"<span class='id'>{esc(r['sense_id'])}</span>{en}"
        f"<span class='badge {tone}'>{esc(r['source'])}</span>"
        f"<span class='badge score'>markers {n_ok}/{n_tot}</span>"
        f"<span class='filename'>{esc(r['filename'])}</span>"
        f"</div>"
        f"<div class='card-body'>"
        f"<div class='card-left'>"
        f"<div class='phones'>"
        f"<div class='prow'><span class='plabel'>reference IPA (Stage 5)</span>"
        f"<span class='pval ref'>{esc(r['ref_ipa'])}</span></div>"
        f"<div class='prow'><span class='plabel'>model heard</span>"
        f"<span class='pval rec'>{rec_display}</span></div>"
        f"</div>"
        f"<ul class='markers'>{''.join(marker_lis)}</ul>"
        f"<div class='sim'>phone-string similarity <b>{r['sim']:.2f}</b></div>"
        f"</div>"
        f"<div class='card-right'>"
        f"<audio controls preload='none' src='{uri}'></audio>"
        f"<div class='review'>"
        f"<span class='rlabel'>play it — did the model judge this clip right?</span>"
        f"<div class='rbtns'>"
        f"<label class='r-yes'><input type='radio' name='{rev_name}' value='model_right'>"
        f"<span>model right</span></label>"
        f"<label class='r-no'><input type='radio' name='{rev_name}' value='model_wrong'>"
        f"<span>model wrong</span></label>"
        f"<label class='r-skip'><input type='radio' name='{rev_name}' value='unsure'>"
        f"<span>unsure</span></label>"
        f"</div></div></div></div></div>"
    )


def main() -> int:
    try:
        results = run_batch(verbose=True)
    except ImportError as exc:  # pragma: no cover - spike
        print(f"missing dependency: {exc}", file=sys.stderr)
        print("  .venv/bin/pip install torch transformers", file=sys.stderr)
        return 1

    means = group_means(results)
    good = means.get("legacy_human_OK", {}).get("markers")
    legbad = means.get("legacy_human_BAD", {}).get("markers")
    gap = (good - legbad) if (good is not None and legbad is not None) else None

    p: list[str] = []
    p.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    p.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    p.append("<title>Phoneme-ASR spike — wav2vec2-espeak vs Brazilian Portuguese</title>")
    p.append(f"<style>{CSS}</style></head><body><div class='wrap'>")

    p.append("<h1>Phoneme-ASR spike — does wav2vec2-espeak hear Brazilian Portuguese?</h1>")
    p.append(
        f"<p class='sub'>Model <code>{esc(MODEL_ID)}</code> · {len(results)} word clips · "
        "audio → IPA phones, CPU inference, no API calls. "
        "Every audio player below is embedded — no server needed.</p>"
    )

    if gap is not None:
        if gap < 0.15:
            verdict_word = "NO DISCRIMINATION"
        elif good >= 0.70 and gap >= 0.25:
            verdict_word = "DISCRIMINATES"
        else:
            verdict_word = "WEAK / AMBIGUOUS"
        p.append(
            f"<div class='banner bad'><b>{verdict_word}.</b> "
            f"Human-confirmed-GOOD clips reproduce {good:.0%} of their BP markers; "
            f"human-confirmed-BAD clips {legbad:.0%} — a {gap:+.0%} gap. "
            "The model's marker judgments do not separate good Brazilian Portuguese "
            "from bad on same-era audio. This page lets you hear that yourself.</div>"
        )

    # ---- how to read ----
    p.append("<details class='howto' open>")
    p.append("<summary>How to read this — and how to verify it yourself</summary>")
    p.append("<div class='howto-body'>")
    p.append(
        "<p><b>What this is.</b> A spike testing whether an open-weights phoneme "
        "recognizer can be an automated check that a TTS clip is really Brazilian "
        "Portuguese. The model takes audio and emits a string of IPA phones — what it "
        "<i>heard</i>. We compare that to the Stage-5 reference IPA (the textbook BP "
        "pronunciation) and check three BP-defining features.</p>"
    )
    p.append("<h4>The three BP markers we check</h4><ul>")
    p.append(
        "<li><b>dʒ palatalization</b> — in BP, <code>d</code> before an i/e sound "
        "becomes <code>[dʒ]</code> (\"dji\"): <i>verde</i> → \"ver-<b>dji</b>\". "
        "European Portuguese keeps a plain <code>[d]</code>.</li>"
    )
    p.append(
        "<li><b>tʃ palatalization</b> — likewise <code>t</code> → <code>[tʃ]</code> "
        "(\"tchi\"): <i>noite</i> → \"noi-<b>tchi</b>\".</li>"
    )
    p.append(
        "<li><b>final-i raising</b> — an unstressed final <code>-e</code> is raised to "
        "<code>[i]</code> in BP: <i>tarde</i> ends in \"<b>i</b>\", not a clean "
        "\"<b>e</b>\" or the reduced central <code>[ɨ]</code> that EP uses.</li>"
    )
    p.append("</ul>")
    p.append("<h4>Other things to listen for</h4><ul>")
    p.append(
        "<li>Unstressed <code>/e/</code> mid-word — BP keeps a clear "
        "<code>[e]</code>/<code>[i]</code>; EP reduces to <code>[ɨ]</code> or drops it.</li>"
    )
    p.append(
        "<li>Coda <code>/r/</code> — BP often retracted / velar / uvular; EP a clean "
        "alveolar tap or trill. This varies a lot even within BP — don't over-weight it.</li>"
    )
    p.append(
        "<li><code>/v/</code> should be a firm labiodental <code>[v]</code>; a soft "
        "Spanish-like <code>[β]</code> signals foreign training data leaking in.</li>"
    )
    p.append("</ul>")
    p.append("<h4>How to verify a clip</h4><ol>")
    p.append("<li>Read the <b>reference IPA</b> — the textbook BP pronunciation.</li>")
    p.append("<li>Play the <b>audio</b> and listen for the markers above.</li>")
    p.append("<li>Read what the <b>model heard</b> and its ✓ / ✗ marker calls.</li>")
    p.append(
        "<li>Decide: did the model judge this clip correctly? Mark <b>model right / "
        "model wrong / unsure</b>, then export your judgments with the button below.</li>"
    )
    p.append("</ol>")
    p.append(
        "<div class='warn-text'><b>Important caveat.</b> The model reports \"missing "
        "marker\" (✗) on confirmed-<i>good</i> clips too — look at the "
        "<code>legacy_human_OK</code> group, where most clips still carry ✗s. So a ✗ is "
        "<i>not</i> proof a clip is bad. That inconsistency is exactly why the verdict "
        "is \"no discrimination\" — and why the n=2 verde result looked convincing but "
        "did not hold up.</div>"
    )
    p.append("</div></details>")

    # ---- scorecard ----
    p.append("<h2>Scorecard</h2>")
    p.append(
        "<table class='scorecard'><tr><th>group</th><th>n</th>"
        "<th>BP-marker reproduction</th><th>phone similarity</th></tr>"
    )
    for group in GROUP_ORDER:
        if group not in means:
            continue
        m = means[group]
        tone = GROUP_TONE.get(group, "")
        barw = round(m["markers"] * 130)
        p.append(
            f"<tr class='row-{tone}'><td class='grpname'>{esc(group)}</td>"
            f"<td class='num'>{m['n']}</td>"
            f"<td class='num'>{m['markers']:.0%}<span class='bar' style='width:{barw}px'></span></td>"
            f"<td class='num'>{m['sim']:.2f}</td></tr>"
        )
    p.append("</table>")
    if gap is not None:
        p.append(
            f"<p class='verdict-note'>Controlled comparison — same TTS era, same label "
            f"source (human), only the label differs: GOOD {good:.0%} vs BAD {legbad:.0%}, "
            f"gap {gap:+.0%}. The <code>v3_*</code> groups are cross-era (different TTS "
            "model) and read with caution. <code>v3_contested</code> is verde v2 — the "
            "clip Gemini passed but the user rejects.</p>"
        )

    # ---- toolbar ----
    group_opts = "".join(f"<option value='{attr(g)}'>{esc(g)}</option>" for g in GROUP_ORDER)
    p.append("<div class='toolbar'>")
    p.append(
        f"<label>group <select id='f-group'><option value=''>all</option>"
        f"{group_opts}</select></label>"
    )
    p.append(
        "<label>review <select id='f-review'><option value=''>all</option>"
        "<option value='unreviewed'>unreviewed</option>"
        "<option value='model_right'>model right</option>"
        "<option value='model_wrong'>model wrong</option>"
        "<option value='unsure'>unsure</option></select></label>"
    )
    p.append("<span class='grow'></span><span id='count'></span>")
    p.append("<button id='export-btn'>Export my judgments (TSV)</button>")
    p.append("</div>")

    # ---- per-clip cards, grouped ----
    p.append("<div id='cards'>")
    for group in GROUP_ORDER:
        grp = [r for r in results if r["group"] == group]
        if not grp:
            continue
        tone = GROUP_TONE.get(group, "")
        p.append(f"<section class='group-section' data-group='{attr(group)}'>")
        p.append(f"<h2 class='grp {tone}'>{esc(group)}</h2>")
        p.append(f"<p class='grp-desc'>{esc(GROUP_DESC[group])}</p>")
        for r in grp:
            p.append(card_html(r))
        p.append("</section>")
    p.append("</div>")

    p.append(f"<script>{JS}</script>")
    p.append("</div></body></html>")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(p), encoding="utf-8")
    size_kb = REPORT.stat().st_size / 1024
    print(f"wrote {REPORT}  ({len(results)} clips, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
