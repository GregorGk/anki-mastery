#!/usr/bin/env python3
"""
Render the Whisper+espeak spike as a self-contained HTML report -- a
side-by-side audit of the two-stage pipeline.

For each of the 21 clips: embedded audio player, Stage-5 reference IPA,
what Whisper transcribed (the text), and the BP IPA espeak-ng pt-br
phonemized from it. Play the clip, compare what you hear to both the
Whisper text and the recognized IPA, and decide whether the model
judged it right. Export your judgments as TSV.

Reuses `build/phoneme_asr_spike_whisper.py`'s `run_batch()` -- one source
of truth for the recognition logic. Writes
`reports/phoneme_asr_spike_whisper.html`.

Run:
    .venv/bin/python build/phoneme_asr_spike_whisper_html.py
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phoneme_asr_spike_whisper import (  # noqa: E402
    AUDIO_CACHE,
    GROUP_DESC,
    GROUP_ORDER,
    MODEL_ID,
    group_means,
    run_batch,
)

REPORT = Path(__file__).resolve().parent.parent / "reports" / "phoneme_asr_spike_whisper.html"

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
.banner.warn{background:rgba(251,191,36,.12);border-color:rgba(251,191,36,.4);}
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
.plabel{color:var(--muted);font-size:11px;width:170px;flex-shrink:0;}
.pval{font-family:var(--mono);font-size:15px;}
.pval.ref{color:var(--text);}
.pval.asr{color:var(--warn);}
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
.ranking-box{border:2px solid var(--accent);border-radius:8px;padding:14px 18px;margin:18px 0;background:rgba(90,169,255,.06);box-shadow:0 0 0 1px rgba(90,169,255,.15) inset;}
.ranking-box h2{margin-top:0;}
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
                  'whisper_text','recognized','markers','sim','your_verdict'];
    const lines = [cols.join('\\t')];
    for (const el of cards){
      const checked = el.querySelector('input[type=radio]:checked');
      lines.push([el.dataset.sense, el.dataset.pt, el.dataset.group, el.dataset.source,
        el.dataset.filename, el.dataset.ref, el.dataset.asr, el.dataset.rec,
        el.dataset.markers, el.dataset.sim, checked ? checked.value : ''].join('\\t'));
    }
    const blob = new Blob([lines.join('\\n')], {type:'text/tab-separated-values'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'phoneme_asr_spike_whisper_review.tsv';
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
        sym = "✓ present" if ok else "✗ absent"
        marker_lis.append(f"<li class='{cls}'>{esc(label)} <span class='sym'>{sym}</span></li>")
    en = f"<span class='en'>— {esc(r['en'])}</span>" if r["en"] else ""
    rec_display = esc(r["recognized"]) or "<em>(empty)</em>"
    asr_display = esc(r["whisper_text"]) or "<em>(empty)</em>"
    uri = audio_data_uri(AUDIO_CACHE / r["filename"])
    rev_name = f"rev-{attr(r['filename'])}"
    return (
        f"<div class='card' data-group='{attr(r['group'])}' data-review=''"
        f" data-sense='{attr(r['sense_id'])}' data-pt='{attr(r['pt'])}'"
        f" data-source='{attr(r['source'])}' data-filename='{attr(r['filename'])}'"
        f" data-ref='{attr(r['ref_ipa'])}' data-asr='{attr(r['whisper_text'])}'"
        f" data-rec='{attr(r['recognized'])}'"
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
        f"<div class='prow'><span class='plabel'>Whisper heard (text)</span>"
        f"<span class='pval asr'>{asr_display}</span></div>"
        f"<div class='prow'><span class='plabel'>espeak pt-br IPA</span>"
        f"<span class='pval rec'>{rec_display}</span></div>"
        f"</div>"
        f"<ul class='markers'>{''.join(marker_lis)}</ul>"
        f"<div class='sim'>phone-string similarity <b>{r['sim']:.2f}</b></div>"
        f"</div>"
        f"<div class='card-right'>"
        f"<audio controls preload='none' src='{uri}'></audio>"
        f"<div class='review'>"
        f"<span class='rlabel'>play it — did the pipeline judge this clip right?</span>"
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
        print("  .venv/bin/pip install torch transformers phonemizer", file=sys.stderr)
        return 1

    means = group_means(results)
    good = means.get("legacy_human_OK", {}).get("markers")
    legbad = means.get("legacy_human_BAD", {}).get("markers")
    v3bad = means.get("v3_gemini_BAD", {}).get("markers")
    gap = (good - legbad) if (good is not None and legbad is not None) else None

    p: list[str] = []
    p.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    p.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    p.append("<title>Phoneme-ASR spike v2 — Whisper + espeak BP</title>")
    p.append(f"<style>{CSS}</style></head><body><div class='wrap'>")

    p.append("<h1>Phoneme-ASR spike v2 — Whisper + espeak BP G2P</h1>")
    p.append(
        f"<p class='sub'>"
        f"ASR <code>{esc(MODEL_ID)}</code> → text → "
        f"<code>phonemizer/espeak-ng pt-br</code> → BP IPA · {len(results)} word clips · "
        "audio players embedded base64.</p>"
    )

    # ---- top-line verdict banner ----
    if good is not None:
        if good >= 0.85:
            tone = "good"
            verdict = "TRANSCRIPTION QUALITY OK"
            desc = (
                f"GOOD clips reproduce <b>{good:.0%}</b> of their BP markers. "
                "But Whisper is word-aware — on a mispronounced clip it can "
                "<i>rescue</i> the intended word and produce canonical IPA "
                "anyway, masking the defect. Use as a transcription-quality "
                "check, not a true acoustic gate."
            )
        elif good >= 0.70:
            tone = "warn"
            verdict = "PARTIAL"
            desc = (
                f"GOOD clips: <b>{good:.0%}</b> BP markers reproduced. Some "
                "espeak-normalization edge cases remain, or Whisper struggled "
                "with a small minority of clips."
            )
        else:
            tone = "bad"
            verdict = "WEAK"
            desc = (
                f"Only <b>{good:.0%}</b> on confirmed-GOOD clips. "
                "Either Whisper is failing to transcribe BP cleanly or the "
                "espeak normalization needs more work."
            )
        p.append(f"<div class='banner {tone}'><b>{verdict}.</b> {desc}</div>")

    # ---- how to read ----
    p.append("<details class='howto' open>")
    p.append("<summary>How this differs from spike v1 (wav2vec2-espeak)</summary>")
    p.append("<div class='howto-body'>")
    p.append(
        "<p><b>v1 was acoustic.</b> The wav2vec2-espeak model emitted IPA "
        "directly from raw audio — what the model literally <i>heard</i>. "
        "It was EP-biased: even on confirmed-GOOD BP clips it stripped the "
        "BP palatalization (<code>dʒ</code>→<code>d</code>, <code>tʃ</code>"
        "→<code>t</code>) and the final-i raising (<code>i</code>→<code>ɨ</code>). "
        "Useful in principle as a defect detector, but unreliable enough on "
        "good clips that you can't separate signal from noise.</p>"
    )
    p.append(
        "<p><b>v2 is two-stage.</b> Whisper-large-v3-turbo transcribes the "
        "audio to BP text — this is the strongest open BP ASR available. "
        "Then espeak-ng's pt-br voice phonemizes that text to IPA. The IPA "
        "now comes from a deterministic rule-based BP phonemizer, not a "
        "potentially-biased acoustic model — so on good audio the output "
        "matches the Stage-5 reference almost exactly.</p>"
    )
    p.append("<h4>The trade-off</h4>")
    p.append(
        "<div class='warn-text'>Whisper hears words, not phonemes. On a "
        "<i>mispronounced</i> clip Whisper can still recognize the intended "
        "word — and then espeak emits the textbook BP IPA, which matches the "
        "reference, which makes the clip look fine even though it isn't. "
        "This pipeline can only flag clips that confuse Whisper enough to "
        "produce wrong text (e.g. <code>sede</code> → <code>'Cd'</code>, "
        "<code>de</code> → <code>'Dio'</code>). Cleanly mispronounced clips "
        "the recognizer can still parse correctly — like a soft-EP "
        "<i>verde</i> — will pass through silently.</div>"
    )
    p.append("<h4>How to read each card</h4><ol>")
    p.append("<li><b>reference IPA</b> — the textbook BP pronunciation from Stage 5.</li>")
    p.append("<li><b>Whisper heard (text)</b> — exactly what Whisper transcribed.</li>")
    p.append(
        "<li><b>espeak pt-br IPA</b> — Whisper's text run through espeak's BP "
        "phonemizer, normalized to Stage-5 convention "
        "(<code>y</code>→<code>i</code>, <code>x</code>→<code>ʁ</code>, "
        "<code>lj</code>→<code>ʎ</code>, etc.).</li>"
    )
    p.append(
        "<li>Compare the reference IPA to the espeak IPA. On a GOOD clip "
        "they should be near-identical. Big divergence ⇒ Whisper failed to "
        "transcribe, which itself is a defect signal.</li>"
    )
    p.append("</ol>")
    p.append("</div></details>")

    # ---- scorecard (wrapped in a blue ranking-box) ----
    p.append("<div class='ranking-box'>")
    p.append("<h2>Scorecard</h2>")
    p.append(
        "<table class='scorecard'><tr><th>group</th><th>n</th>"
        "<th>BP-marker reproduction</th><th>phone similarity</th></tr>"
    )
    for group in GROUP_ORDER:
        if group not in means:
            continue
        m = means[group]
        gtone = GROUP_TONE.get(group, "")
        barw = round(m["markers"] * 130)
        p.append(
            f"<tr class='row-{gtone}'><td class='grpname'>{esc(group)}</td>"
            f"<td class='num'>{m['n']}</td>"
            f"<td class='num'>{m['markers']:.0%}"
            f"<span class='bar' style='width:{barw}px'></span></td>"
            f"<td class='num'>{m['sim']:.2f}</td></tr>"
        )
    p.append("</table>")
    if gap is not None:
        v3str = f", v3 Gemini-BAD {v3bad:.0%}" if v3bad is not None else ""
        p.append(
            f"<p class='verdict-note'>GOOD {good:.0%} vs human-BAD {legbad:.0%}, "
            f"gap {gap:+.0%}{v3str}. <b>A small gap here is the expected outcome "
            "for a transcription-quality pipeline</b> — Whisper recovers the "
            "intended word on most BAD clips too, so their IPA also matches "
            "the reference. The <i>useful</i> signal is the absolute level on "
            "GOOD clips (is the transcription faithful?) plus the cases where "
            "Whisper fails to transcribe at all (a defect signal).</p>"
        )

    p.append("</div>")  # close ranking-box

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
