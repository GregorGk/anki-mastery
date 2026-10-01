#!/usr/bin/env python3
"""
Build the 300-row interactive borderline review HTML.

Goal
----
Front-load human review of the clips most likely to be defective, with
three independent BP IPA sources side-by-side so the human can spot
disagreement at a glance. Their verdicts + free-text notes are exported
as TSV -- the input to the MFA judge phase that comes next.

Selection priority (high → low; first match wins; dedup by sense_id;
stop at 300):

  1. confirmed-bad        `_audio_mispronunciation_confirmed.tsv`    (55)
  2. unclear              `_audio_calibration_unclear.tsv`           (48)
  3. user-reported        `_audio_user_reported_failures.tsv` (pt-based, mapped to sense_ids via 06-final)
  4. spike-bad            hard-coded sense_ids from phoneme_asr_spike (12)
  5. v4-flag              `_audio_v4_flag_list.tsv` rows with flag=FLAG
  6. calibration-bad      `_audio_calibration_labels.tsv` MISPRONOUNCED (53)
  7. asr-fail-v5          `_audio_human_review.tsv` notes contain v5
  8. asr-fail-v4          `_audio_human_review.tsv` notes contain v4
  9. asr-fail-v3          `_audio_human_review.tsv` notes contain v3
 10. asr-fail             `_audio_human_review.tsv` remainder

IPA sources per word (3 cols):
  - Stage 5    : `ipa_word` from `data/06-final.tsv` (the current LLM output)
  - MFA dict   : `portuguese_brazil_mfa.dict` lookup, all variants joined
                 with " / " (multiple variants = legitimate regional spread)
  - espeak     : `phonemizer / espeak-ng pt-br`, normalized to Stage-5
                 convention (y→i, x→ʁ, lj→ʎ, nj→ɲ, strip ZWJ)

IPA per example sentence (2 cols):
  - Stage 5    : `ipa_example`
  - espeak     : phonemizer espeak-ng on the sentence

Outputs:
  - `data/_borderline_review_seed.tsv` : 300 senses + priority source label
                                         + audit trail (idempotent)
  - `reports/borderline_review.html`   : interactive review HTML

Run:
  .venv/bin/python build/borderline_review.py
"""

from __future__ import annotations

import base64
import csv
import html
import os
import sys
import warnings
from pathlib import Path

# espeak-ng dylib lives in homebrew on this machine; phonemizer needs the
# explicit path because its default search doesn't cover /opt/homebrew.
os.environ.setdefault(
    "PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib"
)

ROOT = Path(__file__).resolve().parent.parent
FINAL_TSV = ROOT / "data" / "06-final.tsv"
SEED_TSV = ROOT / "data" / "_borderline_review_seed.tsv"
REPORT_HTML = ROOT / "reports" / "borderline_review.html"
MFA_DICT = ROOT / "data" / "_mfa_dict" / "portuguese_brazil_mfa.dict"

# Spike known-bads (from `build/phoneme_asr_spike.py BATCH`, BAD side only).
SPIKE_BADS: list[str] = [
    # legacy_human_BAD
    "0937.00.01", "3009.00.01", "1175.00.01", "3536.00.01", "3013.00.01", "4064.00.01",
    # v3_gemini_BAD (verde, de, partido, sítio, doze, garantir)
    "0534.00.01", "0002.00.01", "0304.00.01", "0913.00.01", "1454.00.01", "0580.00.01",
]

# Priority pools. Walked top→bottom; first occurrence of a sense_id wins.
# `tsv_filename` is relative to `data/`. `pred` filters the rows of that
# TSV before contributing sense_ids. `spike-bad` is a special case that
# pulls from the hardcoded SPIKE_BADS list, not a TSV.
PRIORITY_POOLS = [
    ("confirmed-bad",   "_audio_mispronunciation_confirmed.tsv", lambda r: True),
    ("unclear",         "_audio_calibration_unclear.tsv",        lambda r: True),
    ("user-reported",   "_audio_user_reported_failures.tsv",     lambda r: True),
    ("spike-bad",       None,                                    None),
    ("v4-flag",         "_audio_v4_flag_list.tsv",               lambda r: r.get("flag") == "FLAG"),
    ("calibration-bad", "_audio_calibration_labels.tsv",         lambda r: r.get("label") == "MISPRONOUNCED"),
    ("asr-fail-v5",     "_audio_human_review.tsv",               lambda r: "v5" in (r.get("notes") or "")),
    ("asr-fail-v4",     "_audio_human_review.tsv",               lambda r: "v4" in (r.get("notes") or "")),
    ("asr-fail-v3",     "_audio_human_review.tsv",               lambda r: "v3" in (r.get("notes") or "")),
    ("asr-fail",        "_audio_human_review.tsv",               lambda r: True),
]

TARGET_N = 300


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, dialect="excel-tab"))


def _pt_to_sense_ids() -> dict[str, list[str]]:
    """Build {pt: [sense_id, ...]} from 06-final for pt-based pools.

    Used to resolve `_audio_user_reported_failures.tsv`, which is keyed
    by pt only (a user reported "the audio for X sounds wrong" before
    knowing which sense_id was affected). Loaded lazily so the main flow
    isn't taxed when no pt-based pool is configured.
    """
    out: dict[str, list[str]] = {}
    with open(FINAL_TSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, dialect="excel-tab"):
            out.setdefault(r["pt"], []).append(r["sense_id"])
    return out


def assemble_seed_list() -> list[dict]:
    """Walk PRIORITY_POOLS in order, dedup by sense_id, cap at TARGET_N.

    Returns list of {sense_id, source, priority, priority_idx, pool_row}.
    Some pools are special: `spike-bad` is a hard-coded list; pools whose
    TSV is keyed by `pt` (not `sense_id`) get their pts mapped through
    `_pt_to_sense_ids()` before being walked.
    """
    seen: dict[str, dict] = {}
    counts = {label: 0 for label, *_ in PRIORITY_POOLS}
    pt_to_sids: dict[str, list[str]] | None = None  # lazy

    def expand_rows(label: str, tsv_filename: str | None, pred) -> list[dict]:
        nonlocal pt_to_sids
        if label == "spike-bad":
            return [{"sense_id": s} for s in SPIKE_BADS]
        path = ROOT / "data" / tsv_filename
        raw = [r for r in read_tsv(path) if pred(r)]
        # If the TSV doesn't carry sense_id, fan its `pt` out to every
        # matching sense.
        if raw and "sense_id" not in raw[0]:
            if pt_to_sids is None:
                pt_to_sids = _pt_to_sense_ids()
            expanded: list[dict] = []
            for r in raw:
                for sid in pt_to_sids.get(r.get("pt", ""), []):
                    expanded.append({**r, "sense_id": sid})
            return expanded
        return raw

    for priority_idx, (label, tsv_filename, pred) in enumerate(PRIORITY_POOLS):
        rows = expand_rows(label, tsv_filename, pred)
        for r in rows:
            sid = r["sense_id"]
            if sid in seen:
                continue
            if len(seen) >= TARGET_N:
                break
            seen[sid] = dict(sense_id=sid, source=label, priority=label,
                             priority_idx=priority_idx, pool_row=r)
            counts[label] += 1
        if len(seen) >= TARGET_N:
            break
    # log composition
    print("Selection composition (after dedup):")
    for label in counts:
        print(f"  {label:<18} {counts[label]:>3}")
    print(f"  {'TOTAL':<18} {len(seen):>3}\n")
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# IPA sources
# ─────────────────────────────────────────────────────────────────────────────


def load_mfa_dict() -> dict[str, list[str]]:
    """Parse `portuguese_brazil_mfa.dict` -> {word: [variant_ipa, ...]}.

    Each line is `word\\tphone phone phone ...` (space-separated phones).
    A word can appear multiple times (alternate pronunciations) -- collect
    all variants into a list, preserving file order.
    """
    out: dict[str, list[str]] = {}
    with open(MFA_DICT, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            word, ipa = line.split("\t", 1)
            word = word.strip().lower()
            ipa = ipa.strip()
            if word.startswith("[") or word.startswith("<"):
                continue
            out.setdefault(word, []).append(ipa)
    return out


def mfa_lookup(word: str, mfa: dict[str, list[str]]) -> str:
    """Look up a single PT lemma in the MFA dict, joining variants with ` / `."""
    if not word:
        return ""
    variants = mfa.get(word.strip().lower())
    return " / ".join(variants) if variants else ""


def espeak_normalize(s: str) -> str:
    """Map espeak pt-br IPA into Stage-5 IPA convention.

    Same mapping as build/phoneme_asr_spike_whisper.py:espeak_normalize --
    y→i, x→ʁ, lj→ʎ, nj→ɲ, strip ZWJ + trailing ʊ + ŋ-after-nasal.
    """
    s = s.replace("‍", "")
    s = s.replace("y", "i")
    s = s.replace("x", "ʁ")
    s = s.replace("lj", "ʎ").replace("nj", "ɲ")
    if s.endswith("ʊ"):
        s = s[:-1] + "u"
    out: list[str] = []
    nasal_vowels = {"ɐ̃", "ẽ", "ĩ", "õ", "ũ"}
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "ŋ" and out and (
            out[-1] in nasal_vowels
            or (len(out) >= 2 and out[-2] + out[-1] in nasal_vowels)
            or (i > 0 and s[i - 1] == "̃")
        ):
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def espeak_batch(texts: list[str]) -> list[str]:
    """Phonemize a batch of texts (words or sentences) with espeak pt-br.

    Single-call form is much faster than per-string (~50ms overhead per
    call). Output is normalized to Stage-5 convention.
    """
    from phonemizer import phonemize
    if not texts:
        return []
    cleaned = [(t or "").strip() for t in texts]
    # phonemizer chokes on empty strings -- substitute a sentinel
    sentinel = "_"
    feed = [t if t else sentinel for t in cleaned]
    raw = phonemize(feed, language="pt-br", backend="espeak",
                    strip=True, with_stress=True, njobs=1)
    return [espeak_normalize(r) if cleaned[i] else "" for i, r in enumerate(raw)]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset assembly
# ─────────────────────────────────────────────────────────────────────────────


def assemble_dataset() -> list[dict]:
    warnings.filterwarnings("ignore")
    seed = assemble_seed_list()
    if not seed:
        print("ERROR: no senses selected", file=sys.stderr)
        sys.exit(1)

    print("Loading 06-final.tsv ...")
    with open(FINAL_TSV, encoding="utf-8") as fh:
        final = {r["sense_id"]: r for r in csv.DictReader(fh, dialect="excel-tab")}

    print(f"Loading MFA dict ({MFA_DICT}) ...")
    mfa = load_mfa_dict()
    print(f"  {len(mfa):,} unique lemmas in dict")

    # Look up everything from 06-final, drop senses with no audio_word.
    enriched: list[dict] = []
    missing_audio = 0
    missing_final = 0
    for s in seed:
        sid = s["sense_id"]
        r = final.get(sid)
        if not r:
            missing_final += 1
            continue
        if not r.get("audio_word"):
            missing_audio += 1
            continue
        enriched.append(dict(
            sense_id=sid,
            source=s["source"],
            priority_idx=s["priority_idx"],
            pt=r.get("pt", ""),
            pt_display=r.get("pt_display_safe") or r.get("pt_display") or r.get("pt", ""),
            en_primary=r.get("en_primary", ""),
            ipa_word_stage5=r.get("ipa_word", ""),
            ipa_word_mfa=mfa_lookup(r.get("pt", ""), mfa),
            example_pt=r.get("example_pt", ""),
            example_en=r.get("example_en", ""),
            ipa_example_stage5=r.get("ipa_example", ""),
            audio_word=r.get("audio_word", ""),
            audio_example=r.get("audio_example", ""),
            audio_en_example=r.get("audio_en_example", ""),
            voice_id=r.get("voice_id", ""),
            bp_validity=r.get("bp_validity", ""),
            pool_row=s["pool_row"],
        ))
    if missing_final:
        print(f"  !! {missing_final} senses not in 06-final.tsv (skipped)")
    if missing_audio:
        print(f"  !! {missing_audio} senses missing audio_word (skipped)")

    # Compute espeak IPA for words + example sentences in two batches.
    print(f"Phonemizing {len(enriched)} words + examples with espeak pt-br ...")
    words = [e["pt"] for e in enriched]
    examples = [e["example_pt"] for e in enriched]
    word_ipas = espeak_batch(words)
    example_ipas = espeak_batch(examples)
    for e, w, x in zip(enriched, word_ipas, example_ipas):
        e["ipa_word_espeak"] = w
        e["ipa_example_espeak"] = x

    # Sort by priority (most-critical first), then by sense_id within priority.
    enriched.sort(key=lambda e: (e["priority_idx"], e["sense_id"]))

    return enriched


def write_seed_tsv(rows: list[dict]) -> None:
    """Audit trail: which 300 senses were picked, why, in what order."""
    fields = ["sense_id", "pt", "en_primary", "source", "priority_idx",
              "ipa_word_stage5", "ipa_word_mfa", "ipa_word_espeak",
              "bp_validity"]
    SEED_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SEED_TSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, dialect="excel-tab",
                           extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {SEED_TSV}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────────────


SOURCE_TONE = {
    "confirmed-bad":   "bad",
    "unclear":         "warn",
    "user-reported":   "bad",
    "spike-bad":       "bad",
    "v4-flag":         "warn",
    "calibration-bad": "bad",
    "asr-fail-v5":     "warn",
    "asr-fail-v4":     "warn",
    "asr-fail-v3":     "warn",
    "asr-fail":        "muted",
}

SOURCE_DESC = {
    "confirmed-bad":   "User-confirmed mispronunciation",
    "unclear":         "Calibration label UNCLEAR (borderline by definition)",
    "user-reported":   "User-reported failure (Stage 7 listening checklist)",
    "spike-bad":       "Phoneme-ASR spike BAD subset (Stage 16.8 non_bp candidate)",
    "v4-flag":         "Stage v4 alias-respelling flag",
    "calibration-bad": "Calibration label MISPRONOUNCED",
    "asr-fail-v5":     "Still failing after regen-human-queue v5",
    "asr-fail-v4":     "Still failing after regen-human-queue v4",
    "asr-fail-v3":     "Still failing after regen-human-queue v3",
    "asr-fail":        "Twice-failed ASR roundtrip",
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
.wrap{max-width:1180px;margin:0 auto;}
h1{font-size:22px;margin:0 0 6px;}
h2{font-size:16px;margin:22px 0 10px;}
.sub{color:var(--muted);margin:0 0 16px;}
code{background:var(--panel-2);padding:1px 5px;border-radius:3px;
  font-family:var(--mono);font-size:12px;}
details.howto{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:12px 16px;margin:0 0 18px;}
details.howto summary{cursor:pointer;font-weight:600;color:var(--accent);}
.howto-body{margin-top:10px;}
.howto-body h4{margin:14px 0 4px;color:var(--muted);text-transform:uppercase;
  font-size:11px;letter-spacing:.5px;}
.howto-body ul,.howto-body ol{margin:4px 0;padding-left:20px;}
.howto-body li{margin:3px 0;}
table.legend{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;}
table.legend th,table.legend td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border);}
table.legend th{color:var(--muted);font-weight:500;}
.toolbar{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;
  flex-wrap:wrap;background:var(--panel);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;margin:18px 0;}
.toolbar label{color:var(--muted);font-size:12px;}
.toolbar select,.toolbar input{background:var(--panel-2);color:var(--text);
  border:1px solid var(--border);border-radius:4px;padding:5px 8px;font-size:13px;}
.toolbar .grow{flex:1;}
#count{color:var(--muted);font-size:12px;}
button{background:var(--accent);color:#fff;border:0;padding:7px 14px;border-radius:5px;
  cursor:pointer;font-weight:600;font-size:13px;}
button.ghost{background:var(--panel-2);color:var(--text);border:1px solid var(--border);}
button:hover{filter:brightness(1.1);}
.card.hidden{display:none;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:14px 18px;margin-bottom:12px;}
.card-head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px;}
.card-head .pt{font-size:20px;font-weight:600;}
.card-head .id{font-family:var(--mono);font-size:12px;color:var(--muted);}
.card-head .en{color:var(--muted);font-size:13px;}
.card-head .voice{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--muted);}
.badge{font-size:11px;padding:2px 8px;border-radius:3px;font-weight:600;
  text-transform:uppercase;letter-spacing:.3px;}
.badge.bad{background:rgba(255,107,107,.18);color:var(--bad);}
.badge.warn{background:rgba(251,191,36,.18);color:var(--warn);}
.badge.muted{background:var(--panel-2);color:var(--muted);}
.badge.good{background:rgba(74,222,128,.18);color:var(--good);}

.card-body{display:grid;grid-template-columns:1fr 320px;gap:18px;}
.phones{background:var(--panel-2);border-radius:6px;padding:10px 14px;}
.phones h5{margin:0 0 6px;font-size:11px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.5px;}
.prow{display:grid;grid-template-columns:130px 1fr;gap:10px;align-items:baseline;
  padding:3px 0;}
.plabel{color:var(--muted);font-size:11px;}
.pval{font-family:var(--mono);font-size:14px;word-break:break-word;}
.pval.s5{color:var(--text);}
.pval.mfa{color:var(--good);}
.pval.es{color:var(--accent);}
.pval.empty{color:var(--muted);font-style:italic;}
.ex{margin-top:10px;font-size:13px;color:var(--muted);}
.ex .pt-ex{color:var(--text);font-style:italic;}

.right{display:flex;flex-direction:column;gap:8px;}
.right audio{width:100%;height:32px;}
.audio-row{display:flex;flex-direction:column;gap:3px;}
.audio-row .alab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;}
.review{background:var(--panel-2);border-radius:6px;padding:10px 12px;margin-top:4px;}
.rlabel{color:var(--muted);font-size:12px;display:block;margin-bottom:6px;}
.rbtns{display:flex;gap:4px;}
.rbtns label{flex:1;text-align:center;font-size:12px;padding:6px 0;border-radius:4px;
  border:1px solid var(--border);background:var(--bg);cursor:pointer;}
.rbtns input{display:none;}
.rbtns label:hover{border-color:var(--accent);}
.rbtns .r-good:has(input:checked){background:rgba(74,222,128,.25);border-color:var(--good);color:#fff;}
.rbtns .r-bad:has(input:checked){background:rgba(255,107,107,.25);border-color:var(--bad);color:#fff;}
.rbtns .r-unsure:has(input:checked){background:rgba(251,191,36,.25);border-color:var(--warn);color:#fff;}
.notes{display:block;width:100%;background:var(--bg);color:var(--text);
  border:1px solid var(--border);border-radius:4px;padding:6px 8px;font-size:13px;
  font-family:inherit;resize:vertical;min-height:54px;margin-top:8px;}
.notes:focus{outline:0;border-color:var(--accent);}
@media(max-width:760px){.card-body{grid-template-columns:1fr;}}
"""

JS = r"""
(() => {
  const cards = Array.from(document.querySelectorAll('.card'));
  const fSrc = document.getElementById('f-src');
  const fRev = document.getElementById('f-rev');
  const fText = document.getElementById('f-text');
  const countEl = document.getElementById('count');

  function reviewedCount(){
    return cards.filter(c => c.dataset.review).length;
  }
  function apply(){
    const s = fSrc.value;
    const r = fRev.value;
    const t = (fText.value || '').toLowerCase();
    let shown = 0;
    for (const el of cards){
      const okS = !s || el.dataset.source === s;
      const rev = el.dataset.review || '';
      const okR = !r || (r === 'unreviewed' ? rev === '' : rev === r);
      const hay = (el.dataset.pt + ' ' + el.dataset.en + ' ' + el.dataset.sense).toLowerCase();
      const okT = !t || hay.includes(t);
      const show = okS && okR && okT;
      el.classList.toggle('hidden', !show);
      if (show) shown++;
    }
    countEl.textContent = reviewedCount() + ' / ' + cards.length + ' reviewed · ' + shown + ' shown';
  }
  fSrc.addEventListener('change', apply);
  fRev.addEventListener('change', apply);
  fText.addEventListener('input', apply);

  document.addEventListener('change', e => {
    if (e.target.matches('input[type=radio]')){
      e.target.closest('.card').dataset.review = e.target.value;
      // also restore the notes from textarea if any
      apply();
    }
  });

  // Persist verdicts + notes in localStorage so reloads don't lose work.
  const LS_KEY = 'borderline_review_state_v1';
  function saveState(){
    const out = {};
    for (const el of cards){
      const r = el.querySelector('input[type=radio]:checked');
      const n = el.querySelector('textarea.notes');
      if (r || (n && n.value)) {
        out[el.dataset.sense] = {v: r ? r.value : '', n: n ? n.value : ''};
      }
    }
    try { localStorage.setItem(LS_KEY, JSON.stringify(out)); } catch(e){}
  }
  function loadState(){
    let data = {};
    try { data = JSON.parse(localStorage.getItem(LS_KEY) || '{}'); } catch(e){}
    for (const el of cards){
      const s = data[el.dataset.sense];
      if (!s) continue;
      if (s.v){
        const inp = el.querySelector('input[type=radio][value="'+s.v+'"]');
        if (inp){ inp.checked = true; el.dataset.review = s.v; }
      }
      if (s.n){
        const ta = el.querySelector('textarea.notes');
        if (ta) ta.value = s.n;
      }
    }
  }
  document.addEventListener('change', saveState);
  document.addEventListener('input', e => { if (e.target.matches('textarea.notes')) saveState(); });

  // Keyboard: G = good, B = bad, U = unsure -- applies to first visible
  // unreviewed card, focuses notes after.
  document.addEventListener('keydown', e => {
    if (e.target.matches('input,select,textarea')) return;
    const map = {g:'good', b:'bad', u:'unsure'};
    const v = map[e.key.toLowerCase()];
    if (!v) return;
    const first = cards.find(c => !c.classList.contains('hidden') && !c.dataset.review);
    if (!first) return;
    const inp = first.querySelector('input[type=radio][value="'+v+'"]');
    if (inp){ inp.checked = true; inp.dispatchEvent(new Event('change',{bubbles:true})); }
  });

  document.getElementById('export-btn').addEventListener('click', () => {
    const cols = ['sense_id','pt','en_primary','source',
                  'ipa_word_stage5','ipa_word_mfa','ipa_word_espeak',
                  'example_pt','ipa_example_stage5','ipa_example_espeak',
                  'audio_word','audio_example','your_verdict','your_notes'];
    const lines = [cols.join('\t')];
    for (const el of cards){
      const r = el.querySelector('input[type=radio]:checked');
      const n = el.querySelector('textarea.notes');
      const row = [
        el.dataset.sense, el.dataset.pt, el.dataset.en, el.dataset.source,
        el.dataset.s5, el.dataset.mfa, el.dataset.es,
        el.dataset.ex_pt, el.dataset.ex_s5, el.dataset.ex_es,
        el.dataset.aw, el.dataset.ax,
        r ? r.value : '', n ? n.value.replace(/[\t\n\r]/g,' ') : ''
      ];
      lines.push(row.join('\t'));
    }
    const blob = new Blob([lines.join('\n')], {type:'text/tab-separated-values'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'borderline_review.tsv';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  document.getElementById('clear-btn').addEventListener('click', () => {
    if (!confirm('Clear all verdicts + notes? This will reset the local-storage cache too.')) return;
    try { localStorage.removeItem(LS_KEY); } catch(e){}
    location.reload();
  });

  loadState();
  apply();
})();
"""


def esc(s: object) -> str:
    return html.escape(str(s))


def attr(s: object) -> str:
    return html.escape(str(s), quote=True)


def render_pval(value: str, kind: str) -> str:
    if value:
        return f"<span class='pval {kind}'>{esc(value)}</span>"
    return f"<span class='pval {kind} empty'>—</span>"


def card_html(r: dict) -> str:
    tone = SOURCE_TONE.get(r["source"], "muted")
    voice = f"<span class='voice'>voice {esc(r.get('voice_id',''))}</span>" if r.get("voice_id") else ""
    bpv = r.get("bp_validity", "")
    bpv_badge = (
        f"<span class='badge {'good' if bpv == 'standard' else 'warn'}'>"
        f"{esc(bpv)}</span>"
    ) if bpv else ""

    ex_lines = ""
    if r["example_pt"]:
        ex_lines = (
            f"<div class='ex'>"
            f"<div class='pt-ex'>{esc(r['example_pt'])}</div>"
            f"<div>{esc(r['example_en'])}</div>"
            f"</div>"
        )

    audio_word = (
        f"<div class='audio-row'><span class='alab'>word</span>"
        f"<audio controls preload='none' src='{attr(r['audio_word'])}'></audio></div>"
    )
    audio_ex = (
        f"<div class='audio-row'><span class='alab'>example (PT)</span>"
        f"<audio controls preload='none' src='{attr(r['audio_example'])}'></audio></div>"
    ) if r.get("audio_example") else ""

    rev_name = f"rev-{attr(r['sense_id'])}"
    return (
        f"<div class='card' data-source='{attr(r['source'])}' data-review=''"
        f" data-sense='{attr(r['sense_id'])}' data-pt='{attr(r['pt'])}'"
        f" data-en='{attr(r['en_primary'])}'"
        f" data-s5='{attr(r['ipa_word_stage5'])}'"
        f" data-mfa='{attr(r['ipa_word_mfa'])}'"
        f" data-es='{attr(r['ipa_word_espeak'])}'"
        f" data-ex_pt='{attr(r['example_pt'])}'"
        f" data-ex_s5='{attr(r['ipa_example_stage5'])}'"
        f" data-ex_es='{attr(r['ipa_example_espeak'])}'"
        f" data-aw='{attr(r['audio_word'])}'"
        f" data-ax='{attr(r['audio_example'])}'>"
        f"<div class='card-head'>"
        f"<span class='pt'>{esc(r['pt_display'])}</span>"
        f"<span class='id'>{esc(r['sense_id'])}</span>"
        f"<span class='en'>— {esc(r['en_primary'])}</span>"
        f"<span class='badge {tone}'>{esc(r['source'])}</span>"
        f"{bpv_badge}"
        f"{voice}"
        f"</div>"
        f"<div class='card-body'>"
        f"<div>"
        f"<div class='phones'><h5>word IPA — 3 sources</h5>"
        f"<div class='prow'><span class='plabel'>Stage 5 (LLM)</span>"
        f"{render_pval(r['ipa_word_stage5'], 's5')}</div>"
        f"<div class='prow'><span class='plabel'>MFA dict</span>"
        f"{render_pval(r['ipa_word_mfa'], 'mfa')}</div>"
        f"<div class='prow'><span class='plabel'>espeak pt-br</span>"
        f"{render_pval(r['ipa_word_espeak'], 'es')}</div>"
        f"</div>"
        f"{ex_lines}"
        f"<div class='phones' style='margin-top:8px;'><h5>example IPA — 2 sources</h5>"
        f"<div class='prow'><span class='plabel'>Stage 5 (LLM)</span>"
        f"{render_pval(r['ipa_example_stage5'], 's5')}</div>"
        f"<div class='prow'><span class='plabel'>espeak pt-br</span>"
        f"{render_pval(r['ipa_example_espeak'], 'es')}</div>"
        f"</div>"
        f"</div>"
        f"<div class='right'>"
        f"{audio_word}{audio_ex}"
        f"<div class='review'>"
        f"<span class='rlabel'>your verdict (keys: G B U)</span>"
        f"<div class='rbtns'>"
        f"<label class='r-good'><input type='radio' name='{rev_name}' value='good'>"
        f"<span>good</span></label>"
        f"<label class='r-bad'><input type='radio' name='{rev_name}' value='bad'>"
        f"<span>bad</span></label>"
        f"<label class='r-unsure'><input type='radio' name='{rev_name}' value='unsure'>"
        f"<span>unsure</span></label>"
        f"</div>"
        f"<textarea class='notes' placeholder='notes (which phone? EP-ish? wrong stress? cut off? …)'></textarea>"
        f"</div></div></div></div>"
    )


def write_html(rows: list[dict]) -> None:
    # Per-source counts for the toolbar dropdown labels.
    src_counts: dict[str, int] = {}
    for r in rows:
        src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1

    src_options = "".join(
        f"<option value='{attr(s)}'>{esc(s)} ({src_counts[s]})</option>"
        for s in sorted(src_counts, key=lambda k: -src_counts[k])
    )

    legend_rows = "".join(
        f"<tr><td><span class='badge {SOURCE_TONE.get(s,'muted')}'>{esc(s)}</span></td>"
        f"<td>{esc(SOURCE_DESC.get(s,''))}</td>"
        f"<td style='text-align:right;color:var(--muted);'>{src_counts[s]}</td></tr>"
        for s in sorted(src_counts, key=lambda k: -src_counts[k])
    )

    p: list[str] = []
    p.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    p.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    p.append("<title>Borderline review — 300 BP clips × 3 IPA sources</title>")
    p.append(f"<style>{CSS}</style></head><body><div class='wrap'>")

    p.append(f"<h1>Borderline review — {len(rows)} BP clips × 3 IPA sources</h1>")
    p.append(
        "<p class='sub'>Streamed audio (R2) · 3-column word IPA "
        "(Stage 5 LLM · MFA dict · espeak pt-br) · 2-column example IPA "
        "(Stage 5 · espeak) · per-clip verdict + free-text notes. "
        "Your work is auto-saved to <code>localStorage</code> — reloading "
        "won't lose it. Export to TSV when done.</p>"
    )

    # ---- how to read ----
    p.append("<details class='howto' open>")
    p.append("<summary>How to use this — and how the three IPA columns work together</summary>")
    p.append("<div class='howto-body'>")
    p.append(
        "<p><b>Per clip</b>, play the audio and decide whether the spoken "
        "pronunciation matches a textbook BP target. The three word-IPA "
        "columns give you three independent guesses at that target:</p>"
    )
    p.append("<h4>IPA columns</h4><ul>")
    p.append(
        "<li><b>Stage 5 (LLM)</b> — what's currently stored in "
        "<code>06-final.tsv</code>, produced by an LLM during the IPA pass. "
        "Inconsistent on tricky cases, hence this review.</li>"
    )
    p.append(
        "<li><b>MFA dict</b> — <code>portuguese_brazil_mfa.dict</code> "
        "(34,978 entries). Curated for the Montreal Forced Aligner's BP "
        "acoustic model. Multiple variants are joined with <code> / </code>. "
        "Empty if the word is OOV (out of vocabulary).</li>"
    )
    p.append(
        "<li><b>espeak pt-br</b> — <code>phonemizer + espeak-ng</code>, "
        "rule-based, deterministic. Quirks normalized to Stage-5 convention "
        "(<code>y</code>→<code>i</code>, <code>x</code>→<code>ʁ</code>, "
        "<code>lj</code>→<code>ʎ</code>, …).</li>"
    )
    p.append("</ul>")
    p.append("<h4>Decision heuristic</h4><ul>")
    p.append("<li><b>All 3 agree + audio matches</b> → good</li>")
    p.append("<li><b>All 3 agree but audio diverges</b> → bad (mispronunciation)</li>")
    p.append("<li><b>The 3 disagree</b> → this row was always going to be controversial. Note it.</li>")
    p.append("</ul>")
    p.append("<h4>Sources of the 300</h4>")
    p.append("<table class='legend'><tr><th>label</th><th>meaning</th><th style='text-align:right'>n</th></tr>")
    p.append(legend_rows)
    p.append("</table>")
    p.append("<h4>Keyboard</h4><ul>")
    p.append("<li><b>G</b> / <b>B</b> / <b>U</b> — verdict for the first visible unreviewed card</li>")
    p.append("</ul>")
    p.append("</div></details>")

    # ---- toolbar ----
    p.append("<div class='toolbar'>")
    p.append(
        f"<label>source <select id='f-src'><option value=''>all</option>"
        f"{src_options}</select></label>"
    )
    p.append(
        "<label>review <select id='f-rev'><option value=''>all</option>"
        "<option value='unreviewed'>unreviewed</option>"
        "<option value='good'>good</option>"
        "<option value='bad'>bad</option>"
        "<option value='unsure'>unsure</option></select></label>"
    )
    p.append(
        "<label>search <input id='f-text' type='text' "
        "placeholder='pt / en / sense_id' size='18'></label>"
    )
    p.append("<span class='grow'></span><span id='count'></span>")
    p.append("<button id='export-btn'>Export judgments (TSV)</button>")
    p.append("<button id='clear-btn' class='ghost'>Clear all</button>")
    p.append("</div>")

    # ---- per-clip cards ----
    p.append("<div id='cards'>")
    for r in rows:
        p.append(card_html(r))
    p.append("</div>")

    p.append(f"<script>{JS}</script>")
    p.append("</div></body></html>")

    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    REPORT_HTML.write_text("".join(p), encoding="utf-8")
    size_kb = REPORT_HTML.stat().st_size / 1024
    print(f"wrote {REPORT_HTML}  ({len(rows)} clips, {size_kb:.0f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    if not FINAL_TSV.exists():
        print(f"missing {FINAL_TSV}", file=sys.stderr)
        return 1
    if not MFA_DICT.exists():
        print(f"missing {MFA_DICT}", file=sys.stderr)
        print("  curl -sL -o data/_mfa_dict/portuguese_brazil_mfa.dict \\", file=sys.stderr)
        print("    https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/portuguese/brazil_mfa/portuguese_brazil_mfa.dict",
              file=sys.stderr)
        return 1

    rows = assemble_dataset()
    write_seed_tsv(rows)
    write_html(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
