"""Stage 16 / Step 3 — Build the listening HTML for the audio-judge A/B test.

For each clip in `data/_audio_judge_ab.tsv`, render a row with:
  • Audio player
  • Word / example text
  • Voice + dialect tag
  • EP/BP listening cues (BP IPA, EP IPA, headline marker, 2-4 cues,
    loaded from data/_ep_cues.tsv if present) — collapsed by default so
    you can listen "blind" first; click 🎓 to reveal per-row.
  • Each judge's verdict (gpt-4o, gpt-audio-1.5, gemini-3.1-pro)
  • Radio buttons for your verdict: bp_ok / non_bp (EP) / non_bp (other)
    / unclear

JavaScript at the bottom exports your labels as TSV — paste into
`data/_audio_judge_ab_human_labels.tsv`, then run 16_4_score_judges.py.

Usage:
    .venv/bin/python build/16_3_judge_ab_html.py
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from collections import defaultdict
from html import escape
from pathlib import Path


def _norm_ipa_strict(s: str) -> str:
    """Normalize IPA for equality comparison: NFD, strip whitespace/stress."""
    s = unicodedata.normalize("NFD", (s or "").strip().lower())
    s = "".join(s.split())
    for ch in "ˈˌ.—‑-":
        s = s.replace(ch, "")
    return s

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"
INPUT_TSV = DATA / "_audio_judge_ab.tsv"
CUES_TSV = DATA / "_ep_cues.tsv"
VERDICTS = AUDIT / "16_judge_verdicts.jsonl"
OUT_HTML = AUDIT / "16_audio_judge_ab.html"

JUDGE_LABELS = {
    "gpt4o": "gpt-4o-audio-preview",
    "gpt_audio_15": "gpt-audio-1.5",
    "gemini_31_pro": "gemini-3.1-pro",
}
JUDGE_ORDER = ["gpt4o", "gpt_audio_15", "gemini_31_pro"]

# Per-judge audit JSONL paths (appended row-by-row by 16_2_run_judges.py via
# the AudioJudgeClient / GeminiAudioJudgeClient `audit_path=` kwarg). These
# are populated LIVE during the 16_2 run; the merged JSONL above is only
# written at run-end. We read per-judge audits to surface in-flight verdicts.
PER_JUDGE_AUDITS = {
    "gpt4o":          AUDIT / "16_judge_gpt4o.jsonl",
    "gpt_audio_15":   AUDIT / "16_judge_gpt_audio_15.jsonl",
    "gemini_31_pro":  AUDIT / "16_judge_gemini_31_pro.jsonl",
}


def _verdict_class(v: str) -> str:
    return {
        "bp_ok": "ok",
        "non_bp": "no",
        "unclear": "warn",
    }.get(v, "")


def _judge_correct(judge_verdict: str, dialect: str) -> bool | None:
    """Return True/False/None given a verdict ('bp_ok'/'non_bp'/'unclear')
    and the row's dialect-of-origin ('BP'/'EP').

    None when the judge didn't return a usable verdict (error / missing).
    'unclear' counts as incorrect — the judge failed to commit.
    """
    if not judge_verdict:
        return None
    if dialect == "BP":
        return judge_verdict == "bp_ok"
    if dialect == "EP":
        return judge_verdict == "non_bp"
    return None


def _load_cues() -> dict[tuple[str, str], dict]:
    """{(text, clip_type): cue_dict} from data/_ep_cues.tsv."""
    if not CUES_TSV.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    for r in csv.DictReader(CUES_TSV.open(encoding="utf-8"), dialect="excel-tab"):
        out[(r["text"], r["clip_type"])] = r
    return out


def main() -> int:
    if not INPUT_TSV.exists():
        print(f"ERROR: {INPUT_TSV} missing. Run build/16_1_audio_judge_ab_prep.py first.",
              file=sys.stderr)
        return 1

    rows = list(csv.DictReader(INPUT_TSV.open(encoding="utf-8"), dialect="excel-tab"))
    cues = _load_cues()

    # Build (sense_id, clip_type, voice_id) → row_id map so we can look up
    # judge audits that don't carry row_id directly.
    row_id_lookup: dict[tuple[str, str, str], str] = {
        (r["sense_id"], r["clip_type"], r["voice_id"]): r["row_id"]
        for r in rows
    }

    # 1. Merged JSONL (final state, written at 16_2 end). Carries row_id.
    verdicts: dict[str, dict[str, dict]] = defaultdict(dict)
    if VERDICTS.exists():
        for line in VERDICTS.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if "row_id" in rec and "judge" in rec:
                verdicts[rec["row_id"]][rec["judge"]] = rec

    # 2. Per-judge audit JSONLs (live, appended during 16_2). These don't
    # carry row_id — we join by (sense_id, clip_type, voice_id). When the
    # merged file is missing a verdict, fall back to the per-judge audit.
    for judge_key, audit_path in PER_JUDGE_AUDITS.items():
        if not audit_path.exists():
            continue
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("sense_id", ""), rec.get("clip_type", ""),
                   rec.get("voice_id", ""))
            rid = row_id_lookup.get(key)
            if not rid:
                continue
            # Only fill if the merged file didn't already have this judge.
            if judge_key in verdicts.get(rid, {}):
                continue
            verdicts[rid][judge_key] = rec

    n_with_cues = sum(1 for r in rows if (r["text"], r["clip_type"]) in cues)

    # Mark rows where BP IPA == EP IPA (no audible dialect contrast — both
    # the LLM judges and the human ear are guessing on those). They're
    # marked with data-same-ipa=1 in the HTML and excluded from the
    # default disagreement filter.
    def _same_ipa(row: dict) -> bool:
        c = cues.get((row["text"], row["clip_type"]))
        if not c:
            return False
        return (_norm_ipa_strict(c.get("bp_ipa", ""))
                == _norm_ipa_strict(c.get("ep_ipa", "")))

    row_same_ipa: dict[str, bool] = {r["row_id"]: _same_ipa(r) for r in rows}
    n_same_ipa = sum(1 for v in row_same_ipa.values() if v)

    # Compute per-row disagreement count: how many of the 3 judges disagree
    # with the dialect-of-origin gold. Used to drive the dropdown filter.
    def _row_wrong_count(row: dict) -> int:
        rid = row["row_id"]
        dialect = row["dialect"]
        wrong = 0
        for j in JUDGE_ORDER:
            v = verdicts.get(rid, {}).get(j, {})
            ok = _judge_correct(v.get("verdict", ""), dialect)
            if ok is False:
                wrong += 1
        return wrong

    row_wrong: dict[str, int] = {r["row_id"]: _row_wrong_count(r) for r in rows}
    # Counts that EXCLUDE identical-IPA rows (the meaningful disagreements).
    n_any_wrong = sum(1 for rid, w in row_wrong.items()
                      if w >= 1 and not row_same_ipa[rid])
    n_two_wrong = sum(1 for rid, w in row_wrong.items()
                      if w >= 2 and not row_same_ipa[rid])
    n_all_wrong = sum(1 for rid, w in row_wrong.items()
                      if w >= 3 and not row_same_ipa[rid])
    n_unanimous_correct = sum(1 for rid, w in row_wrong.items()
                              if w == 0 and not row_same_ipa[rid])

    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Stage 16 — Audio-judge A/B: human × 3 models</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;",
        " max-width:1700px;margin:24px auto;padding:0 20px;color:#222;}",
        "h1{font-size:1.4em;}",
        ".toolbar{position:sticky;top:0;background:#ffd;padding:.6em 1em;",
        " border:2px solid #ba0;border-radius:4px;margin:.5em 0;z-index:100;",
        " display:flex;align-items:center;gap:1em;flex-wrap:wrap;}",
        "table{border-collapse:collapse;width:100%;font-size:.85em;}",
        "th,td{border:1px solid #ddd;padding:6px 8px;vertical-align:top;}",
        "th{background:#eef;font-weight:600;text-align:left;}",
        "audio{width:240px;height:28px;}",
        ".bp{background:#e7f5e6;}",
        ".ep{background:#fef0e6;}",
        ".verdict-ok{color:#155724;font-weight:600;}",
        ".verdict-no{color:#842029;font-weight:600;}",
        ".verdict-warn{color:#664d03;font-weight:600;}",
        ".sm{font-size:.78em;color:#666;}",
        "td.text{font-style:italic;color:#333;max-width:240px;}",
        "td.judge{font-family:ui-monospace,monospace;max-width:200px;}",
        "td.human label{display:block;white-space:nowrap;margin:1px 0;}",
        "textarea#export-out{width:100%;height:160px;font-family:ui-monospace,monospace;",
        " font-size:.85em;margin-top:.5em;}",
        # Cues column — toggleable per row
        "td.cues{font-size:.78em;color:#444;max-width:380px;}",
        ".cues-toggle{cursor:pointer;background:#eef;border:1px solid #99c;",
        " border-radius:3px;padding:2px 8px;font-size:.85em;user-select:none;}",
        ".cues-toggle.diff-obvious{background:#dff5d7;border-color:#5a8;}",
        ".cues-toggle.diff-moderate{background:#fff3cd;border-color:#aa6;}",
        ".cues-toggle.diff-subtle{background:#fde2e2;border-color:#a55;}",
        ".cues-body{display:none;margin-top:4px;}",
        ".cues-body.revealed{display:block;}",
        "body.all-cues-shown .cues-body{display:block;}",
        # Row-disagreement filter: dropdown adds a body class; per-row data
        # attribute marks how many judges erred (0..3).
        "body[data-filter='wrong1'] tr[data-wrong='0']{display:none;}",
        "body[data-filter='wrong2'] tr[data-wrong='0'],",
        "body[data-filter='wrong2'] tr[data-wrong='1']{display:none;}",
        "body[data-filter='wrong3'] tr[data-wrong='0'],",
        "body[data-filter='wrong3'] tr[data-wrong='1'],",
        "body[data-filter='wrong3'] tr[data-wrong='2']{display:none;}",
        "body[data-filter='correct'] tr:not([data-wrong='0']){display:none;}",
        # Also exclude identical-IPA rows from any 'wrong*' or 'correct' view
        "body[data-filter='wrong1'] tr[data-same-ipa='1'],",
        "body[data-filter='wrong2'] tr[data-same-ipa='1'],",
        "body[data-filter='wrong3'] tr[data-same-ipa='1'],",
        "body[data-filter='correct'] tr[data-same-ipa='1']{display:none;}",
        # Dedicated 'same-ipa' filter shows ONLY those rows.
        "body[data-filter='same_ipa'] tr:not([data-same-ipa='1']){display:none;}",
        ".wrong-badge{display:inline-block;padding:1px 6px;border-radius:3px;",
        " font-size:.78em;margin-left:4px;}",
        ".wrong-0{background:#d4edda;color:#155724;}",
        ".wrong-1{background:#fff3cd;color:#664d03;}",
        ".wrong-2{background:#fde2e2;color:#842029;}",
        ".wrong-3{background:#f5c0c0;color:#5a1010;font-weight:600;}",
        # Identical-IPA rows get a striped background so the user knows they're
        # un-distinguishable (no audible dialect difference).
        "tr[data-same-ipa='1']{background:repeating-linear-gradient(45deg,",
        " transparent,transparent 6px,#f7f7f7 6px,#f7f7f7 12px);}",
        ".same-ipa-badge{display:inline-block;padding:1px 6px;border-radius:3px;",
        " font-size:.72em;background:#ddd;color:#555;margin-left:4px;}",
        ".cues-headline{font-weight:600;color:#06598f;margin-bottom:4px;}",
        ".cues-ipa{font-family:ui-monospace,monospace;color:#555;font-size:.88em;",
        " margin:2px 0;}",
        ".cues-list{margin:6px 0 0 1em;padding:0;list-style:'·  ';}",
        ".cues-list li{margin:3px 0;line-height:1.35;}",
        "</style></head><body>",

        "<h1>Stage 16 — Audio-judge A/B test</h1>",
        f"<p class='sm'>{len(rows)} clips. ",
        f"BP clips from v3 regen (live voices). ",
        f"EP clips synthesized on v3 with Nelson Silvestre (Lisbon accent). ",
        f"You listen, mark each, export TSV at the bottom. ",
        f"<strong>EP cues loaded for {n_with_cues}/{len(rows)} rows.</strong></p>",

        "<div class='toolbar'>",
        "<button onclick='exportLabels()'>Export human labels</button>",
        "<button onclick='toggleAllCues()' id='cues-toggle-btn'>"
        "Show all listening cues</button>",
        # NEW: filter dropdown — wrong* + correct buckets exclude
        # identical-IPA rows; 'same_ipa' shows ONLY those.
        "<label class='sm'>Filter rows: "
        "<select id='filter-select' onchange='applyFilter(this.value)'>"
        f"<option value='all'>All ({len(rows)})</option>"
        f"<option value='wrong1' selected>≥1 judge wrong, excl. same-IPA ({n_any_wrong})</option>"
        f"<option value='wrong2'>≥2 judges wrong, excl. same-IPA ({n_two_wrong})</option>"
        f"<option value='wrong3'>All 3 wrong, excl. same-IPA ({n_all_wrong})</option>"
        f"<option value='correct'>All 3 correct, excl. same-IPA ({n_unanimous_correct})</option>"
        f"<option value='same_ipa'>Same-IPA only ({n_same_ipa})</option>"
        "</select></label>",
        f"<span class='sm'>Judge verdicts loaded: "
        f"{sum(1 for v in verdicts.values() if len(v) >= len(JUDGE_ORDER))}/{len(rows)} rows</span>",
        "<span class='sm'>Difficulty legend: "
        "<span class='cues-toggle diff-obvious'>obvious</span> "
        "<span class='cues-toggle diff-moderate'>moderate</span> "
        "<span class='cues-toggle diff-subtle'>subtle</span></span>",
        "</div>",
        "<textarea id='export-out' placeholder='Click \"Export human labels\" — TSV appears here.'></textarea>",

        "<table><thead><tr>",
        "<th>row_id</th><th>sense_id</th><th>dialect</th>",
        "<th>text</th><th>voice</th><th>audio</th>",
        "<th>listening cues</th>",
    ]
    for j in JUDGE_ORDER:
        parts.append(f"<th>{escape(JUDGE_LABELS[j])}</th>")
    parts.append("<th>YOUR verdict</th>")
    parts.append("</tr></thead><tbody>")

    for r in rows:
        rid = r["row_id"]
        dialect_class = "bp" if r["dialect"] == "BP" else "ep"
        url = r["url"]
        v_row = verdicts.get(rid, {})
        cue = cues.get((r["text"], r["clip_type"]))
        wrong_n = row_wrong[rid]
        same_ipa_flag = "1" if row_same_ipa[rid] else "0"

        parts.append(f"<tr class='{dialect_class}' "
                     f"data-wrong='{wrong_n}' data-same-ipa='{same_ipa_flag}'>")
        same_ipa_marker = (" <span class='same-ipa-badge' "
                           "title='BP and EP IPA are identical — no audible dialect contrast'>"
                           "same-IPA</span>") if row_same_ipa[rid] else ""
        parts.append(f"<td class='sm'>{escape(rid)}"
                     f"<span class='wrong-badge wrong-{wrong_n}'>{wrong_n}/3</span>"
                     f"{same_ipa_marker}</td>")
        parts.append(f"<td class='sm'>{escape(r['sense_id'])}<br>{escape(r['clip_type'])}</td>")
        parts.append(f"<td><strong>{escape(r['dialect'])}</strong></td>")
        parts.append(f"<td class='text'>{escape(r['text'])}</td>")
        parts.append(f"<td class='sm'>{escape(r.get('voice_name', ''))}</td>")
        parts.append(f"<td><audio controls preload='none' src='{escape(url)}'></audio></td>")

        # ----- listening cues cell -----
        parts.append("<td class='cues'>")
        if cue:
            diff = cue.get("difficulty", "")
            parts.append(
                f"<span class='cues-toggle diff-{escape(diff)}' "
                f"onclick='this.nextElementSibling.classList.toggle(\"revealed\")'>"
                f"🎓 {escape(diff or 'cues')}</span>"
            )
            parts.append("<div class='cues-body'>")
            if cue.get("headline_marker"):
                parts.append(
                    f"<div class='cues-headline'>👂 "
                    f"{escape(cue['headline_marker'])}</div>"
                )
            if cue.get("bp_ipa"):
                parts.append(
                    f"<div class='cues-ipa'>BP IPA: "
                    f"<strong>{escape(cue['bp_ipa'])}</strong></div>"
                )
            if cue.get("ep_ipa"):
                parts.append(
                    f"<div class='cues-ipa'>EP IPA: "
                    f"<strong>{escape(cue['ep_ipa'])}</strong></div>"
                )
            cue_items = [cue.get(f"cue_{i}") for i in (1, 2, 3, 4)]
            cue_items = [c for c in cue_items if c]
            if cue_items:
                parts.append("<ul class='cues-list'>")
                for c in cue_items:
                    parts.append(f"<li>{escape(c)}</li>")
                parts.append("</ul>")
            parts.append("</div>")
        else:
            parts.append("<span class='sm'>(no cues — run 16_5)</span>")
        parts.append("</td>")

        # ----- judge verdict cells -----
        for j in JUDGE_ORDER:
            v = v_row.get(j)
            if not v or v.get("error"):
                inner = (f"<span class='sm verdict-warn'>"
                         f"{escape((v or {}).get('error', '(missing)'))[:40]}"
                         f"</span>")
            else:
                klass = _verdict_class(v.get("verdict", ""))
                inner = (
                    f"<span class='verdict-{klass}'>{escape(v.get('verdict', '?'))}</span><br>"
                    f"<span class='sm'>drift={escape(v.get('drift', '-'))} "
                    f"conf={escape(v.get('confidence', '-'))}</span><br>"
                    f"<span class='sm'>{escape((v.get('evidence') or '')[:120])}</span>"
                )
            parts.append(f"<td class='judge'>{inner}</td>")

        # ----- your verdict cell -----
        parts.append("<td class='human'>")
        for label, value in [
            ("BP-OK",      "bp_ok"),
            ("non-BP (EP)", "non_bp_ep"),
            ("non-BP (other)", "non_bp_other"),
            ("unclear",    "unclear"),
        ]:
            parts.append(
                f"<label><input type='radio' name='h_{escape(rid)}' "
                f"value='{value}'> {label}</label>"
            )
        parts.append("</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")

    parts.append("""
<script>
function exportLabels() {
  const lines = ['row_id\\thuman_verdict'];
  document.querySelectorAll('tbody tr').forEach(tr => {
    const rid_cell = tr.querySelector('td');
    const rid = rid_cell.textContent.trim();
    const sel = tr.querySelector('input[type=radio]:checked');
    if (sel) lines.push(rid + '\\t' + sel.value);
  });
  document.getElementById('export-out').value = lines.join('\\n');
}
function toggleAllCues() {
  const body = document.body;
  body.classList.toggle('all-cues-shown');
  const btn = document.getElementById('cues-toggle-btn');
  btn.textContent = body.classList.contains('all-cues-shown')
    ? 'Hide all listening cues'
    : 'Show all listening cues';
}
function applyFilter(mode) {
  document.body.setAttribute('data-filter', mode);
  // Count visible rows for sanity:
  const total = document.querySelectorAll('tbody tr').length;
  const visible = Array.from(document.querySelectorAll('tbody tr')).filter(
    tr => window.getComputedStyle(tr).display !== 'none'
  ).length;
  console.log('filter=' + mode + ': ' + visible + '/' + total + ' rows visible');
}
// Initialize default filter (matches `selected` option in the select).
applyFilter(document.getElementById('filter-select').value);
document.querySelectorAll('audio').forEach(a => {
  a.addEventListener('play', () => {
    document.querySelectorAll('audio').forEach(o => { if (o !== a) o.pause(); });
  });
});
</script>
</body></html>
""")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT_HTML}  ({len(rows)} rows, cues for {n_with_cues})")
    print(f"open with: open {OUT_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
