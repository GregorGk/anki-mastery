"""Stage 16 / Step 4 — Score each judge against your human labels.

Reads:
    data/_audio_judge_ab.tsv                — row metadata (dialect col is the gold)
    data/_audio_judge_ab_human_labels.tsv   — labels you exported from the HTML
    audit/16_judge_verdicts.jsonl           — verdicts from 16_2_run_judges.py

Computes for each judge AND for your own ears:
    - overall accuracy vs gold dialect (BP = bp_ok; EP = non_bp_ep)
    - precision/recall on non_bp detection
    - confusion matrix
    - cost + average latency
    - rate of agreement with you (the human gold)

A judge "wins" by matching the human gold most reliably while spending least.
The script prints a single comparison table.

Usage:
    .venv/bin/python build/16_4_score_judges.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"
INPUT_TSV = DATA / "_audio_judge_ab.tsv"
HUMAN_LABELS = DATA / "_audio_judge_ab_human_labels.tsv"
VERDICTS_JSONL = AUDIT / "16_judge_verdicts.jsonl"

JUDGE_LABELS = {
    "gpt4o": "gpt-4o-audio-preview",
    "gpt_audio_15": "gpt-audio-1.5",
    "gemini_31_pro": "gemini-3.1-pro",
}

# Per-judge audit JSONLs (appended live as 16_2 runs; the merged file
# above only contains the LAST run's verdicts, so we read the per-judge
# audits as the canonical source).
PER_JUDGE_AUDITS = {
    "gpt4o":         AUDIT / "16_judge_gpt4o.jsonl",
    "gpt_audio_15":  AUDIT / "16_judge_gpt_audio_15.jsonl",
    "gemini_31_pro": AUDIT / "16_judge_gemini_31_pro.jsonl",
}


def _normalize_verdict_to_class(v: str) -> str:
    """Map any verdict form to one of: bp / nonbp / unclear."""
    v = (v or "").lower().strip()
    if v in ("bp_ok", "bp", "ok"):
        return "bp"
    if v.startswith("non_bp") or v == "non-bp" or v == "nonbp":
        return "nonbp"
    return "unclear"


def _confusion(predictions: list[tuple[str, str]]) -> dict:
    """Return {(gold, pred): count}."""
    out: Counter = Counter()
    for gold, pred in predictions:
        out[(gold, pred)] += 1
    return dict(out)


def _metrics(predictions: list[tuple[str, str]]) -> dict:
    """Accuracy + precision/recall on 'nonbp' detection.

    predictions: list of (gold_class, pred_class) in {'bp','nonbp','unclear'}.
    """
    n = len(predictions)
    if n == 0:
        return {"n": 0, "acc": 0.0, "precision_nonbp": 0.0, "recall_nonbp": 0.0,
                "f1_nonbp": 0.0}
    correct = sum(1 for g, p in predictions if g == p)
    tp = sum(1 for g, p in predictions if g == "nonbp" and p == "nonbp")
    fp = sum(1 for g, p in predictions if g != "nonbp" and p == "nonbp")
    fn = sum(1 for g, p in predictions if g == "nonbp" and p != "nonbp")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"n": n, "acc": correct / n, "precision_nonbp": precision,
            "recall_nonbp": recall, "f1_nonbp": f1, "correct": correct,
            "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    if not INPUT_TSV.exists():
        print(f"ERROR: missing {INPUT_TSV}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(INPUT_TSV.open(encoding="utf-8"), dialect="excel-tab"))
    row_meta = {r["row_id"]: r for r in rows}
    gold_dialect = {r["row_id"]: ("bp" if r["dialect"] == "BP" else "nonbp")
                    for r in rows}
    row_id_lookup = {(r["sense_id"], r["clip_type"], r["voice_id"]): r["row_id"]
                     for r in rows}

    # Load judge verdicts — prefer per-judge audits (canonical, 800 entries
    # each), fall back to the merged JSONL for legacy compatibility.
    judge_verdicts: dict[str, dict[str, dict]] = {}
    for jkey, path in PER_JUDGE_AUDITS.items():
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("sense_id", ""), rec.get("clip_type", ""),
                   rec.get("voice_id", ""))
            rid = row_id_lookup.get(key)
            if rid:
                # Last write wins — per-judge audits are append-only,
                # so retries naturally supersede earlier failures.
                judge_verdicts.setdefault(rid, {})[jkey] = rec
    # Optional supplement from merged file (no overwrite if already present).
    if VERDICTS_JSONL.exists():
        for line in VERDICTS_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("row_id")
            j = rec.get("judge")
            if rid and j and j not in judge_verdicts.get(rid, {}):
                judge_verdicts.setdefault(rid, {})[j] = rec

    # Load human labels (gold from your ears).
    human: dict[str, str] = {}
    if HUMAN_LABELS.exists():
        for r in csv.DictReader(HUMAN_LABELS.open(encoding="utf-8"), dialect="excel-tab"):
            if r.get("row_id"):
                human[r["row_id"]] = _normalize_verdict_to_class(r.get("human_verdict", ""))

    print(f"=== Stage 16 audio-judge A/B scoring ===")
    print(f"  pool size:      {len(rows)} clips ({sum(1 for r in rows if r['dialect']=='BP')} BP / "
          f"{sum(1 for r in rows if r['dialect']=='EP')} EP)")
    print(f"  judge verdicts: {sum(len(v) for v in judge_verdicts.values())}")
    print(f"  human labels:   {len(human)}")
    print()

    if not human:
        print("WARN: no human labels found at "
              f"{HUMAN_LABELS}. Listing judge-vs-dialect metrics only.\n",
              file=sys.stderr)

    # ─── Metrics vs. the dialect gold (synthetic) ───
    print(f"--- Judges vs. dialect-of-origin (synthetic gold) ---")
    print(f"  Treats BP-voiced clips as gold=bp, EP-voiced clips as gold=nonbp.")
    print(f"  Judge is correct when it predicts bp on BP and nonbp on EP.")
    print()
    print(f"{'judge':<28}{'n':>5}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}{'cost_usd':>10}{'latency_ms':>13}")
    print("-" * 96)
    for jkey, jlabel in JUDGE_LABELS.items():
        preds = []
        cost_sum = 0.0
        lat_sum = 0
        for rid, gold in gold_dialect.items():
            v = judge_verdicts.get(rid, {}).get(jkey)
            if not v or v.get("error"):
                continue
            preds.append((gold, _normalize_verdict_to_class(v.get("verdict", ""))))
            cost_sum += v.get("cost_usd", 0.0) or 0.0
            lat_sum += v.get("latency_ms", 0) or 0
        m = _metrics(preds)
        avg_lat = lat_sum / m["n"] if m["n"] else 0
        print(f"{jlabel:<28}{m['n']:>5}{m['acc']:>8.3f}{m['precision_nonbp']:>8.3f}"
              f"{m['recall_nonbp']:>8.3f}{m['f1_nonbp']:>8.3f}"
              f"{cost_sum:>10.4f}{avg_lat:>13.0f}")

    # ─── Metrics vs. human labels (true gold, where the user disagreed
    #     with the synthetic dialect tag — e.g. a BP clip that sounds EP) ───
    if human:
        print()
        print(f"--- Judges vs. your ears (true gold) ---")
        print(f"  Only counts rows where you provided a label.")
        print()
        print(f"{'judge':<28}{'n':>5}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}{'cost_usd':>10}")
        print("-" * 81)
        for jkey, jlabel in JUDGE_LABELS.items():
            preds = []
            cost_sum = 0.0
            for rid, gold in human.items():
                v = judge_verdicts.get(rid, {}).get(jkey)
                if not v or v.get("error"):
                    continue
                preds.append((gold, _normalize_verdict_to_class(v.get("verdict", ""))))
                cost_sum += v.get("cost_usd", 0.0) or 0.0
            m = _metrics(preds)
            print(f"{jlabel:<28}{m['n']:>5}{m['acc']:>8.3f}{m['precision_nonbp']:>8.3f}"
                  f"{m['recall_nonbp']:>8.3f}{m['f1_nonbp']:>8.3f}{cost_sum:>10.4f}")

        # Disagreement rows — interesting to surface.
        print()
        print(f"--- Top human-vs-synthetic-gold disagreements (your label ≠ dialect tag) ---")
        n_disagree = 0
        for rid, hg in human.items():
            sg = gold_dialect.get(rid, "")
            if hg != sg:
                meta = row_meta[rid]
                n_disagree += 1
                if n_disagree <= 20:
                    print(f"  {rid}  dialect={sg:<6} human={hg:<8} sense={meta['sense_id']} "
                          f"voice={meta['voice_name']:<25} text={meta['text'][:60]}")
        if n_disagree > 20:
            print(f"  ... + {n_disagree - 20} more")
        print(f"  Total disagreements: {n_disagree}/{len(human)} "
              f"({100*n_disagree/max(len(human),1):.1f}%)")

    # ─── Per-judge confusion matrix vs. dialect gold ───
    print()
    print(f"--- Confusion matrices (rows=gold dialect, cols=judge prediction) ---")
    for jkey, jlabel in JUDGE_LABELS.items():
        print(f"\n  {jlabel}:")
        preds = []
        for rid, gold in gold_dialect.items():
            v = judge_verdicts.get(rid, {}).get(jkey)
            if not v or v.get("error"):
                continue
            preds.append((gold, _normalize_verdict_to_class(v.get("verdict", ""))))
        conf = _confusion(preds)
        classes = ["bp", "nonbp", "unclear"]
        print(f"    {'':<10}" + "".join(f"{c:>10}" for c in classes))
        for g in classes:
            print(f"    {g:<10}" + "".join(f"{conf.get((g,p), 0):>10}" for p in classes))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
