"""Analyze the existing audit/ab_asr.jsonl without re-running API calls.

Reads the per-clip results from `build/ab_asr_models.py`, prints the summary
+ pairwise agreement matrix + disagreement log. Used to recover from a
crash in the analysis step or to re-render with different formatting
without paying for ASR again.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JSONL = REPO_ROOT / "audit" / "ab_asr.jsonl"
OUT_SUMMARY = REPO_ROOT / "audit" / "ab_asr_summary.tsv"
OUT_DISAGREE = REPO_ROOT / "audit" / "ab_asr_disagreements.log"

PRICE_PER_MIN = {
    "whisper-1": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
    "gpt-4o-transcribe": 0.006,
    "scribe_v2": 0.0067,
}


def main() -> int:
    if not JSONL.exists():
        print(f"missing {JSONL}", file=sys.stderr)
        return 1
    rows: list[dict] = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    by_model: dict[str, list[dict]] = defaultdict(list)
    by_clip: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_model[r["model"]].append(r)
        by_clip[(r["sense_id"], r["clip_type"])][r["model"]] = r

    models = list(by_model.keys())
    print(f"\n=== A/B summary by model ({len(rows):,} clip-runs) ===")
    print(
        f"{'model':<28} {'pass':>5} {'regen':>5} {'pass%':>7} {'short_pass%':>12} {'long_pass%':>11} {'cost_usd':>9}"
    )

    summary_rows: list[dict] = []
    for m in models:
        rs = by_model[m]
        passes = sum(1 for r in rs if r["decision"] == "pass")
        regens = sum(1 for r in rs if r["decision"] == "regen")
        short = [r for r in rs if r.get("is_short")]
        long_ = [r for r in rs if not r.get("is_short")]
        spass = sum(1 for r in short if r["decision"] == "pass")
        lpass = sum(1 for r in long_ if r["decision"] == "pass")
        # Cost from durations × per-minute price
        total_dur_sec = sum(r.get("cost_usd", 0) / max(PRICE_PER_MIN.get(m, 0.006) / 60, 1e-9) for r in rs)
        # Easier: just sum r.cost_usd
        cost = sum(r.get("cost_usd", 0) or 0 for r in rs)
        pr = passes / len(rs) if rs else 0
        spr = spass / len(short) if short else 0
        lpr = lpass / len(long_) if long_ else 0
        print(
            f"{m:<28} {passes:>5} {regens:>5} {pr * 100:>6.1f}% {spr * 100:>11.1f}% {lpr * 100:>10.1f}% ${cost:>7.3f}"
        )
        summary_rows.append(
            {
                "model": m,
                "total": len(rs),
                "pass": passes,
                "regen": regens,
                "pass_rate": f"{pr:.4f}",
                "short_clips": len(short),
                "short_pass": spass,
                "short_pass_rate": f"{spr:.4f}" if short else "",
                "long_clips": len(long_),
                "long_pass": lpass,
                "long_pass_rate": f"{lpr:.4f}" if long_ else "",
                "cost_usd": f"{cost:.4f}",
                "price_per_min": f"{PRICE_PER_MIN.get(m, 0):.4f}",
            }
        )

    # Pairwise PASS-agreement
    print("\n=== Pairwise PASS-agreement matrix ===")
    print("            " + "".join(f"{m[:14]:<16}" for m in models))
    for m1 in models:
        cells = []
        for m2 in models:
            both_pass = 0
            both_count = 0
            for per_model in by_clip.values():
                if m1 in per_model and m2 in per_model:
                    both_count += 1
                    if per_model[m1]["decision"] == "pass" and per_model[m2]["decision"] == "pass":
                        both_pass += 1
            agree = both_pass / both_count if both_count else 0
            cells.append(f"{agree * 100:>14.1f}% ")
        print(f"{m1[:11]:<11} " + "".join(cells))

    # Disagreement log
    disagree_lines: list[str] = []
    n_disagree = 0
    for (sid, ctype), per_model in by_clip.items():
        decisions = {m: r["decision"] for m, r in per_model.items()}
        if len(set(decisions.values())) > 1:
            n_disagree += 1
            base = next(iter(per_model.values()))
            disagree_lines.append(
                f"=== {sid} {ctype}  '{base.get('input_text', '')}'  (rank={base.get('rank', 0)}, voice={(base.get('voice_id') or '')[:6]}…)"
            )
            for m, r in per_model.items():
                pdist = r.get("phonetic_dist")
                pdist_str = "  -  " if pdist is None else f"{pdist:>4.2f}"
                tr = (r.get("transcript") or "").replace("\n", " ")[:80]
                disagree_lines.append(
                    f"   {m:<28} {r['decision']:<5} sim={r.get('text_sim', 0.0):>4.2f} pdist={pdist_str}  ->  {tr}"
                )
            disagree_lines.append("")

    OUT_DISAGREE.write_text("\n".join(disagree_lines), encoding="utf-8")
    OUT_SUMMARY.write_text(
        "\t".join(summary_rows[0].keys())
        + "\n"
        + "\n".join("\t".join(str(r[k]) for k in r) for r in summary_rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n=== Disagreements: {n_disagree} clip(s) had at least one model differ ===")
    print(f"  -> {OUT_DISAGREE}")
    print(f"  -> {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
