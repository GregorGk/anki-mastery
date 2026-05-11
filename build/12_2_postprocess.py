"""Stage 12 / Step 2 — Apply deterministic post-LLM rules to the sidecar.

Idempotent: safe to re-run. Reads data/_usage_hints.tsv, applies the
four rules from build/lib/usage_hint_rules.py (tier-1 overrides,
essential→useful downgrades, parenthetical strip, risk_note rerouting),
writes back. Adds the `risk_note` column if missing.

Usage:
    .venv/bin/python build/12_2_postprocess.py
    .venv/bin/python build/12_2_postprocess.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from build.lib.usage_hint_rules import (  # noqa: E402
    ESSENTIAL_TO_USEFUL_LEMMAS,
    RISK_NOTE_OVERRIDES,
    TIER_1_OVERRIDES,
    apply_post_process,
)

SIDECAR = REPO / "data" / "_usage_hints.tsv"

POST_FIELDS = [
    "sense_id", "pos", "pt", "usage_hint", "hint_priority", "confidence",
    "reason", "risk_note", "model_id", "generated_at",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print diff stats but don't rewrite the TSV.")
    args = ap.parse_args()

    if not SIDECAR.exists():
        print(f"ERROR: missing {SIDECAR}", file=sys.stderr)
        return 1

    with SIDECAR.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    diffs = {
        "tier1_override_installed": 0,
        "essential_downgraded_to_useful": 0,
        "parenthetical_stripped": 0,
        "risk_note_rerouted": 0,
    }

    new_rows: list[dict] = []
    for row in rows:
        original_hint = (row.get("usage_hint") or "").strip()
        original_priority = row.get("hint_priority") or "omit"
        original_risk = (row.get("risk_note") or "").strip()
        lemma = row.get("pt", "")

        row.setdefault("risk_note", "")
        out = apply_post_process(dict(row))

        new_hint = out["usage_hint"]
        new_priority = out["hint_priority"]
        new_risk = out["risk_note"]

        if lemma in TIER_1_OVERRIDES and new_hint == TIER_1_OVERRIDES[lemma]:
            if original_hint != new_hint:
                diffs["tier1_override_installed"] += 1
        if (original_priority == "essential" and new_priority == "useful"
                and lemma in ESSENTIAL_TO_USEFUL_LEMMAS):
            diffs["essential_downgraded_to_useful"] += 1
        if "(" in original_hint and "(" not in new_hint:
            diffs["parenthetical_stripped"] += 1
        if lemma in RISK_NOTE_OVERRIDES and not original_risk and new_risk:
            diffs["risk_note_rerouted"] += 1

        # Ensure all output fields present
        for k in POST_FIELDS:
            out.setdefault(k, "")
        new_rows.append({k: out.get(k, "") for k in POST_FIELDS})

    # Summary
    pri_counts: dict[str, int] = {"essential": 0, "useful": 0, "omit": 0}
    for r in new_rows:
        p = r.get("hint_priority", "")
        if p in pri_counts:
            pri_counts[p] += 1
    n_hint = pri_counts["essential"] + pri_counts["useful"]

    print("=== Stage 12.2 postprocess ===")
    for k, v in diffs.items():
        print(f"  {k:<40} {v}")
    print()
    print(f"  Priority after postprocess: essential={pri_counts['essential']}  "
          f"useful={pri_counts['useful']}  omit={pri_counts['omit']}")
    print(f"  Non-empty hints: {n_hint}/{len(new_rows)}")

    if args.dry_run:
        print("\n--dry-run: not writing TSV.")
        return 0

    with SIDECAR.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POST_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(new_rows)
    print(f"\nRewrote {SIDECAR} ({len(new_rows)} rows, {len(POST_FIELDS)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
