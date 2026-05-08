#!/usr/bin/env python
"""Re-derive tags + pt_display in data/03-enriched.tsv without calling the LLM.

Use this when the deterministic logic in build/lib/enrich.py changes
(tag taxonomy, regional list, pt_display formatting). Reads the existing
03-enriched.tsv, recomputes `tags` and `pt_display` per row, writes back.

Idempotent. No API costs.

Run:
    python3 build/rederive_tags.py

Note: this does NOT change `gender`, `pos`, or `is_cognate_en` — those
come from the LLM (or `_manual_gender.tsv`) and are preserved verbatim.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.enrich import compose_pt_display, derive_tags  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

ENRICHED_PATH = REPO_ROOT / "data" / "03-enriched.tsv"


def main() -> int:
    rows = read_tsv(ENRICHED_PATH)
    if not rows:
        print(f"ERROR: {ENRICHED_PATH} not found or empty", file=sys.stderr)
        return 1

    # Preserve column order from input
    fieldnames = list(rows[0].keys())

    changed_tags = 0
    changed_display = 0

    for r in rows:
        # Coerce is_cognate_en string -> bool
        cognate_str = (r.get("is_cognate_en") or "").strip().lower()
        cognate_en = cognate_str in ("true", "1", "yes")

        gender = (r.get("gender") or "").strip()
        pos = (r.get("pos") or "").strip()
        pt = (r.get("pt") or "").strip()
        pt_type = (r.get("pt_type") or "").strip()

        new_tags = derive_tags(r, gender=gender, pos=pos, cognate_en=cognate_en)
        if new_tags != (r.get("tags") or ""):
            changed_tags += 1
            r["tags"] = new_tags

        new_display = compose_pt_display(pt, gender, pt_type)
        if new_display != (r.get("pt_display") or ""):
            changed_display += 1
            r["pt_display"] = new_display

    written = write_tsv(ENRICHED_PATH, rows, fieldnames=fieldnames)
    print(f"Wrote {written} rows to {ENRICHED_PATH.relative_to(REPO_ROOT)}")
    print(f"  tags changed:       {changed_tags}")
    print(f"  pt_display changed: {changed_display}")

    # Sanity: print tag counts for the previously-broken tags
    from collections import Counter

    tag_counts: Counter[str] = Counter()
    for r in rows:
        for t in (r.get("tags") or "").split():
            tag_counts[t] += 1

    print("\n=== Tag counts (key categories) ===")
    for tag in [
        "#single-word",
        "#hyphenated-compound",
        "#space-compound",
        "#abbreviation-expansion",
        "#idiom",
        "#reflexive",
        "#regional",
        "#gendered-meaning",
        "#function-word",
        "#pronoun",
        "#bp-rare",
        "#nsfw",
        "#false-friend",
        "#cognate-en",
    ]:
        print(f"  {tag:30s} {tag_counts.get(tag, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
