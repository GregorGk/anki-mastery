"""Stage 1.8 — Post-normalization duplicate detection.

Reads:
    data/015-bp_status.tsv                (Stage 1.5 output)

Writes:
    data/018-deduped.tsv                  (active rows; deterministic dedupe applied)
    data/_normalized_duplicates.tsv       (audit log of any duplicates found)

Behaviour:
    1. Group rows by `pt`.
    2. For each cluster (≥2 rows sharing the same `pt`), categorize:
       - `exact_duplicate`         all en_all values identical → merge, keep first
       - `same_headword_different_senses`  glosses overlap → pass-through; Stage 2
                                            will split into distinct sense_ids
       - `ambiguous_collision`     glosses don't overlap → flag manual_review
    3. Write 018-deduped.tsv (same schema as input + a `dedupe_action` column).
    4. Write _normalized_duplicates.tsv with one row per cluster.

For the current corpus this stage finds 0 duplicates because Stages 1b/1c
already produced unique pts. The stage exists as a sanity gate before Stage
2 — if a future re-run with different inputs produces duplicates, this is
where they get caught and either auto-resolved or routed for review.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.lexical import gloss_similarity  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

INPUT_PATH = DATA_DIR / "015-bp_status.tsv"
OUTPUT_PATH = DATA_DIR / "018-deduped.tsv"
DUPLICATES_PATH = DATA_DIR / "_normalized_duplicates.tsv"

DUPLICATES_FIELDS = [
    "pt",
    "cluster_size",
    "ranks",
    "source_pts",
    "source_lines",
    "en_all_values",
    "duplicate_type",
    "suggested_action",
    "manual_status",
    "notes",
]

DUP_TYPE_EXACT = "exact_duplicate"
DUP_TYPE_SAME_HEADWORD_SENSES = "same_headword_different_senses"
DUP_TYPE_AMBIGUOUS = "ambiguous_collision"


def categorize_cluster(rows: list[dict]) -> str:
    """Classify a cluster of rows sharing the same `pt`."""
    if len(rows) < 2:
        return "single"

    en_alls = [r.get("en_all", "") for r in rows]
    if len(set(en_alls)) == 1:
        return DUP_TYPE_EXACT

    # Pairwise gloss similarity
    pairwise = []
    for i in range(len(en_alls)):
        for j in range(i + 1, len(en_alls)):
            pairwise.append(gloss_similarity(en_alls[i], en_alls[j]))
    if pairwise and all(s.is_similar for s in pairwise):
        return DUP_TYPE_SAME_HEADWORD_SENSES
    if pairwise and any(s.is_similar for s in pairwise):
        # Mixed: at least one pair similar, at least one not.
        return DUP_TYPE_AMBIGUOUS
    return DUP_TYPE_AMBIGUOUS


def run(
    *,
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    duplicates_path: Path = DUPLICATES_PATH,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found at {input_path}. Run Stage 1.5 first."
        )

    rows = read_tsv(input_path)

    # Group by pt
    by_pt: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_pt[r["pt"]].append(i)

    duplicate_clusters = {pt: idxs for pt, idxs in by_pt.items() if len(idxs) > 1}

    # Build duplicates audit
    duplicate_audit: list[dict] = []
    drop_indices: set[int] = set()
    cluster_action_per_index: dict[int, str] = {}

    counts: dict[str, int] = {
        DUP_TYPE_EXACT: 0,
        DUP_TYPE_SAME_HEADWORD_SENSES: 0,
        DUP_TYPE_AMBIGUOUS: 0,
    }

    for pt, idxs in duplicate_clusters.items():
        cluster = [rows[i] for i in idxs]
        dup_type = categorize_cluster(cluster)
        counts[dup_type] = counts.get(dup_type, 0) + 1

        if dup_type == DUP_TYPE_EXACT:
            # Keep first, mark others for drop
            for i in idxs[1:]:
                drop_indices.add(i)
                cluster_action_per_index[i] = "drop_exact_duplicate"
            cluster_action_per_index[idxs[0]] = "kept_exact_duplicate"
            suggested = "drop_others"
        elif dup_type == DUP_TYPE_SAME_HEADWORD_SENSES:
            # Pass through; Stage 2 will assign distinct sense_index per row
            for i in idxs:
                cluster_action_per_index[i] = "pass_through_for_sense_split"
            suggested = "pass_through"
        else:
            # Ambiguous: flag all for manual review
            for i in idxs:
                cluster_action_per_index[i] = "manual_review"
            suggested = "manual_review"

        duplicate_audit.append(
            {
                "pt": pt,
                "cluster_size": len(idxs),
                "ranks": ",".join(str(rows[i].get("rank", "")) for i in idxs),
                "source_pts": ",".join(rows[i].get("source_pt", "") for i in idxs),
                "source_lines": ",".join(
                    str(rows[i].get("source_line_number", "")) for i in idxs
                ),
                "en_all_values": " || ".join(rows[i].get("en_all", "") for i in idxs),
                "duplicate_type": dup_type,
                "suggested_action": suggested,
                "manual_status": "pending" if suggested == "manual_review" else "auto",
                "notes": "",
            }
        )

    # Build output
    output_fields = list(rows[0].keys()) if rows else []
    if "dedupe_action" not in output_fields:
        output_fields.append("dedupe_action")

    output_rows: list[dict] = []
    for i, r in enumerate(rows):
        if i in drop_indices:
            continue
        out = dict(r)
        out["dedupe_action"] = cluster_action_per_index.get(i, "")
        output_rows.append(out)

    out_count = write_tsv(output_path, output_rows, fieldnames=output_fields)
    dup_count = write_tsv(duplicates_path, duplicate_audit, fieldnames=DUPLICATES_FIELDS)

    # --- Hard invariants ---
    # After dedupe, the only remaining same-pt clusters are
    # `same_headword_different_senses` (pass-through) or `ambiguous_collision`.
    pt_counts: dict[str, int] = defaultdict(int)
    for r in output_rows:
        pt_counts[r["pt"]] += 1
    bad = [
        pt for pt, n in pt_counts.items()
        if n > 1
        and pt in {row["pt"] for row in duplicate_audit
                   if row["duplicate_type"] == DUP_TYPE_EXACT}
    ]
    if bad:
        raise AssertionError(
            f"exact_duplicate clusters survived dedupe: {bad}"
        )

    return {
        "input_rows": len(rows),
        "output_rows": out_count,
        "duplicate_clusters": len(duplicate_clusters),
        "exact_duplicates": counts[DUP_TYPE_EXACT],
        "same_headword_clusters": counts[DUP_TYPE_SAME_HEADWORD_SENSES],
        "ambiguous_clusters": counts[DUP_TYPE_AMBIGUOUS],
        "rows_dropped_exact_duplicate": len(drop_indices),
        "rows_flagged_manual_review": sum(
            1 for v in cluster_action_per_index.values() if v == "manual_review"
        ),
        "duplicates_audit_rows": dup_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1.8 — post-normalization duplicate detection"
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--duplicates", type=Path, default=DUPLICATES_PATH)
    args = parser.parse_args()

    summary = run(
        input_path=args.input,
        output_path=args.output,
        duplicates_path=args.duplicates,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
