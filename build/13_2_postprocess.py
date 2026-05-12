"""Stage 13 / Step 2 — Apply deterministic post-LLM passes to the sidecar.

Idempotent: safe to re-run. Reads data/_family_roots.tsv + data/06-final.tsv,
applies:

  1. Per-row idempotent rules from build/lib/family_root_rules.py
     (already applied at classifier-write time, re-applied here defensively).
  2. Deck-level cluster-size enforcement (≥ 2 distinct pt lemmas per root).
  3. Self-root propagation: every cluster head also gets family_root=head on
     its own row(s), provenance source='postprocess_self_root'.

Writes back the sidecar with the same SIDECAR_FIELDS schema as the
classifier. Self-root propagation may APPEND new rows for sense_ids that
weren't part of the LLM run.

Usage:
    .venv/bin/python build/13_2_postprocess.py
    .venv/bin/python build/13_2_postprocess.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.family_root_rules import (  # noqa: E402
    CONTENT_POS,
    apply_post_process,
    cluster_pt_counts,
    enforce_cluster_size,
    propagate_self_root,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
SIDECAR_TSV = DATA_DIR / "_family_roots.tsv"
FINAL_TSV = DATA_DIR / "06-final.tsv"

# Same column order as the classifier emits.
SIDECAR_FIELDS = [
    "sense_id", "pos", "pt", "en_primary",
    "family_root", "family_relation", "confidence", "reason",
    "source", "model_id", "generated_at",
]

POSTPROCESS_MODEL_TAG = "postprocess_self_root"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print diff stats but don't rewrite the TSV.")
    args = ap.parse_args()

    if not SIDECAR_TSV.exists():
        print(f"ERROR: missing {SIDECAR_TSV}", file=sys.stderr)
        return 1
    if not FINAL_TSV.exists():
        print(f"ERROR: missing {FINAL_TSV}", file=sys.stderr)
        return 1

    sidecar_rows = read_tsv(SIDECAR_TSV)
    deck_rows = read_tsv(FINAL_TSV)

    # Index the deck by pt for fast lookups during self-root propagation.
    deck_rows_by_pt: dict[str, list[dict]] = defaultdict(list)
    for r in deck_rows:
        pt = (r.get("pt") or "").strip()
        if pt:
            deck_rows_by_pt[pt].append(r)

    all_pts: set[str] = set(deck_rows_by_pt.keys())

    # --- Pass 1: per-row idempotent rules (already applied at write time,
    # but re-applying is safe and catches manual edits).
    initial_root_counts: dict[str, int] = {"set": 0, "empty": 0}
    for r in sidecar_rows:
        if (r.get("family_root") or "").strip():
            initial_root_counts["set"] += 1
        else:
            initial_root_counts["empty"] += 1
    for r in sidecar_rows:
        r.setdefault("family_relation", "")
        r.setdefault("confidence", "")
        r.setdefault("reason", "")
        r.setdefault("source", "llm")
        apply_post_process(r, all_pts)

    after_perrow = sum(1 for r in sidecar_rows
                       if (r.get("family_root") or "").strip())

    # --- Pass 2: deck-level cluster-size enforcement.
    cleared_by_cluster = enforce_cluster_size(sidecar_rows, min_distinct_pts=2)
    after_cluster = sum(1 for r in sidecar_rows
                        if (r.get("family_root") or "").strip())

    # --- Pass 3: self-root propagation. For every surviving cluster head Y,
    # ensure every deck row with pt=Y has a sidecar entry pointing to Y.
    new_rows = propagate_self_root(
        sidecar_rows,
        deck_rows_by_pt=deck_rows_by_pt,
        timestamp_iso=_now_iso(),
        model_id=POSTPROCESS_MODEL_TAG,
    )
    sidecar_rows.extend(new_rows)

    # Final cluster stats.
    final_clusters = cluster_pt_counts(sidecar_rows)
    final_cluster_count = len(final_clusters)
    cluster_size_dist: dict[int, int] = defaultdict(int)
    for pts in final_clusters.values():
        cluster_size_dist[len(pts)] += 1

    by_source: dict[str, int] = defaultdict(int)
    by_relation: dict[str, int] = defaultdict(int)
    n_shipped = 0
    for r in sidecar_rows:
        root = (r.get("family_root") or "").strip()
        if root:
            n_shipped += 1
            by_source[r.get("source", "")] += 1
            by_relation[r.get("family_relation", "")] += 1

    print("=== Stage 13.2 postprocess ===")
    print(f"  Initial sidecar rows:               {len(sidecar_rows) - len(new_rows)}")
    print(f"    with non-empty family_root:       {initial_root_counts['set']}")
    print(f"    empty family_root:                {initial_root_counts['empty']}")
    print(f"  After per-row idempotent pass:      {after_perrow}")
    print(f"  Cleared by cluster-size rule:       {cleared_by_cluster}")
    print(f"  Survived cluster-size rule:         {after_cluster}")
    print(f"  Self-root propagation appended:     {len(new_rows)} new rows")
    print(f"  Final sidecar rows:                 {len(sidecar_rows)}")
    print(f"  Final shipped roots (total):        {n_shipped}")
    print(f"  Distinct clusters:                  {final_cluster_count}")
    print(f"  Cluster size distribution:")
    for size in sorted(cluster_size_dist.keys()):
        print(f"    size={size:>2}: {cluster_size_dist[size]} clusters")
    print(f"  by source:   " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    print(f"  by relation: " + ", ".join(f"{k}={v}" for k, v in sorted(by_relation.items())))

    if args.dry_run:
        print("\n--dry-run: not writing TSV.")
        return 0

    # Sort by sense_id for deterministic output.
    sidecar_rows.sort(key=lambda r: r.get("sense_id", ""))

    with SIDECAR_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIDECAR_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL,
                           extrasaction="ignore")
        w.writeheader()
        for r in sidecar_rows:
            sanitized = {
                k: ("" if r.get(k) is None else str(r.get(k, "")))
                for k in SIDECAR_FIELDS
            }
            w.writerow(sanitized)
    print(f"\nRewrote {SIDECAR_TSV} ({len(sidecar_rows)} rows, {len(SIDECAR_FIELDS)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
