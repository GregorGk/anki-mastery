"""Stage 1c — Lexical EP→BP replacement + collision merge.

Reads:
    data/01-normalized.tsv                (Stage 1b output)
    data/_lexical_bp_replacements.tsv     (hand-curated; ~14 rows)
    data/_source_ledger.tsv               (mutated in place)
    data/_sense_source_map.tsv            (mutated in place)

Writes:
    data/012-lexical_replaced.tsv         (rows after replacement + merge)
    data/_ep_drop_or_replace_review.tsv   (manual queue for ambiguous merges)
    Updates ledger (action, bp_replacement, merge_target_*, drop_reason)
    Appends provenance edges to _sense_source_map.tsv (provenance_type=
        lexical_replacement | merged)

Behaviour:
    For each normalized row whose `pt` matches a `source_pt` in the
    replacement table:
      1. Compute target_pt = bp_replacement.
      2. Look up existing rows in 01-normalized.tsv with `pt == target_pt`
         (collision detection).
      3. If a collision exists:
           - Compute gloss-similarity (Jaccard over content tokens).
           - HIGH similarity (>= 1 overlapping token AND Jaccard >= 0.10):
                merge into the existing BP row. The EP row's en_all is
                appended to the BP row's en_all (with separator). Source
                edge added (provenance_type=merged). EP row dropped from
                output but kept in ledger with action=merge_into_existing_bp_row.
           - LOW similarity:
                both rows persist; both flagged manual_review; logged to
                _ep_drop_or_replace_review.tsv.
      4. If no collision: pt becomes the BP form, action becomes
         replace_with_bp_equivalent, edge added (provenance_type=
         lexical_replacement).

Hard invariants:
    - Every action='drop_ep_only' (none in 1c, but if any) has drop_reason.
    - Every action='replace_with_bp_equivalent' has bp_replacement.
    - Every action='merge_into_existing_bp_row' has merge_target_* fields.
    - 012-lexical_replaced.tsv has unique (rank, expansion_index).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.ledger import (  # noqa: E402
    ACTION_KEEP,
    ACTION_MANUAL_REVIEW,
    ACTION_MERGE_INTO_BP_ROW,
    ACTION_REPLACE_WITH_BP,
    LEDGER_FIELDS,
    PROVENANCE_LEXICAL_REPLACEMENT,
    PROVENANCE_MERGED,
    SENSE_SOURCE_MAP_FIELDS,
    LedgerRow,
    SenseSourceEdge,
    write_ledger,
)
from build.lib.lexical import (  # noqa: E402
    GlossSimilarity,
    gloss_similarity,
    load_lexical_replacements,
)
from build.lib.tsv import append_tsv, read_tsv, write_tsv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

NORMALIZED_PATH = DATA_DIR / "01-normalized.tsv"
LEXICAL_PATH = DATA_DIR / "_lexical_bp_replacements.tsv"
LEDGER_PATH = DATA_DIR / "_source_ledger.tsv"
SENSE_MAP_PATH = DATA_DIR / "_sense_source_map.tsv"
OUTPUT_PATH = DATA_DIR / "012-lexical_replaced.tsv"
REVIEW_PATH = DATA_DIR / "_ep_drop_or_replace_review.tsv"

OUTPUT_FIELDS = [
    "source_line_number",
    "rank",
    "expansion_index",
    "source_pt",
    "pt",
    "pt_type",
    "gender",
    "pos",
    "en_all",
    "annotation",
    "normalization_action",
    "bp_replacement",
    "merge_target_rank",
    "merge_target_pt",
    "merged_from_source_lines",
    "source_variants",
    "source_line",
]

REVIEW_FIELDS = [
    "source_line_number",
    "rank",
    "ep_pt",
    "bp_target_pt",
    "ep_en_all",
    "bp_en_all",
    "overlap_tokens",
    "jaccard_score",
    "suggested_action",
    "manual_status",
    "notes",
]


def _glosses_join(en_a: str, en_b: str) -> str:
    """Union two glosses for the surviving merge target. Keeps order, dedupes."""
    if not en_a:
        return en_b
    if not en_b:
        return en_a
    if en_a == en_b:
        return en_a
    # Simple concat with separator for audit transparency.
    return f"{en_a} | {en_b}"


def run(
    *,
    normalized_path: Path = NORMALIZED_PATH,
    lexical_path: Path = LEXICAL_PATH,
    ledger_path: Path = LEDGER_PATH,
    sense_map_path: Path = SENSE_MAP_PATH,
    output_path: Path = OUTPUT_PATH,
    review_path: Path = REVIEW_PATH,
    update_ledger: bool = True,
    update_sense_map: bool = True,
) -> dict:
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"01-normalized.tsv not found at {normalized_path}. Run Stage 1b first."
        )

    normalized_rows = read_tsv(normalized_path)
    rules = load_lexical_replacements(lexical_path)

    # Index by pt for collision detection. Same `pt` may appear multiple times
    # (e.g., from idiom expansions sharing a normalized form, or future
    # repeats); we collect all entries.
    pt_index: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(normalized_rows):
        pt_index[row["pt"]].append(i)

    # Load existing ledger so we can mutate and rewrite atomically.
    ledger_rows = read_tsv(ledger_path)
    # Index ledger rows by (source_line_number, expansion_index) for updates.
    # NOTE: ledger doesn't currently have expansion_index; we derive it via
    # the same enumeration logic Stage 1b used.
    from build.stage_1b import _enumerate_expansion_indices

    ledger_exp_indices = _enumerate_expansion_indices(ledger_rows)
    ledger_by_key: dict[tuple[int, int], int] = {}
    for i, lrow in enumerate(ledger_rows):
        ledger_by_key[(int(lrow["source_line_number"]), ledger_exp_indices[i])] = i

    # Track changes
    output_rows: list[dict] = []
    review_rows: list[dict] = []
    new_edges: list[SenseSourceEdge] = []
    stats = {
        "replacements_applied": 0,
        "merges_applied": 0,
        "manual_review_collisions": 0,
        "rules_with_no_match": 0,
    }

    # Collect indices that get absorbed (consumed) into a merge target.
    consumed_indices: set[int] = set()
    # Track gloss accumulations for survivors: index → en_all updated value
    survivor_gloss_updates: dict[int, str] = {}
    # Track variants accumulation: index → list of source_pts merged in
    survivor_variants: dict[int, list[str]] = defaultdict(list)
    # Track merged-from source lines: index → list of source_line_numbers
    survivor_merged_lines: dict[int, list[int]] = defaultdict(list)

    # First pass: classify each row.
    classifications: list[dict] = []  # one entry per normalized row
    for i, row in enumerate(normalized_rows):
        pt = row["pt"]
        rule = rules.get(pt)
        if rule is None:
            classifications.append({"action": "passthrough", "row_index": i})
            continue

        target_pt = rule.bp_replacement
        # Find existing rows with pt == target_pt (excluding self).
        existing = [j for j in pt_index.get(target_pt, []) if j != i]

        if not existing:
            # No collision: simple replacement.
            classifications.append(
                {
                    "action": "replace",
                    "row_index": i,
                    "target_pt": target_pt,
                    "rule": rule,
                }
            )
            continue

        # Collision: pick the first existing row whose gloss is most similar.
        best_target: int | None = None
        best_sim: GlossSimilarity | None = None
        for j in existing:
            sim = gloss_similarity(row["en_all"], normalized_rows[j]["en_all"])
            if best_sim is None or sim.score > best_sim.score:
                best_target = j
                best_sim = sim

        if best_sim is not None and best_sim.is_similar and best_target is not None:
            classifications.append(
                {
                    "action": "merge",
                    "row_index": i,
                    "target_index": best_target,
                    "target_pt": target_pt,
                    "rule": rule,
                    "similarity": best_sim,
                }
            )
        else:
            classifications.append(
                {
                    "action": "manual_review",
                    "row_index": i,
                    "target_index": best_target,
                    "target_pt": target_pt,
                    "rule": rule,
                    "similarity": best_sim,
                }
            )

    # Second pass: apply classifications.
    rules_hit: set[str] = set()
    for cls in classifications:
        i = cls["row_index"]
        row = normalized_rows[i]

        if cls["action"] == "passthrough":
            continue

        if cls["action"] == "replace":
            rule = cls["rule"]
            rules_hit.add(rule.source_pt)
            row["pt"] = cls["target_pt"]
            row["bp_replacement"] = cls["target_pt"]
            row["normalization_action"] = _compose_norm(
                row.get("normalization_action", "none"), "lexical"
            )
            stats["replacements_applied"] += 1
            new_edges.append(
                SenseSourceEdge(
                    sense_id=_placeholder_sense_id(row),
                    source_line_number=int(row["source_line_number"]),
                    provenance_type=PROVENANCE_LEXICAL_REPLACEMENT,
                    notes=f"{rule.source_pt} -> {cls['target_pt']}",
                )
            )
            _update_ledger_row(
                ledger_rows,
                ledger_by_key,
                row,
                action=ACTION_REPLACE_WITH_BP,
                bp_replacement=cls["target_pt"],
                normalization_action_add="lexical",
                stage="1c",
            )
            continue

        if cls["action"] == "merge":
            rule = cls["rule"]
            rules_hit.add(rule.source_pt)
            target_idx = cls["target_index"]
            target_row = normalized_rows[target_idx]
            consumed_indices.add(i)

            # Accumulate survivor's en_all and variants.
            current_gloss = survivor_gloss_updates.get(
                target_idx, target_row["en_all"]
            )
            survivor_gloss_updates[target_idx] = _glosses_join(
                current_gloss, row["en_all"]
            )
            survivor_variants[target_idx].append(row["pt"])
            survivor_variants[target_idx].extend([row["source_pt"]])
            survivor_merged_lines[target_idx].append(int(row["source_line_number"]))

            stats["merges_applied"] += 1
            new_edges.append(
                SenseSourceEdge(
                    sense_id=_placeholder_sense_id(target_row),
                    source_line_number=int(row["source_line_number"]),
                    provenance_type=PROVENANCE_MERGED,
                    notes=f"merged from {rule.source_pt} (rank {row['rank']})",
                )
            )
            _update_ledger_row(
                ledger_rows,
                ledger_by_key,
                row,
                action=ACTION_MERGE_INTO_BP_ROW,
                bp_replacement="",
                merge_target_rank=int(target_row["rank"]),
                merge_target_pt=target_row["pt"],
                normalization_action_add="lexical",
                stage="1c",
            )
            continue

        if cls["action"] == "manual_review":
            rule = cls["rule"]
            rules_hit.add(rule.source_pt)
            sim = cls["similarity"]
            target_idx = cls["target_index"]
            target_row = normalized_rows[target_idx] if target_idx is not None else None
            review_rows.append(
                {
                    "source_line_number": row["source_line_number"],
                    "rank": row["rank"],
                    "ep_pt": row["pt"],
                    "bp_target_pt": cls["target_pt"],
                    "ep_en_all": row["en_all"],
                    "bp_en_all": target_row["en_all"] if target_row else "",
                    "overlap_tokens": ",".join(sorted(sim.overlap)) if sim else "",
                    "jaccard_score": f"{sim.score:.3f}" if sim else "",
                    "suggested_action": "merge_or_keep_separate",
                    "manual_status": "pending",
                    "notes": (
                        f"low gloss overlap with target row {target_row['rank']} "
                        f"({target_row['pt']})"
                        if target_row
                        else "no clear target"
                    ),
                }
            )
            stats["manual_review_collisions"] += 1
            _update_ledger_row(
                ledger_rows,
                ledger_by_key,
                row,
                action=ACTION_MANUAL_REVIEW,
                normalization_action_add="lexical",
                stage="1c",
                notes_extra="lexical replacement collision: low gloss overlap",
            )
            # The row STAYS in output for manual triage.

    # Build output rows: skip consumed rows; apply gloss/variant updates to survivors.
    for i, row in enumerate(normalized_rows):
        if i in consumed_indices:
            continue
        out_row = dict(row)  # shallow copy
        if i in survivor_gloss_updates:
            out_row["en_all"] = survivor_gloss_updates[i]
            out_row["source_variants"] = ", ".join(
                sorted(set(survivor_variants[i]))
            )
            out_row["merged_from_source_lines"] = ", ".join(
                str(x) for x in sorted(set(survivor_merged_lines[i]))
            )
        else:
            out_row["source_variants"] = out_row.get("source_variants", "")
            out_row["merged_from_source_lines"] = out_row.get(
                "merged_from_source_lines", ""
            )
        # bp_replacement carries through if set during replace classification
        out_row["bp_replacement"] = out_row.get("bp_replacement", "")
        # merge_target_* on the surviving row are blank (it's the target, not the merged-in)
        out_row["merge_target_rank"] = out_row.get("merge_target_rank", "")
        out_row["merge_target_pt"] = out_row.get("merge_target_pt", "")
        output_rows.append(out_row)

    # Track rules with no source match (pure no-ops in this corpus).
    for source_pt in rules:
        if source_pt not in rules_hit:
            stats["rules_with_no_match"] += 1

    # Write outputs.
    out_count = write_tsv(output_path, output_rows, fieldnames=OUTPUT_FIELDS)
    review_count = write_tsv(review_path, review_rows, fieldnames=REVIEW_FIELDS)

    if update_ledger:
        # Coerce ledger_rows back into LedgerRow instances and rewrite.
        coerced: list[LedgerRow] = []
        for r in ledger_rows:
            coerced.append(
                LedgerRow(
                    source_line_number=int(r["source_line_number"]),
                    rank=int(r["rank"]),
                    source_raw=r.get("source_raw", ""),
                    source_pt=r.get("source_pt", ""),
                    source_en_all=r.get("source_en_all", ""),
                    action=r.get("action", "keep"),
                    normalization_action=r.get("normalization_action", "none"),
                    bp_replacement=r.get("bp_replacement", ""),
                    merge_target_rank=r.get("merge_target_rank", ""),
                    merge_target_pt=r.get("merge_target_pt", ""),
                    output_sense_ids=r.get("output_sense_ids", ""),
                    drop_reason=r.get("drop_reason", ""),
                    manual_review_status=r.get("manual_review_status", "not_required"),
                    stage_decided=r.get("stage_decided", "1a"),
                    notes=r.get("notes", ""),
                )
            )
        write_ledger(ledger_path, coerced)

    if update_sense_map and new_edges:
        append_tsv(
            sense_map_path,
            (e.to_dict() for e in new_edges),
            fieldnames=SENSE_SOURCE_MAP_FIELDS,
        )

    # --- Hard invariants ----------------------------------------------------
    seen_keys: set[tuple[int, int]] = set()
    for r in output_rows:
        key = (int(r["rank"]), int(r["expansion_index"]))
        if key in seen_keys:
            raise AssertionError(f"Duplicate (rank, expansion_index): {key}")
        seen_keys.add(key)

    # Every replace_with_bp_equivalent has bp_replacement
    for lrow in ledger_rows:
        if lrow.get("action") == ACTION_REPLACE_WITH_BP:
            assert lrow.get("bp_replacement"), (
                f"line {lrow['source_line_number']}: replace_with_bp_equivalent "
                f"without bp_replacement"
            )
        if lrow.get("action") == ACTION_MERGE_INTO_BP_ROW:
            assert lrow.get("merge_target_rank") and lrow.get("merge_target_pt"), (
                f"line {lrow['source_line_number']}: merge_into_existing_bp_row "
                f"without merge_target_*"
            )

    return {
        "input_rows": len(normalized_rows),
        "output_rows": out_count,
        "rules_loaded": len(rules),
        **stats,
        "review_rows_logged": review_count,
        "new_provenance_edges": len(new_edges),
    }


def _compose_norm(existing: str, new: str) -> str:
    """Append `new` to comma-joined normalization_action string if not present."""
    parts = (
        [p.strip() for p in existing.split(",") if p.strip()]
        if existing and existing != "none"
        else []
    )
    if new not in parts:
        parts.append(new)
    return ",".join(parts) if parts else "none"


def _placeholder_sense_id(row: dict) -> str:
    """Compute the placeholder sense_id `RRRR.EE.00` from a normalized row."""
    rank = int(row["rank"])
    exp = int(row.get("expansion_index", 0))
    return f"{rank:04d}.{exp:02d}.00"


def _update_ledger_row(
    ledger_rows: list[dict],
    ledger_by_key: dict[tuple[int, int], int],
    normalized_row: dict,
    *,
    action: str,
    bp_replacement: str = "",
    merge_target_rank: int | str = "",
    merge_target_pt: str = "",
    normalization_action_add: str = "",
    stage: str = "1c",
    notes_extra: str = "",
) -> None:
    """Mutate the ledger row corresponding to a normalized row."""
    key = (
        int(normalized_row["source_line_number"]),
        int(normalized_row.get("expansion_index", 0)),
    )
    idx = ledger_by_key.get(key)
    if idx is None:
        return
    lrow = ledger_rows[idx]
    lrow["action"] = action
    if bp_replacement:
        lrow["bp_replacement"] = bp_replacement
    if merge_target_rank != "":
        lrow["merge_target_rank"] = str(merge_target_rank)
    if merge_target_pt:
        lrow["merge_target_pt"] = merge_target_pt
    if normalization_action_add:
        lrow["normalization_action"] = _compose_norm(
            lrow.get("normalization_action", "none"), normalization_action_add
        )
    lrow["stage_decided"] = stage
    if notes_extra:
        existing_notes = lrow.get("notes", "")
        lrow["notes"] = f"{existing_notes}; {notes_extra}".strip("; ")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1c — lexical EP→BP replacement with collision merge"
    )
    parser.add_argument("--normalized", type=Path, default=NORMALIZED_PATH)
    parser.add_argument("--lexical", type=Path, default=LEXICAL_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--sense-map", type=Path, default=SENSE_MAP_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--review", type=Path, default=REVIEW_PATH)
    parser.add_argument(
        "--no-ledger-update",
        action="store_true",
        help="Do not rewrite the ledger (test mode)",
    )
    parser.add_argument(
        "--no-sense-map-update",
        action="store_true",
        help="Do not append to _sense_source_map.tsv (test mode)",
    )
    args = parser.parse_args()

    summary = run(
        normalized_path=args.normalized,
        lexical_path=args.lexical,
        ledger_path=args.ledger,
        sense_map_path=args.sense_map,
        output_path=args.output,
        review_path=args.review,
        update_ledger=not args.no_ledger_update,
        update_sense_map=not args.no_sense_map_update,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
