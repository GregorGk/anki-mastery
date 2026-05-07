"""Stage 1b — Orthographic normalization (spelling + hyphen).

Reads:
    data/_source_ledger.tsv               (output of Stage 1a)
    data/_ep_spelling_map.tsv             (hand-curated; ~28 rows)
    data/_hyphen_rules.tsv                (hand-curated; preserve + dehyphenate)

Writes:
    data/01-normalized.tsv                (full normalized rows)
    data/_source_ledger.tsv               (in-place: append to normalization_action)

Behaviour:
    1. For each ledger row whose action ∈ {keep, expand_idiom_added}:
       - Apply EP→BP spelling map (whole-word).
       - Apply hyphen rules (literal dehyphenate first, regex preserve second).
       - Set `pt` (normalized) and `pt_type` (single_word / hyphenated_compound /
         space_compound).
    2. Update ledger `normalization_action`: `none` → `spelling`, `hyphen`,
       `idiom_expansion`, or composables (comma-joined).
    3. Write `01-normalized.tsv` with the schema downstream stages expect.

Lexical replacement (`comboio` → `trem`) is Stage 1c, not here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.bp_rules import (  # noqa: E402
    load_hyphen_rules,
    load_spelling_map,
    normalize_pt,
)
from build.lib.ledger import LEDGER_FIELDS, write_ledger, LedgerRow  # noqa: E402
from build.lib.parse import detect_pt_type  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

LEDGER_PATH = DATA_DIR / "_source_ledger.tsv"
SPELLING_MAP_PATH = DATA_DIR / "_ep_spelling_map.tsv"
HYPHEN_RULES_PATH = DATA_DIR / "_hyphen_rules.tsv"
OUTPUT_PATH = DATA_DIR / "01-normalized.tsv"

NORMALIZED_FIELDS = [
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
    "source_line",
]


def _compose_normalization_action(existing: str, new_actions: list[str]) -> str:
    """Compose a comma-joined normalization_action string.

    `existing` is the current ledger value (`none`, `idiom_expansion`, or composed).
    `new_actions` is the Stage 1b list (spelling, hyphen).
    """
    parts = []
    if existing and existing != "none":
        parts.extend([p.strip() for p in existing.split(",") if p.strip()])
    for action in new_actions:
        if action not in parts:
            parts.append(action)
    return ",".join(parts) if parts else "none"


def _enumerate_expansion_indices(rows: list[dict]) -> list[int]:
    """Compute per-row expansion_index based on action and source_line_number.

    Original rows (action != expand_idiom_added) get expansion_index=0.
    Idiom expansions get 1, 2, ... in order of appearance per source_line.
    """
    counters: dict[int, int] = {}
    out: list[int] = []
    for row in rows:
        ln = int(row["source_line_number"])
        if row.get("action") == "expand_idiom_added":
            counters[ln] = counters.get(ln, 0) + 1
            out.append(counters[ln])
        else:
            out.append(0)
    return out


def run(
    *,
    ledger_path: Path = LEDGER_PATH,
    spelling_map_path: Path = SPELLING_MAP_PATH,
    hyphen_rules_path: Path = HYPHEN_RULES_PATH,
    output_path: Path = OUTPUT_PATH,
    update_ledger: bool = True,
) -> dict:
    """Run Stage 1b end-to-end. Returns a summary dict."""
    if not ledger_path.exists():
        raise FileNotFoundError(
            f"Ledger not found at {ledger_path}. Run Stage 1a first."
        )

    spelling_map = load_spelling_map(spelling_map_path)
    hyphen_rules = load_hyphen_rules(hyphen_rules_path)

    ledger_rows = read_tsv(ledger_path)
    expansion_indices = _enumerate_expansion_indices(ledger_rows)

    normalized_rows: list[dict] = []
    spelling_hits = 0
    hyphen_hits = 0
    skipped = 0
    updated_ledger: list[LedgerRow] = []

    for row, exp_idx in zip(ledger_rows, expansion_indices):
        action = row.get("action", "keep")
        # Carry-through for non-active rows
        if action in {"manual_review", "drop_ep_only"}:
            updated_ledger.append(_row_to_ledger(row))
            skipped += 1
            continue

        source_pt = row.get("source_pt", "").strip()
        if not source_pt:
            updated_ledger.append(_row_to_ledger(row))
            skipped += 1
            continue

        normalized_pt, actions = normalize_pt(source_pt, spelling_map, hyphen_rules)
        if "spelling" in actions:
            spelling_hits += 1
        if "hyphen" in actions:
            hyphen_hits += 1

        # Build the normalized output row.
        normalized_rows.append(
            {
                "source_line_number": row["source_line_number"],
                "rank": row["rank"],
                "expansion_index": exp_idx,
                "source_pt": source_pt,
                "pt": normalized_pt,
                "pt_type": detect_pt_type(normalized_pt),
                "gender": "",
                "pos": "",
                "en_all": row.get("source_en_all", ""),
                "annotation": "",  # Stage 2 will populate from the parser's structured annotation
                "normalization_action": _compose_normalization_action(
                    row.get("normalization_action", "none"), actions
                ),
                "source_line": row.get("source_raw", ""),
            }
        )

        # Update ledger row in memory: composed normalization_action
        new_norm_action = _compose_normalization_action(
            row.get("normalization_action", "none"), actions
        )
        updated_ledger.append(
            _row_to_ledger({**row, "normalization_action": new_norm_action})
        )

    # Write the normalized output
    out_count = write_tsv(output_path, normalized_rows, fieldnames=NORMALIZED_FIELDS)

    # Optionally rewrite the ledger in place
    if update_ledger:
        write_ledger(ledger_path, updated_ledger)

    # --- Hard invariants ----------------------------------------------------
    # No `=` in any pt of an active row
    for r in normalized_rows:
        if "=" in r["pt"]:
            raise AssertionError(
                f"Line {r['source_line_number']}: pt contains '=': {r['pt']!r}"
            )
    # (rank, expansion_index) uniqueness across active rows
    seen_keys: set[tuple[int, int]] = set()
    for r in normalized_rows:
        key = (int(r["rank"]), int(r["expansion_index"]))
        if key in seen_keys:
            raise AssertionError(
                f"Duplicate (rank={key[0]}, expansion_index={key[1]}) in 01-normalized.tsv"
            )
        seen_keys.add(key)
    # Ranks non-decreasing
    last_rank = -1
    for r in normalized_rows:
        rank = int(r["rank"])
        if rank < last_rank:
            raise AssertionError(
                f"Rank order violated: rank {rank} after {last_rank}"
            )
        last_rank = rank

    return {
        "ledger_rows_read": len(ledger_rows),
        "active_rows_normalized": len(normalized_rows),
        "rows_skipped": skipped,
        "spelling_rules_applied": spelling_hits,
        "hyphen_rules_applied": hyphen_hits,
        "output_rows": out_count,
    }


def _row_to_ledger(row: dict) -> LedgerRow:
    """Coerce a dict-row (from read_tsv) into a LedgerRow."""
    return LedgerRow(
        source_line_number=int(row["source_line_number"]),
        rank=int(row["rank"]),
        source_raw=row.get("source_raw", ""),
        source_pt=row.get("source_pt", ""),
        source_en_all=row.get("source_en_all", ""),
        action=row.get("action", "keep"),
        normalization_action=row.get("normalization_action", "none"),
        bp_replacement=row.get("bp_replacement", ""),
        merge_target_rank=row.get("merge_target_rank", ""),
        merge_target_pt=row.get("merge_target_pt", ""),
        output_sense_ids=row.get("output_sense_ids", ""),
        drop_reason=row.get("drop_reason", ""),
        manual_review_status=row.get("manual_review_status", "not_required"),
        stage_decided=row.get("stage_decided", "1a"),
        notes=row.get("notes", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1b — orthographic normalization (spelling + hyphen)"
    )
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--spelling-map", type=Path, default=SPELLING_MAP_PATH)
    parser.add_argument("--hyphen-rules", type=Path, default=HYPHEN_RULES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--no-ledger-update",
        action="store_true",
        help="Do not rewrite _source_ledger.tsv (useful for testing)",
    )
    args = parser.parse_args()

    summary = run(
        ledger_path=args.ledger,
        spelling_map_path=args.spelling_map,
        hyphen_rules_path=args.hyphen_rules,
        output_path=args.output,
        update_ledger=not args.no_ledger_update,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
