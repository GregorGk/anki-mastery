"""Surgical 1:1 voice substitution in data/045-speaker_gender.tsv.

Replaces voice_id values per a user-supplied OLD:NEW mapping. ONLY touches
rows where voice_id is in the mapping; every other row is preserved
byte-for-byte. Does NOT re-run the seeded shuffle, does NOT demote
confidence buckets, does NOT change voice_gender_assigned.

Use this for a clean 1:1 voice swap where the new voice replaces the old
at the same pool_index in `config/voices.tsv` (preserving gender). For
adding/removing voices from the pool — which forces re-shuffle — use
`build/recompute_voice_assignment.py` instead.

Usage:
    .venv/bin/python build/swap_voices_surgical.py \\
        --swap uju3wxzG5OhpWcoi3SMy:4r3G9XKliGgVZLKMgjik \\
        --swap 4za2kOXGgUd57HRSQ1fn:AaeZyyi87RCxtFnHPS3e \\
        --confirm

    # dry run (no write):
    .venv/bin/python build/swap_voices_surgical.py \\
        --swap OLD:NEW

Idempotent: running again with the same mapping after a successful swap
is a no-op (the new voice_ids are not in the mapping's "OLD" set).

Exit codes:
    0   success (or dry run)
    1   bad arguments
    2   would-overwrite without --confirm (after a non-zero diff)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
SPEAKER_GENDER_PATH = DATA_DIR / "045-speaker_gender.tsv"


def parse_swap(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--swap value {s!r} must be OLD:NEW (got no colon)"
        )
    old, new = s.split(":", 1)
    old, new = old.strip(), new.strip()
    if not old or not new:
        raise argparse.ArgumentTypeError(
            f"--swap value {s!r} has empty side"
        )
    if old == new:
        raise argparse.ArgumentTypeError(
            f"--swap value {s!r} maps voice to itself"
        )
    return old, new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--swap",
        type=parse_swap,
        action="append",
        required=True,
        help="OLD:NEW voice ID pair (repeat for multiple swaps in one run)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually write the file. Omit to dry-run.",
    )
    parser.add_argument(
        "--path",
        default=str(SPEAKER_GENDER_PATH),
        help=f"Path to speaker_gender TSV (default: {SPEAKER_GENDER_PATH})",
    )
    args = parser.parse_args()

    swaps: dict[str, str] = dict(args.swap)
    if len(swaps) != len(args.swap):
        parser.error("duplicate --swap OLD entries (each OLD must be unique)")

    new_set = set(swaps.values())
    if new_set & set(swaps):
        parser.error(
            "swap chain detected: a NEW voice_id appears as an OLD elsewhere "
            "(would lose information). Aborting."
        )

    path = Path(args.path)
    if not path.exists():
        parser.error(f"speaker_gender file not found: {path}")

    rows = read_tsv(path)
    if not rows:
        parser.error(f"speaker_gender file is empty: {path}")

    if "voice_id" not in rows[0]:
        parser.error(f"no voice_id column in {path}")

    # Apply swap
    per_pair_counts: dict[tuple[str, str], int] = {(o, n): 0 for o, n in swaps.items()}
    already_swapped_count = 0
    untouched_count = 0
    for row in rows:
        vid = row["voice_id"]
        if vid in swaps:
            row["voice_id"] = swaps[vid]
            per_pair_counts[(vid, swaps[vid])] += 1
        elif vid in new_set:
            already_swapped_count += 1
        else:
            untouched_count += 1

    total_changed = sum(per_pair_counts.values())
    total_rows = len(rows)

    print(f"Surgical voice swap on {path}:")
    for (old, new), n in per_pair_counts.items():
        print(f"  {old} → {new}:  {n} rows")
    print(f"  already-swapped (no-op): {already_swapped_count}")
    print(f"  untouched:               {untouched_count}")
    print(f"  total rows in file:      {total_rows}")
    print(f"  total rows changed now:  {total_changed}")

    # Sanity: untouched + already + changed should equal total
    assert already_swapped_count + untouched_count + total_changed == total_rows, (
        f"row accounting mismatch: "
        f"already={already_swapped_count} untouched={untouched_count} "
        f"changed={total_changed} total={total_rows}"
    )

    if total_changed == 0:
        print()
        print("Nothing to do (no rows match any OLD voice_id). File untouched.")
        return 0

    if not args.confirm:
        print()
        print("DRY RUN — no file written. Pass --confirm to apply.")
        return 0

    # Write back, preserving column order from the original file
    fieldnames = list(rows[0].keys())
    write_tsv(path, rows, fieldnames=fieldnames)
    print()
    print(f"Wrote {total_rows} rows to {path} (columns: {len(fieldnames)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
