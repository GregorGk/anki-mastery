#!/usr/bin/env python
"""Re-apply deterministic checks to data/04-examples.tsv without re-calling the LLM.

Use after fixing build/lib/validate.py to surface the corrected token-match
status, then rebuild data/_example_fixes.tsv based on the new outcome.

Idempotent. No API costs.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.lib.validate import (  # noqa: E402
    example_within_word_limit,
    token_in_sentence,
)
from build.stage_4 import FIXES_FIELDS, OUTPUT_FIELDS  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
EXAMPLES_PATH = DATA_DIR / "04-examples.tsv"
FIXES_PATH = DATA_DIR / "_example_fixes.tsv"


def main() -> int:
    rows = read_tsv(EXAMPLES_PATH)
    if not rows:
        print("No examples to recheck.")
        return 0

    flipped_token = 0
    flipped_wc = 0
    fixes: list[dict] = []
    pass_count = 0
    fail_count = 0
    borderline_count = 0
    skipped_count = 0

    for r in rows:
        target = (r.get("target_word_used") or "").strip()
        ex_pt = r.get("example_pt") or ""

        new_token = bool(target) and token_in_sentence(target, ex_pt)
        new_wc = example_within_word_limit(ex_pt, limit=15)

        old_token = r.get("example_token_match", "") == "pass"
        old_wc = r.get("example_word_count_ok", "") == "pass"

        if new_token != old_token:
            flipped_token += 1
            r["example_token_match"] = "pass" if new_token else "fail"
        if new_wc != old_wc:
            flipped_wc += 1
            r["example_word_count_ok"] = "pass" if new_wc else "fail"

        val_status = r.get("example_validation_status", "")
        if val_status == "pass":
            pass_count += 1
        elif val_status == "fail":
            fail_count += 1
        elif val_status == "borderline":
            borderline_count += 1
        else:
            skipped_count += 1

        # Rebuild fixes queue with new deterministic status
        needs_fix = (
            (not new_token)
            or (not new_wc)
            or (val_status == "fail")
            or (
                val_status == "skipped"
                and r.get("example_method", "") != "manual_override"
            )
        )
        if needs_fix:
            failure_axes = []
            if val_status == "fail":
                # Re-derive axes from validator reason if available
                reason = r.get("example_validation_reason", "")
                # Soft heuristic: caller can re-validate if needed
                failure_axes.append("validator")
            if not new_token:
                failure_axes.append("token_match")
            if not new_wc:
                failure_axes.append("word_count")
            fixes.append(
                {
                    "sense_id": r["sense_id"],
                    "rank": r["rank"],
                    "pt": r["pt"],
                    "en_primary": r["en_primary"],
                    "example_pt": ex_pt,
                    "example_en": r.get("example_en", ""),
                    "target_word_used": target,
                    "failure_reason": r.get("example_validation_reason", "")
                    or ("token mismatch" if not new_token else "word count")
                    if not new_token or not new_wc
                    else "validator failure",
                    "failure_axes": ",".join(failure_axes),
                }
            )

    n_out = write_tsv(EXAMPLES_PATH, rows, fieldnames=OUTPUT_FIELDS)
    n_fix = write_tsv(FIXES_PATH, fixes, fieldnames=FIXES_FIELDS)

    print(f"Recheck complete on {n_out} rows.")
    print(f"  Token-match flipped:    {flipped_token}")
    print(f"  Word-count flipped:     {flipped_wc}")
    print(f"  Fixes queue rebuilt:    {n_fix} rows (was 76 before)")
    print(f"  Validator pass:         {pass_count}")
    print(f"  Validator fail:         {fail_count}")
    print(f"  Validator borderline:   {borderline_count}")
    print(f"  Validator skipped:      {skipped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
