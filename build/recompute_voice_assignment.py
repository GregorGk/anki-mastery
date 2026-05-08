#!/usr/bin/env python
"""Re-apply Stage 4.5 Phase 2 (gender resolution + voice round-robin) without
calling the LLM.

Reads the existing classifications from data/045-speaker_gender.tsv and
re-derives `voice_gender_assigned` and `voice_id` using the current Phase 2
logic in build/stage_45.py (including the confidence policy: only `high`
confidence keeps explicit gender; medium/low are demoted to neutral).

Use this after:
- Updating the confidence policy in stage_45.py
- Editing config/voices.tsv (added/removed/reassigned voice)
- Editing data/_manual_speaker_gender.tsv

Idempotent. No API costs.

Run:
    python3 build/recompute_voice_assignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.lib.voices import (  # noqa: E402
    SEED_FEMALE_POOL,
    SEED_MALE_POOL,
    SEED_NEUTRAL_BALANCE,
    assign_voice_ids,
    load_voices,
    resolve_neutrals,
    split_by_gender,
    usage_counts,
    verify_balanced_within_pool,
)
from build.stage_45 import OUTPUT_FIELDS, _load_overrides  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"

EXISTING_PATH = DATA_DIR / "045-speaker_gender.tsv"
OVERRIDES_PATH = DATA_DIR / "_manual_speaker_gender.tsv"
VOICES_PATH = CONFIG_DIR / "voices.tsv"


def main() -> int:
    if not EXISTING_PATH.exists():
        print(
            f"ERROR: {EXISTING_PATH} not found. Run Stage 4.5 first.",
            file=sys.stderr,
        )
        return 1

    rows = read_tsv(EXISTING_PATH)
    voices = load_voices(VOICES_PATH)
    overrides = _load_overrides(OVERRIDES_PATH)
    female_voices, male_voices = split_by_gender(voices)

    # --- Apply confidence policy + manual gender overrides -----------------

    raw_gender: dict[str, str] = {}
    demoted = 0
    manual_gender_count = 0

    for r in rows:
        sid = r["sense_id"]
        sg = (r.get("speaker_gender") or "neutral").strip().lower()
        confidence = (r.get("confidence") or "").strip().lower()
        method = (r.get("assignment_method") or "").strip().lower()

        ov = overrides.get(sid, {})

        # Manual override on speaker_gender wins (treated as authoritative)
        if ov.get("speaker_gender"):
            raw_gender[sid] = ov["speaker_gender"]
            manual_gender_count += 1
            continue

        # Manual override marker preserved from existing TSV (in case the
        # row was originally produced via override)
        if method == "manual_override":
            raw_gender[sid] = sg
            manual_gender_count += 1
            continue

        # Confidence policy
        if sg in ("male", "female") and confidence != "high":
            raw_gender[sid] = "neutral"
            demoted += 1
        else:
            raw_gender[sid] = sg

    # Apply explicit voice_gender overrides (post-confidence policy)
    for sid, ov in overrides.items():
        if sid in raw_gender and ov.get("voice_gender"):
            raw_gender[sid] = ov["voice_gender"]

    # --- Phase 2a: balanced gender resolution ------------------------------

    voice_gender_assigned = resolve_neutrals(raw_gender, seed=SEED_NEUTRAL_BALANCE)

    # --- Phase 2b: per-voice round-robin -----------------------------------

    voice_id_for = assign_voice_ids(
        voice_gender_assigned,
        voices,
        female_seed=SEED_FEMALE_POOL,
        male_seed=SEED_MALE_POOL,
    )

    # Apply explicit voice_id overrides last
    for sid, ov in overrides.items():
        if sid in voice_id_for and ov.get("voice_id"):
            voice_id_for[sid] = ov["voice_id"]

    # --- Build updated rows ------------------------------------------------

    flipped_gender = 0
    flipped_voice = 0
    out_rows: list[dict] = []
    for r in rows:
        sid = r["sense_id"]
        new_voice_gender = voice_gender_assigned[sid]
        new_voice_id = voice_id_for[sid]
        if new_voice_gender != r.get("voice_gender_assigned", ""):
            flipped_gender += 1
        if new_voice_id != r.get("voice_id", ""):
            flipped_voice += 1
        out_rows.append(
            {
                "sense_id": sid,
                "speaker_gender": r["speaker_gender"],
                "evidence": r["evidence"],
                "confidence": r["confidence"],
                "voice_gender_assigned": new_voice_gender,
                "voice_id": new_voice_id,
                "assignment_method": r.get("assignment_method", "llm"),
            }
        )

    n_out = write_tsv(EXISTING_PATH, out_rows, fieldnames=OUTPUT_FIELDS)

    # --- Verify invariants -------------------------------------------------

    valid_voice_ids = {v.voice_id for v in voices}
    missing = [r["sense_id"] for r in out_rows if r["voice_id"] not in valid_voice_ids]
    if missing:
        print(
            f"ERROR: {len(missing)} rows have voice_id not in pool: {missing[:5]}",
            file=sys.stderr,
        )
        return 1

    is_balanced, balance_msg = verify_balanced_within_pool(voice_id_for, voices)

    counts = usage_counts(voice_id_for)
    from collections import Counter

    voice_gender_counts = Counter(r["voice_gender_assigned"] for r in out_rows)

    print(f"Re-applied Phase 2 to {n_out} rows.")
    print(f"  Demoted to neutral (confidence != high):  {demoted}")
    print(f"  Manual gender overrides preserved:        {manual_gender_count}")
    print(f"  voice_gender_assigned flipped:            {flipped_gender}")
    print(f"  voice_id flipped:                          {flipped_voice}")
    print(
        f"  Final voice_gender split: female={voice_gender_counts['female']} "
        f"male={voice_gender_counts['male']}"
    )
    print(f"  Per-pool balance: {balance_msg}  [balanced={is_balanced}]")
    print()
    print("Voice usage:")
    for v in sorted(voices, key=lambda v: (v.gender, v.pool_index)):
        print(f"  {v.voice_id}  {v.gender:6}  {counts.get(v.voice_id, 0):4d} senses")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
