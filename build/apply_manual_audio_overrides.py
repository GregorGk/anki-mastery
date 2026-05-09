"""Apply manual audio overrides from data/_manual_audio.tsv.

Schema:
    sense_id   clip_type   decision_override   reason   set_at

decision_override ∈ {pass, regen}:
  - `pass`  → manifest's asr_decision := pass (audio is fine, ASR was wrong).
              Any matching row in _audio_human_review.tsv is removed.
  - `regen` → manifest status := pending (Stage 6 resume mode will re-render
              the clip on the next run with bumped version).

Idempotent: re-running with the same override file produces the same result.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_manifest import (  # noqa: E402
    STATUS_PENDING,
    bump_version,
    index_by_key,
    read_manifest,
    summarize,
    write_manifest,
)
from build.lib.r2_client import R2Client  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
MANUAL_AUDIO_PATH = DATA_DIR / "_manual_audio.tsv"
HUMAN_REVIEW_PATH = DATA_DIR / "_audio_human_review.tsv"

MANUAL_AUDIO_FIELDS = [
    "sense_id",
    "clip_type",
    "decision_override",
    "reason",
    "set_at",
]
HUMAN_REVIEW_FIELDS = [
    "sense_id",
    "clip_type",
    "voice_id",
    "input_text",
    "asr_transcript",
    "similarity",
    "phonetic_distance",
    "url",
    "added_at",
    "notes",
]


def main() -> int:
    if not MANUAL_AUDIO_PATH.exists():
        print(f"missing {MANUAL_AUDIO_PATH}; nothing to apply", file=sys.stderr)
        return 0

    overrides = read_tsv(MANUAL_AUDIO_PATH)
    if not overrides:
        print(f"empty {MANUAL_AUDIO_PATH}", file=sys.stderr)
        return 0

    manifest = read_manifest()
    rows_by_key = index_by_key(manifest)

    public_base = ""
    try:
        r2 = R2Client()
        public_base = r2.config.public_base
    except Exception:
        # No R2 credentials available; bump_version still works without
        # accurate URL recompute (the URL gets fixed at next Stage 6 upload).
        public_base = "https://placeholder"

    applied_pass = 0
    applied_regen = 0
    skipped = 0
    affected_keys: set[tuple[str, str]] = set()

    for o in overrides:
        sid = o.get("sense_id", "").strip()
        ctype = o.get("clip_type", "").strip()
        decision = o.get("decision_override", "").strip().lower()
        reason = o.get("reason", "").strip()
        if not sid or not ctype:
            continue
        key = (sid, ctype)
        row = rows_by_key.get(key)
        if row is None:
            print(f"  [warn] no manifest row for {sid} {ctype}; skipping", file=sys.stderr)
            skipped += 1
            continue
        if decision == "pass":
            row["asr_decision"] = "pass"
            row["notes"] = f"manual override: pass — {reason}" if reason else "manual override: pass"
            affected_keys.add(key)
            applied_pass += 1
        elif decision == "regen":
            bump_version(row, public_base=public_base)
            row["status"] = STATUS_PENDING
            row["notes"] = f"manual override: regen — {reason}" if reason else "manual override: regen"
            affected_keys.add(key)
            applied_regen += 1
        else:
            print(f"  [warn] unknown decision_override {decision!r} for {sid} {ctype}; skipping", file=sys.stderr)
            skipped += 1

    write_manifest(manifest)
    print(f"[manual_audio] applied: {applied_pass} pass / {applied_regen} regen / {skipped} skipped", file=sys.stderr)

    # Drop pass-overridden rows from the human review queue
    if HUMAN_REVIEW_PATH.exists():
        queue = read_tsv(HUMAN_REVIEW_PATH)
        before = len(queue)
        kept = []
        removed = 0
        for q in queue:
            key = (q.get("sense_id", ""), q.get("clip_type", ""))
            if key in affected_keys and rows_by_key.get(key, {}).get("asr_decision") == "pass":
                removed += 1
                continue
            kept.append(q)
        if removed:
            write_tsv(HUMAN_REVIEW_PATH, kept, fieldnames=HUMAN_REVIEW_FIELDS)
            print(f"[manual_audio] removed {removed} entries from human review queue ({before} -> {len(kept)})", file=sys.stderr)

    print(f"[manual_audio] manifest: {summarize(manifest).fmt()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
