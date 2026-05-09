"""Mark manifest rows whose voice changed as pending, so Stage 6 resume picks
them up.

Workflow:
  1. config/voices.tsv has been edited (3 IDs swapped).
  2. build/recompute_voice_assignment.py has been run → 045-speaker_gender.tsv
     now reflects the new pool.
  3. THIS script walks data/_audio_manifest.tsv: for any row whose voice_id
     differs from the new assignment for that sense_id, clear the audio
     fields (md5 / url / asr_* / loudness_*) and set status=pending. The
     `voice_id` and `voice_gender` fields are updated to the new voice.
     Version is NOT bumped — the new audio overwrites the same R2 key
     (`{sense_id}-{clip_type}-v1.mp3`).

Idempotent: if the manifest already matches the new assignments, no rows
are touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    STATUS_PENDING,
    read_manifest,
    summarize,
    write_manifest,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
SPEAKER_GENDER_PATH = DATA_DIR / "045-speaker_gender.tsv"


def _clear_audio_fields(row: dict, *, new_voice_id: str, new_gender: str) -> None:
    """Reset everything that depends on the audio bytes; keep identity columns."""
    row["voice_id"] = new_voice_id
    row["voice_gender"] = new_gender
    for f in (
        "md5",
        "asr_transcript",
        "asr_similarity",
        "asr_decision",
        "applied_gain_db",
        "final_lufs",
        "final_tp",
        "loudness_within_tolerance",
        "tp_limited",
        "generated_at",
    ):
        row[f] = ""
    row["status"] = STATUS_PENDING
    row["notes"] = f"voice swapped to {new_voice_id[:10]}…; awaiting re-render"


def main() -> int:
    if not SPEAKER_GENDER_PATH.exists():
        print(f"missing {SPEAKER_GENDER_PATH}; run build/recompute_voice_assignment.py first", file=sys.stderr)
        return 1

    assignments = read_tsv(SPEAKER_GENDER_PATH)
    new_voice_for: dict[str, tuple[str, str]] = {}
    for r in assignments:
        sid = r["sense_id"]
        new_voice_for[sid] = (r["voice_id"], r.get("voice_gender_assigned", ""))

    manifest = read_manifest()
    if not manifest:
        print(f"manifest empty at {DEFAULT_MANIFEST_PATH}", file=sys.stderr)
        return 1

    print(f"[invalidate] manifest before: {summarize(manifest).fmt()}", file=sys.stderr)

    invalidated = 0
    by_voice: dict[str, int] = {}
    for row in manifest:
        sid = row["sense_id"]
        if sid not in new_voice_for:
            # No assignment for this sense (shouldn't happen). Leave it alone.
            continue
        new_voice_id, new_gender = new_voice_for[sid]
        if row["voice_id"] != new_voice_id:
            old_voice = row["voice_id"]
            _clear_audio_fields(row, new_voice_id=new_voice_id, new_gender=new_gender)
            invalidated += 1
            by_voice[old_voice] = by_voice.get(old_voice, 0) + 1

    write_manifest(manifest)

    print(f"[invalidate] {invalidated} manifest rows marked pending (voice swap).", file=sys.stderr)
    print("  by old voice_id:", file=sys.stderr)
    for vid, n in sorted(by_voice.items(), key=lambda x: -x[1]):
        print(f"    {vid}  {n} rows", file=sys.stderr)
    print(f"[invalidate] manifest after:  {summarize(manifest).fmt()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
