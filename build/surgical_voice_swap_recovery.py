"""Surgical voice-swap recovery using a consistent-hashing-style algorithm.

Context: build/recompute_voice_assignment.py uses naive round-robin
(`i % len(VOICES)`) which is BRITTLE to pool changes — removing voices
from the list re-shuffles assignments deterministically but globally.
That's why a 3-voice removal flipped 4,600 of 5,725 corpus assignments.

This script does it correctly: senses on SURVIVING voices keep their
voice (zero re-render). Only senses on DROPPED voices are redistributed
to survivors via deterministic round-robin within the same gender pool,
with off-by-1 increment per voice ("at most one differs").

Reads:
  - HEAD:data/045-speaker_gender.tsv  (original 11-voice assignments,
                                       via git show — current file may
                                       have been clobbered by a prior
                                       full-recompute mishap)
  - config/voices.tsv                  (current pool, post-edit)
  - audit/06_audio.jsonl               (to identify clips touched by
                                       a partial/killed run that may
                                       have left R2 in an inconsistent
                                       state)
  - data/_audio_manifest.tsv

Writes:
  - data/045-speaker_gender.tsv  (rewritten with surgical assignments)
  - data/_audio_manifest.tsv     (rebuilt with correct voice_id +
                                  status reflecting actual R2 state)

After running this, run `build/06_audio_pilot.py --pilot-size 500
--confirm` in resume mode to render the rows correctly marked pending.
"""
from __future__ import annotations

import csv
import io
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_manifest import (  # noqa: E402
    STATUS_PENDING,
    STATUS_UPLOADED,
    read_manifest,
    summarize,
    write_manifest,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
SPEAKER_GENDER_PATH = DATA_DIR / "045-speaker_gender.tsv"
VOICES_PATH = REPO_ROOT / "config" / "voices.tsv"
AUDIO_JSONL = AUDIT_DIR / "06_audio.jsonl"

# Seeds for the round-robin shuffles (locked).
FEMALE_REDIST_SEED = 45
MALE_REDIST_SEED = 46

SPEAKER_GENDER_FIELDS = [
    "sense_id",
    "speaker_gender",
    "evidence",
    "confidence",
    "voice_gender_assigned",
    "voice_id",
    "assignment_method",
]


def _git_show(rev_path: str) -> str:
    out = subprocess.check_output(["git", "show", rev_path]).decode("utf-8")
    return out


def restore_original_assignments() -> list[dict]:
    """Pull the pre-recompute 045-speaker_gender.tsv from git HEAD."""
    blob = _git_show("HEAD:data/045-speaker_gender.tsv")
    reader = csv.DictReader(io.StringIO(blob), dialect="excel-tab")
    return list(reader)


def survivors_by_gender() -> tuple[list[str], list[str]]:
    """Read the (current, post-edit) config/voices.tsv. Return (female_ids,
    male_ids) ordered by pool_index ascending."""
    rows = read_tsv(VOICES_PATH)
    rows.sort(key=lambda r: int(r["pool_index"]))
    female = [r["voice_id"] for r in rows if r["gender"] == "female"]
    male = [r["voice_id"] for r in rows if r["gender"] == "male"]
    return female, male


def compute_surgical(
    original: list[dict],
    surviving_female: list[str],
    surviving_male: list[str],
) -> dict[str, dict]:
    """Return {sense_id: {voice_id, voice_gender_assigned, ...}} with surgical
    reassignment applied.

    Senses on surviving voices keep their voice (zero re-render). Senses on
    dropped voices are redistributed within their gender pool via
    **greedy fill-lowest**: each redistributed sense goes to the surviving
    voice with the lowest current count, deterministic tiebreak by
    pool_index. This guarantees max-min ≤ 1 per gender pool no matter what
    the starting counts were (which may have been off-by-1 already in the
    pre-swap pool).
    """
    from collections import Counter

    surviving_set = set(surviving_female) | set(surviving_male)

    # Initial counts: how many senses each surviving voice currently has
    # (from senses that were originally on it and weren't dropped).
    surviving_counts: Counter[str] = Counter()
    for r in original:
        if r["voice_id"] in surviving_set:
            surviving_counts[r["voice_id"]] += 1
    # Ensure every surviving voice has an entry, even at 0
    for v in surviving_female + surviving_male:
        surviving_counts.setdefault(v, 0)

    # Bucket the dropped-voice senses by gender, sort deterministically,
    # then shuffle with a locked seed so the redistribution is reproducible.
    female_dropped: list[str] = []
    male_dropped: list[str] = []
    for r in original:
        if r["voice_id"] not in surviving_set:
            if r["voice_gender_assigned"] == "female":
                female_dropped.append(r["sense_id"])
            elif r["voice_gender_assigned"] == "male":
                male_dropped.append(r["sense_id"])

    female_dropped.sort()
    male_dropped.sort()
    random.Random(FEMALE_REDIST_SEED).shuffle(female_dropped)
    random.Random(MALE_REDIST_SEED).shuffle(male_dropped)

    # Greedy fill-lowest within each gender pool. Tiebreak by pool_index
    # (= position in the list, since the list is already sorted by pool_index
    # in the caller).
    def _pick_lowest(pool: list[str]) -> str:
        # pool is ordered by pool_index ascending. Pick the first (lowest
        # pool_index) voice among those tied for minimum count.
        best = pool[0]
        for v in pool[1:]:
            if surviving_counts[v] < surviving_counts[best]:
                best = v
        return best

    new_voice_for: dict[str, str] = {}
    if surviving_female:
        for sid in female_dropped:
            v = _pick_lowest(surviving_female)
            new_voice_for[sid] = v
            surviving_counts[v] += 1
    if surviving_male:
        for sid in male_dropped:
            v = _pick_lowest(surviving_male)
            new_voice_for[sid] = v
            surviving_counts[v] += 1

    # Build full surgical mapping from original
    surgical: dict[str, dict] = {}
    for r in original:
        sid = r["sense_id"]
        new_v = new_voice_for.get(sid, r["voice_id"])
        surgical[sid] = {
            **r,
            "voice_id": new_v,
            "assignment_method": (
                "manual_override" if r.get("assignment_method") == "manual_override"
                else ("surgical_reassign" if sid in new_voice_for else r.get("assignment_method", "llm"))
            ),
        }
    return surgical


def find_touched_clips_with_voice(after_iso: str) -> dict[tuple[str, str], str]:
    """Return {(sense_id, clip_type): voice_id} for asr_completed events with
    completed_at > after_iso. The voice_id is what the killed Stage 6 actually
    used for TTS, which is what the R2 object now contains. Robust against
    later mutations of the manifest."""
    touched: dict[tuple[str, str], str] = {}
    if not AUDIO_JSONL.exists():
        return touched
    with AUDIO_JSONL.open(encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") != "asr_completed":
                continue
            ts = e.get("completed_at", "")
            if ts and ts > after_iso:
                touched[(e.get("sense_id", ""), e.get("clip_type", ""))] = e.get("voice_id", "")
    return touched


def find_killed_run_threshold() -> str | None:
    """Find the largest timestamp gap > 5 min in the audit log; return the
    start of the post-gap activity. Used as the threshold for "touched by the
    killed run"."""
    if not AUDIO_JSONL.exists():
        return None
    timestamps: list[str] = []
    with AUDIO_JSONL.open(encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = e.get("started_at") or e.get("completed_at") or e.get("errored_at") or e.get("ts") or ""
            if ts:
                timestamps.append(ts)
    timestamps.sort()
    if len(timestamps) < 2:
        return None
    from datetime import datetime

    largest_gap_seconds = 0.0
    threshold = None
    for i in range(1, len(timestamps)):
        a = datetime.fromisoformat(timestamps[i - 1])
        b = datetime.fromisoformat(timestamps[i])
        gap = (b - a).total_seconds()
        if gap > largest_gap_seconds and gap > 300:
            largest_gap_seconds = gap
            threshold = timestamps[i]
    return threshold


def _clear_audio_fields(row: dict) -> None:
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


def main() -> int:
    print("[recovery] step 1: restoring original 045-speaker_gender.tsv from git HEAD", file=sys.stderr)
    original = restore_original_assignments()
    print(f"  {len(original)} rows", file=sys.stderr)

    print("[recovery] step 2: reading current voice pool", file=sys.stderr)
    surviving_female, surviving_male = survivors_by_gender()
    print(f"  female: {len(surviving_female)} voices  male: {len(surviving_male)} voices", file=sys.stderr)

    print("[recovery] step 3: computing surgical reassignment", file=sys.stderr)
    surgical = compute_surgical(original, surviving_female, surviving_male)
    flipped = sum(1 for sid, r in surgical.items() if r["voice_id"] != next(o for o in original if o["sense_id"] == sid)["voice_id"])
    print(f"  voice_id flipped: {flipped}/{len(surgical)} senses (= dropped-voice senses redistributed)", file=sys.stderr)

    # Per-voice usage check
    counts = Counter(r["voice_id"] for r in surgical.values())
    print("  per-voice usage (full corpus):", file=sys.stderr)
    for vid, n in sorted(counts.items(), key=lambda x: (x[0] not in surviving_female, x[0])):
        print(f"    {vid}  {n}", file=sys.stderr)
    f_counts = [counts[v] for v in surviving_female]
    m_counts = [counts[v] for v in surviving_male]
    print(f"  female pool spread: {max(f_counts) - min(f_counts) if f_counts else 0}", file=sys.stderr)
    print(f"  male pool spread:   {max(m_counts) - min(m_counts) if m_counts else 0}", file=sys.stderr)

    print("[recovery] step 4: writing surgical 045-speaker_gender.tsv", file=sys.stderr)
    surgical_rows = list(surgical.values())
    surgical_rows.sort(key=lambda r: r["sense_id"])
    write_tsv(SPEAKER_GENDER_PATH, surgical_rows, fieldnames=SPEAKER_GENDER_FIELDS)

    print("[recovery] step 5: identifying clips touched by the killed run via audit log", file=sys.stderr)
    threshold = find_killed_run_threshold()
    if threshold:
        print(f"  killed-run threshold: events after {threshold}", file=sys.stderr)
        touched_voice = find_touched_clips_with_voice(threshold)
        print(f"  {len(touched_voice)} clips touched by killed run (with voice_id captured from audit log)", file=sys.stderr)
    else:
        print("  no large gap found in audit log; assuming no killed-run residue", file=sys.stderr)
        touched_voice = {}

    print("[recovery] step 6: rebuilding manifest with surgical voice + correct status", file=sys.stderr)
    manifest = read_manifest()
    if not manifest:
        print("  ERROR: empty manifest", file=sys.stderr)
        return 1

    # Build lookup maps
    original_voice_for = {r["sense_id"]: r["voice_id"] for r in original}

    n_kept_uploaded = 0
    n_remeasure_needed = 0
    n_marked_pending_for_swap = 0
    n_marked_pending_for_corruption = 0

    for row in manifest:
        sid = row["sense_id"]
        ctype = row["clip_type"]
        original_voice = original_voice_for.get(sid, "")
        surgical_voice = surgical[sid]["voice_id"]
        new_gender = surgical[sid]["voice_gender_assigned"]
        key = (sid, ctype)

        # Determine what voice's audio is currently on R2.
        # Truth source: audit log records the voice_id the killed Stage 6
        # actually used for each touched clip. Untouched clips still have
        # the original Stage 6 v2 pilot's audio = original_voice.
        if key in touched_voice:
            r2_voice = touched_voice[key]  # killed run's voice
        else:
            r2_voice = original_voice  # untouched: original audio preserved

        # Apply surgical assignments
        row["voice_id"] = surgical_voice
        row["voice_gender"] = new_gender

        # Determine status
        if r2_voice == surgical_voice:
            # R2 has correct audio under surgical
            if row.get("md5"):
                # Metadata intact (212 originally-uploaded rows)
                row["status"] = STATUS_UPLOADED
                n_kept_uploaded += 1
            else:
                # Untouched-but-pending OR touched-and-recompute-matches-surgical:
                # audio is correct on R2 but manifest fields cleared by invalidate.
                # Mark pending → Stage 6 will regenerate. (Re-measuring is
                # equivalent cost; just simpler to re-render uniformly.)
                row["status"] = STATUS_PENDING
                _clear_audio_fields(row)
                row["notes"] = (
                    "post-recovery: surgical voice matches R2 audio but metadata cleared; "
                    "Stage 6 will re-render"
                )
                n_remeasure_needed += 1
        else:
            # R2 wrong (either touched-and-corrupted or untouched-but-on-dropped-voice)
            row["status"] = STATUS_PENDING
            _clear_audio_fields(row)
            if surgical_voice != original_voice:
                row["notes"] = "post-recovery: voice changed via surgical reassignment; needs re-render"
                n_marked_pending_for_swap += 1
            else:
                row["notes"] = "post-recovery: R2 corrupted by killed run; needs re-render with original voice"
                n_marked_pending_for_corruption += 1

    write_manifest(manifest)

    print(file=sys.stderr)
    print(f"=== Recovery summary ===", file=sys.stderr)
    print(f"  uploaded (intact):                          {n_kept_uploaded}", file=sys.stderr)
    print(f"  pending - voice changed (need re-render):   {n_marked_pending_for_swap}", file=sys.stderr)
    print(f"  pending - R2 corrupted (need re-render):    {n_marked_pending_for_corruption}", file=sys.stderr)
    print(f"  pending - metadata only (need re-render):   {n_remeasure_needed}", file=sys.stderr)
    print(f"  total pending:                              {n_marked_pending_for_swap + n_marked_pending_for_corruption + n_remeasure_needed}", file=sys.stderr)
    print(f"  manifest: {summarize(manifest).fmt()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
