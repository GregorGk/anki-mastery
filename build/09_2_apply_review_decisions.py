"""Stage 9 / Step 2 — Apply review-queue decisions from the HTML export.

Reads `data/_audio_review_queue_decisions.tsv` and processes each row:

  LEAVE              → no-op (audio is fine).
  REGEN_VOICE_SWAP   → update voice_id in 045-speaker_gender.tsv and
                       manifest (both word + example clips for that
                       sense), mark them pending, re-render with the
                       new voice. Also updates voice_gender_assigned if
                       gender changed.
  TAG_ALIAS          → append rule to _pronunciation_aliases.tsv +
                       _pronunciation_alias_applications.tsv (both
                       clip_types), upload new dict version, mark
                       that sense pending, re-render.

After all decisions are applied to the TSVs, runs stage_6 in resume
mode with sense_id_filter limited to the affected senses. Concurrency
defaults to 20 (Pro/Flash ceiling).

Decisions TSV format (with known JS export quirk where sense_id+clip_type
are concatenated into the first column without a tab):
  sense_id+clip_type | (empty) | pt | (empty) | asr | sim | decision | new_voice_id | alias_respelling
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_manifest import (  # noqa: E402
    bump_version,
    read_manifest,
    write_manifest,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"

DECISIONS_PATH = DATA_DIR / "_audio_review_queue_decisions.tsv"
SPEAKER_GENDER_PATH = DATA_DIR / "045-speaker_gender.tsv"
ALIASES_PATH = DATA_DIR / "_pronunciation_aliases.tsv"
APPLICATIONS_PATH = DATA_DIR / "_pronunciation_alias_applications.tsv"
VOICES_PATH = CONFIG_DIR / "voices.tsv"

FLASH_MODEL_ID = "eleven_flash_v2_5"


def _split_sid_cliptype(merged: str) -> tuple[str, str]:
    """Parse '2043.00.01word' → ('2043.00.01', 'word'). The HTML JS
    exporter joins them without a tab. Sense_id is always 10 chars
    (RRRR.EE.SS), so split there."""
    merged = merged.strip()
    if len(merged) < 10:
        return (merged, "")
    sid = merged[:10]
    rest = merged[10:]
    if rest in ("word", "example"):
        return (sid, rest)
    return (merged, "")


def _parse_decisions(path: Path) -> list[dict]:
    """Parse decisions TSV with the known JS export quirks."""
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    out: list[dict] = []
    raw_rows = read_tsv(path)
    for r in raw_rows:
        merged = r.get("sense_id", "")
        sid, ctype = _split_sid_cliptype(merged)
        if not sid or not ctype:
            print(f"  WARN: could not parse {merged!r}, skipping", file=sys.stderr)
            continue
        out.append({
            "sense_id": sid,
            "clip_type": ctype,
            "pt": r.get("pt", ""),
            "asr_transcript": r.get("asr_transcript", ""),
            "similarity": r.get("similarity", ""),
            "decision": r.get("decision", "LEAVE"),
            "new_voice_id": r.get("new_voice_id", ""),
            "alias_respelling": r.get("alias_respelling", ""),
        })
    return out


def _load_voice_gender_map() -> dict[str, str]:
    return {r["voice_id"]: r.get("gender", "")
            for r in read_tsv(VOICES_PATH)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Apply decisions to TSVs but skip dict upload + re-render.")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--apply-only", action="store_true",
                        help="Apply TSV changes + dict upload but skip stage_6 re-render.")
    args = parser.parse_args()

    if not os.environ.get("ELEVENLABS_API_KEY") and not args.dry_run:
        print("ERROR: ELEVENLABS_API_KEY not set.", file=sys.stderr)
        return 1

    decisions = _parse_decisions(DECISIONS_PATH)
    print(f"Parsed {len(decisions)} decisions from {DECISIONS_PATH.name}")
    n_leave = sum(1 for d in decisions if d["decision"] == "LEAVE")
    n_swap = sum(1 for d in decisions if d["decision"] == "REGEN_VOICE_SWAP")
    n_alias = sum(1 for d in decisions if d["decision"] == "TAG_ALIAS")
    print(f"  LEAVE: {n_leave}  REGEN_VOICE_SWAP: {n_swap}  TAG_ALIAS: {n_alias}")
    print()

    if n_swap == 0 and n_alias == 0:
        print("No actionable decisions. Exiting.")
        return 0

    voice_gender = _load_voice_gender_map()

    # ── Apply voice swaps ──────────────────────────────────────────────
    swap_decisions = [d for d in decisions if d["decision"] == "REGEN_VOICE_SWAP"]
    swap_sids: set[str] = set()
    speaker_rows = read_tsv(SPEAKER_GENDER_PATH)
    speaker_idx = {r["sense_id"]: r for r in speaker_rows}

    for d in swap_decisions:
        sid = d["sense_id"]
        new_vid = d["new_voice_id"]
        if not new_vid:
            print(f"  WARN: {sid} swap has empty new_voice_id, skipping", file=sys.stderr)
            continue
        if sid not in speaker_idx:
            print(f"  WARN: {sid} not in speaker_gender TSV, skipping", file=sys.stderr)
            continue
        row = speaker_idx[sid]
        old_vid = row["voice_id"]
        new_gender = voice_gender.get(new_vid, "?")
        row["voice_id"] = new_vid
        row["voice_gender_assigned"] = new_gender
        row["assignment_method"] = (row.get("assignment_method", "") + "+manual_swap_review_queue").strip("+")
        print(f"  swap {sid}: {old_vid[:10]} → {new_vid[:10]} ({new_gender})")
        swap_sids.add(sid)

    write_tsv(SPEAKER_GENDER_PATH, speaker_rows,
              fieldnames=list(speaker_rows[0].keys()) if speaker_rows else None)
    print(f"  ✓ {SPEAKER_GENDER_PATH.name} updated ({len(swap_sids)} senses)")
    print()

    # ── Apply alias tags ───────────────────────────────────────────────
    alias_decisions = [d for d in decisions if d["decision"] == "TAG_ALIAS"]
    alias_sids: set[str] = set()
    if alias_decisions:
        aliases_rows = read_tsv(ALIASES_PATH)
        max_alias_id = max((int(r["alias_id"]) for r in aliases_rows
                            if r.get("alias_id", "").isdigit()), default=0)
        apps_rows = read_tsv(APPLICATIONS_PATH)
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

        for d in alias_decisions:
            sid = d["sense_id"]
            pt = d["pt"]
            respelling = d["alias_respelling"].strip()
            if not pt or not respelling:
                print(f"  WARN: {sid} alias missing pt or respelling, skipping",
                      file=sys.stderr)
                continue
            # Check if alias for this pt already exists
            existing = next((r for r in aliases_rows if r.get("pt") == pt), None)
            if existing:
                print(f"  alias for {pt!r} already exists "
                      f"({existing['pt_respelling']!r}); reusing", file=sys.stderr)
                alias_id = existing["alias_id"]
            else:
                max_alias_id += 1
                alias_id = str(max_alias_id)
                aliases_rows.append({
                    "alias_id": alias_id, "pt": pt, "pt_respelling": respelling,
                    "rationale": f"Stage 9 review queue: ASR-flagged sense, "
                                 f"user-supplied respelling to fix BP pronunciation",
                    "applied_family": "review_queue_v9",
                    "confidence": "high",
                    "source": "review_queue_flash_era",
                    "created_at": now,
                })
                print(f"  alias new id={alias_id}: {pt} → {respelling}")
            # Add applications for both clip_types
            for ct in ("word", "example"):
                if not any(r.get("sense_id") == sid and r.get("clip_type") == ct
                           for r in apps_rows):
                    apps_rows.append({
                        "sense_id": sid, "clip_type": ct,
                        "alias_id": alias_id, "applied_at": now,
                    })
            alias_sids.add(sid)

        write_tsv(ALIASES_PATH, aliases_rows,
                  fieldnames=list(aliases_rows[0].keys()))
        write_tsv(APPLICATIONS_PATH, apps_rows,
                  fieldnames=list(apps_rows[0].keys()))
        print(f"  ✓ {ALIASES_PATH.name} updated ({len(alias_decisions)} new rules)")
        print(f"  ✓ {APPLICATIONS_PATH.name} updated")
    print()

    # ── Mark affected manifest rows as pending ─────────────────────────
    affected_sids = swap_sids | alias_sids
    manifest_rows = read_manifest()
    n_marked = 0
    from build.lib.r2_client import R2Client
    r2 = R2Client()
    public_base = r2.config.public_base
    for r in manifest_rows:
        if r["sense_id"] not in affected_sids:
            continue
        # Update voice_id in manifest if this sense was swapped
        if r["sense_id"] in swap_sids:
            new_vid = next((d["new_voice_id"] for d in swap_decisions
                            if d["sense_id"] == r["sense_id"]), "")
            if new_vid:
                r["voice_id"] = new_vid
                r["voice_gender"] = voice_gender.get(new_vid, r.get("voice_gender", ""))
        bump_version(r, public_base=public_base, model_id=FLASH_MODEL_ID)
        r["status"] = "pending"
        r["notes"] = (r.get("notes", "") + " | stage_9_review_queue").strip(" |")
        n_marked += 1
    write_manifest(manifest_rows)
    print(f"  ✓ {n_marked} manifest rows marked pending "
          f"({len(affected_sids)} unique senses × ~2 clips each)")
    print()

    if args.dry_run:
        print("--dry-run: TSVs updated, skipping dict upload + re-render.")
        return 0

    # ── Upload new dict version if aliases changed ─────────────────────
    if alias_decisions:
        print("Uploading new dict version with updated aliases...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "build/08_4_upload_dictionary.py", "--confirm"],
            cwd=REPO_ROOT, env=os.environ.copy(),
        )
        if result.returncode != 0:
            print("ERROR: dict upload failed.", file=sys.stderr)
            return 1
    print()

    if args.apply_only:
        print("--apply-only: skipping stage_6 re-render. Manifest marked pending; run stage_6 manually.")
        return 0

    # ── Re-render affected senses via stage_6 ──────────────────────────
    from build.lib.tsv import read_tsv as _rt
    meta = _rt(DATA_DIR / "_audio_dictionary_meta.tsv")
    locator = None
    if meta:
        last = meta[-1]
        if last.get("dictionary_id") and last.get("version_id"):
            locator = {
                "pronunciation_dictionary_id": last["dictionary_id"],
                "version_id": last["version_id"],
            }

    print(f"Delegating to stage_6.run() for {len(affected_sids)} senses, "
          f"concurrency={args.concurrency}, model={FLASH_MODEL_ID}...")
    from build.stage_6 import run as stage_6_run
    result = stage_6_run(
        pilot_size=None,
        concurrency=args.concurrency,
        confirm=True,
        sense_id_filter=affected_sids,
        pronunciation_dict_locators=[locator] if locator else None,
        model_id=FLASH_MODEL_ID,
        fail_fast_on_429=True,
    )
    print()
    print("=== Stage 9 review-queue re-render summary ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
