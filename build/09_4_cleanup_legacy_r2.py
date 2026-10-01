"""Stage 9 / Step 4 — Cleanup orphaned legacy Multilingual v2 audio in R2.

After the Flash v2.5 migration, the manifest exclusively references
URLs containing `eleven_flash_v2_5`. All older keys
(`audio/{sid}-{clip}-v{N}.mp3` without model_id, plus any
mid-iteration versions superseded by re-renders) are orphans.

This script:
  1. Lists every object under `audio/` in R2.
  2. Reads the current `_audio_manifest.tsv` and extracts the set of
     keys it references.
  3. Identifies orphans (in R2, not in manifest).
  4. Reports counts + size + sample.
  5. If `--confirm`, deletes orphans via the S3 `DeleteObjects` batch
     API (up to 1000 keys per call).

Safety:
  - Default mode is DRY-RUN. Need `--confirm` to actually delete.
  - Whitelist: any key referenced by the manifest is preserved.
  - The script never deletes anything outside the `audio/` prefix.

Usage:
    .venv/bin/python build/09_4_cleanup_legacy_r2.py           # dry-run
    .venv/bin/python build/09_4_cleanup_legacy_r2.py --confirm # actually delete
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=True)

from build.lib.r2_client import R2Client  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
AUDIO_PREFIX = "audio/"
BATCH_SIZE = 1000  # S3 DeleteObjects max per call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--confirm", action="store_true",
                        help="Actually delete orphans (default: dry-run report).")
    parser.add_argument("--sample", type=int, default=10,
                        help="How many orphan keys to print as samples.")
    parser.add_argument("--keep-manifest", action="append", default=[],
                        help="Also preserve every key referenced by this manifest file "
                             "(repeatable) — e.g. data/_audio_manifest.tsv.bak_pre_stage19 "
                             "keeps the v3 rollback copies after the Stage-19 v4 swap.")
    args = parser.parse_args()

    r2 = R2Client()
    s3 = r2._s3
    bucket = r2.config.bucket

    # Step 1: list all R2 objects under audio/
    print(f"Listing R2 bucket={bucket} prefix={AUDIO_PREFIX!r}...")
    paginator = s3.get_paginator("list_objects_v2")
    r2_keys: list[tuple[str, int]] = []  # (key, size)
    page_n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=AUDIO_PREFIX):
        page_n += 1
        for obj in page.get("Contents", []):
            r2_keys.append((obj["Key"], obj.get("Size", 0)))
        if page_n % 5 == 0:
            print(f"  paginated {page_n} pages, {len(r2_keys):,} keys so far")
    total_r2_keys = len(r2_keys)
    total_r2_bytes = sum(s for _, s in r2_keys)
    print(f"  R2 total: {total_r2_keys:,} objects ({total_r2_bytes / 1024 / 1024:.1f} MB)")

    # Step 2: keys referenced by current manifest
    manifest_keys: set[str] = set()
    for r in read_tsv(MANIFEST_PATH):
        ok = r.get("object_key", "")
        if ok:
            manifest_keys.add(ok)
    print(f"  Manifest references: {len(manifest_keys):,} keys")
    for extra in args.keep_manifest:
        extra_rows = read_tsv(REPO_ROOT / extra if not Path(extra).is_absolute() else Path(extra))
        if not extra_rows:
            print(f"ERROR: --keep-manifest {extra} is empty or missing", file=sys.stderr)
            return 2
        before = len(manifest_keys)
        manifest_keys |= {r["object_key"] for r in extra_rows if r.get("object_key")}
        print(f"  + keep-list {extra}: {len(manifest_keys) - before:,} extra keys preserved")

    # Step 3: orphans = R2 keys NOT in manifest
    orphans = [(k, s) for k, s in r2_keys if k not in manifest_keys]
    orphan_bytes = sum(s for _, s in orphans)
    print(f"  Orphans (in R2, not in manifest): {len(orphans):,} ({orphan_bytes / 1024 / 1024:.1f} MB)")
    print()

    if not orphans:
        print("Nothing to clean. Exiting.")
        return 0

    # Step 4: classify orphans for the report
    models = ("eleven_v4", "eleven_v3", "eleven_flash_v2_5")
    breakdown = {m: sum(1 for k, _ in orphans if m in k) for m in models}
    legacy_no_model = len(orphans) - sum(breakdown.values())
    print(f"=== Orphan breakdown ===")
    print(f"  Legacy Multilingual (no model_id segment): {legacy_no_model:,}")
    for m, n in breakdown.items():
        print(f"  Superseded / unreferenced {m}: {n:,}")
    print()

    # Sample
    print(f"=== Sample orphans (first {args.sample}) ===")
    for k, s in orphans[: args.sample]:
        print(f"  {s:>7} bytes  {k}")
    print()

    if not args.confirm:
        print("Dry-run mode. Re-run with --confirm to delete.")
        return 0

    # Step 5: batch-delete
    print(f"Deleting {len(orphans):,} orphans in batches of {BATCH_SIZE}...")
    t0 = time.time()
    deleted = 0
    errored = 0
    for i in range(0, len(orphans), BATCH_SIZE):
        batch = orphans[i : i + BATCH_SIZE]
        keys = [{"Key": k} for k, _ in batch]
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": keys, "Quiet": True},
        )
        errs = resp.get("Errors", [])
        deleted += len(batch) - len(errs)
        errored += len(errs)
        for e in errs[:3]:
            print(f"  delete error: {e.get('Key')} → {e.get('Message')}", file=sys.stderr)
        if (i // BATCH_SIZE + 1) % 5 == 0 or i + BATCH_SIZE >= len(orphans):
            elapsed = time.time() - t0
            print(f"  {deleted:,} deleted ({errored} errored) in {elapsed:.1f}s")
    elapsed = time.time() - t0
    print()
    print(f"=== Cleanup summary ===")
    print(f"  Deleted: {deleted:,} ({orphan_bytes / 1024 / 1024:.1f} MB freed)")
    print(f"  Errored: {errored}")
    print(f"  Wall:    {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
