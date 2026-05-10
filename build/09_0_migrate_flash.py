"""Stage 9 / Step 0 — Migrate the deck from Multilingual v2 → Flash v2.5.

Full re-render of all 11,450 clips (5,725 word + 5,725 example) on
eleven_flash_v2_5 with a 1-rule alias dict (hospital → ospitau).

Pre-flight prints a budget summary and waits for the operator to type
`GO`. On GO, marks every manifest row as pending (bumping version with
the model_id baked into filenames), then delegates to stage_6.run()
with concurrency=20 and fail-fast-on-429.

Tee'd to audit/09_0_migrate_flash.log so monitoring is possible from
another terminal via `tail -f`.

Usage:
    .venv/bin/python build/09_0_migrate_flash.py
        [--concurrency 20]   # match Pro/Flash ceiling
        [--dry-run]          # pre-flight only, no API calls
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    bump_version,
    read_manifest,
    write_manifest,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
META_PATH = DATA_DIR / "_audio_dictionary_meta.tsv"
LOG_PATH = AUDIT_DIR / "09_0_migrate_flash.log"

FLASH_MODEL_ID = "eleven_flash_v2_5"
TARGET_CONCURRENCY = 20  # Pro tier + Flash ceiling per ElevenLabs docs


class TeeStdout:
    """Mirror stdout to both terminal and log file for tail-from-another-terminal."""

    def __init__(self, *streams) -> None:
        self.streams = streams
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:  # noqa: BLE001
                    pass
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass


def _latest_dictionary_locator() -> dict | None:
    if not META_PATH.exists():
        return None
    rows = read_tsv(META_PATH)
    if not rows:
        return None
    last = rows[-1]
    if not last.get("dictionary_id") or not last.get("version_id"):
        return None
    return {
        "pronunciation_dictionary_id": last["dictionary_id"],
        "version_id": last["version_id"],
    }


def _budget_summary(manifest_rows: list[dict]) -> tuple[int, int, int]:
    """Return (n_total, n_chars, n_pending) from the manifest."""
    n_chars = 0
    n_pending = 0
    for r in manifest_rows:
        text = r.get("text_input", "")
        n_chars += len(text)
        if r.get("status", "") != "uploaded":
            n_pending += 1
    return len(manifest_rows), n_chars, n_pending


def _print_preflight_box(*, n_total: int, n_chars: int, locator: dict, concurrency: int, dry_run: bool) -> None:
    expected_chars = int(n_chars * 1.10)  # +10% retry buffer (realistic)
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  STAGE 9 MIGRATION PRE-FLIGHT                                       │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Model:        {FLASH_MODEL_ID:<55}│")
    print(f"│  Concurrency:  {concurrency:<55}│")
    print(f"│  Clips total:  {n_total:>6} (re-render every row)                          │")
    print(f"│  Chars base:   {n_chars:>6,}                                              │")
    print(f"│  Chars w/10%:  {expected_chars:>6,}  (realistic retry budget)                  │")
    print(f"│  Dict:         1 rule (hospital → ospitau)                          │")
    print(f"│  Dict id:      {locator['pronunciation_dictionary_id']:<55}│")
    print(f"│  Dict ver:     {locator['version_id']:<55}│")
    print(f"│  Filenames:    {{sid}}-{{clip}}-{FLASH_MODEL_ID}-v{{N}}.mp3              │")
    print(f"│  Fail-fast:    429 → ABORT immediately                              │")
    print("│  Monitor:      tail -f audit/09_0_migrate_flash.log                 │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    if dry_run:
        print()
        print("--dry-run: no API calls. Stopping here.")
        return
    print()
    print("Make sure PAYG is toggled ON in your ElevenLabs account if you want")
    print("a safety net against ~$3.50 worst-case overage. Otherwise, hitting")
    print("the monthly cap mid-migration will stop the run (resume after reset).")
    print()


def _mark_all_pending(manifest_rows: list[dict], public_base: str) -> int:
    """Mark every manifest row as pending; bump version with Flash model_id
    baked into filenames. Returns count of rows touched."""
    n = 0
    for r in manifest_rows:
        bump_version(r, public_base=public_base, model_id=FLASH_MODEL_ID)
        r["status"] = "pending"
        r["notes"] = (r.get("notes", "") + " | stage_9_flash_migration").strip(" |")
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--concurrency", type=int, default=TARGET_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pre-flight and exit; no API calls or manifest changes.")
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set in environment.", file=sys.stderr)
        return 1

    # Pre-flight checks
    locator = _latest_dictionary_locator()
    if not locator:
        print("ERROR: no dictionary locator in _audio_dictionary_meta.tsv. "
              "Run 08_4_upload_dictionary.py first.", file=sys.stderr)
        return 1

    manifest_rows = read_manifest()
    if not manifest_rows:
        print(f"ERROR: empty manifest at {DEFAULT_MANIFEST_PATH}", file=sys.stderr)
        return 1

    n_total, n_chars, n_pending_before = _budget_summary(manifest_rows)
    _print_preflight_box(
        n_total=n_total, n_chars=n_chars,
        locator=locator, concurrency=args.concurrency,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    # GO prompt
    sys.stdout.write("Type GO to proceed (anything else cancels): ")
    sys.stdout.flush()
    user_input = sys.stdin.readline().strip()
    if user_input != "GO":
        print(f"Got {user_input!r}, not 'GO' — cancelling. No changes made.")
        return 0

    # Tee stdout from here on
    log_file = LOG_PATH.open("w", encoding="utf-8")
    log_file.write(f"=== stage 9 migration started {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    log_file.flush()
    original_stdout = sys.stdout
    sys.stdout = TeeStdout(original_stdout, log_file)  # type: ignore[assignment]

    print(f"[09_0] GO received. Marking all {n_total} manifest rows pending...")

    # Mark all rows pending (bump version with model_id baked in)
    from build.lib.r2_client import R2Client  # noqa: E402
    r2 = R2Client()
    public_base = r2.config.public_base
    n_touched = _mark_all_pending(manifest_rows, public_base)
    write_manifest(manifest_rows)
    print(f"[09_0] Marked {n_touched} rows pending (version bumped, status cleared).")
    print(f"[09_0] Sample new object_key: {manifest_rows[0]['object_key']}")
    print(f"[09_0] Sample new url: {manifest_rows[0]['url']}")

    # Delegate to stage_6.run()
    print(f"[09_0] Delegating to stage_6.run() with concurrency={args.concurrency}, "
          f"model={FLASH_MODEL_ID}, fail_fast_on_429=True, dict={locator['pronunciation_dictionary_id']}...")
    print()

    from build.stage_6 import run as stage_6_run

    result = stage_6_run(
        pilot_size=None,
        concurrency=args.concurrency,
        confirm=True,
        pronunciation_dict_locators=[locator],
        model_id=FLASH_MODEL_ID,
        fail_fast_on_429=True,
    )

    print()
    print("=== Stage 9 migration summary ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    sys.stdout = original_stdout  # type: ignore[assignment]
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
