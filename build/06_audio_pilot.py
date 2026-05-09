"""Stage 6 — Audio pilot entry point.

Generates audio for the first N (default 500) senses, measures per-voice
loudness baselines, uploads to R2, validates with ASR roundtrip, and writes
the human-readable transcript log.

Usage:
    .venv/bin/python build/06_audio_pilot.py --preflight
    .venv/bin/python build/06_audio_pilot.py --pilot-size 500 --confirm
    .venv/bin/python build/06_audio_pilot.py --pilot-size 50 --confirm --concurrency 4

Tail the live transcript while it runs:
    tail -f audit/06_audio_transcript.log

Snapshot from a second terminal:
    .venv/bin/python build/audio_status.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.stage_6 import DEFAULT_CONCURRENCY, DEFAULT_MAX_ATTEMPTS_PER_CLIP, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 6 — Audio pilot")
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=500,
        help="Number of senses (= word + example clips per sense). Default 500.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Parallel workers. Default {DEFAULT_CONCURRENCY}.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_PER_CLIP,
        help=f"Per-clip retries before HUMAN review. Default {DEFAULT_MAX_ATTEMPTS_PER_CLIP}.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Print cost report and exit (no API calls).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually spend money on the API.",
    )
    args = parser.parse_args()

    summary = run(
        pilot_size=args.pilot_size,
        concurrency=args.concurrency,
        max_attempts_per_clip=args.max_attempts,
        dry_run=args.preflight,
        confirm=args.confirm,
    )
    print(f"\n[summary] {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
