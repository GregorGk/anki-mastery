#!/usr/bin/env python
"""Standalone snapshot tool for Stage 5.5 progress.

Reads `audit/055_progress.jsonl` and prints a current state summary.
Run any time during or after a Stage 5.5 run.

Run:
    python3 build/audit_status.py
    python3 build/audit_status.py --progress audit/055_progress.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROGRESS = REPO_ROOT / "audit" / "055_progress.jsonl"


def parse_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5.5 progress snapshot")
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--total", type=int, default=5725, help="Expected total rows (for ETA).")
    args = parser.parse_args()

    events = parse_events(args.progress)
    if not events:
        print(f"No events in {args.progress}", file=sys.stderr)
        return 1

    started: dict[str, str] = {}
    completed_at: dict[str, str] = {}
    completed_verdicts: list[str] = []
    completed_latencies: list[int] = []
    errored: dict[str, str] = {}

    for e in events:
        sid = e.get("sense_id", "")
        ev = e.get("event", "")
        if ev == "started":
            started[sid] = e.get("started_at", "")
        elif ev == "completed":
            completed_at[sid] = e.get("completed_at", "")
            completed_verdicts.append(e.get("verdict", ""))
            completed_latencies.append(int(e.get("latency_ms", 0)))
            started.pop(sid, None)
        elif ev == "errored":
            errored[sid] = e.get("error_type", "unknown")
            started.pop(sid, None)

    in_flight = sorted(started.items(), key=lambda kv: kv[1])  # by start time

    # Estimate "stuck" via timestamp comparison
    now_iso = datetime.now(timezone.utc).isoformat()
    stuck = []
    for sid, started_at_str in in_flight:
        try:
            started_dt = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
            if elapsed > 120:
                stuck.append((sid, int(elapsed)))
        except Exception:
            pass

    # Compute ETA
    done_count = len(completed_verdicts)
    avg_lat_ms = sum(completed_latencies) / len(completed_latencies) if completed_latencies else 0
    if completed_latencies:
        first_complete = min(
            datetime.fromisoformat(completed_at[sid].replace("Z", "+00:00"))
            for sid in completed_at
            if completed_at[sid]
        )
        elapsed_sec = (datetime.now(timezone.utc) - first_complete).total_seconds()
        rate = done_count / elapsed_sec if elapsed_sec > 0 else 0
        remaining = max(0, args.total - done_count)
        eta_sec = int(remaining / rate) if rate > 0 else 0
    else:
        eta_sec = 0

    verdict_counts = Counter(completed_verdicts)

    print(f"Stage 5.5 progress snapshot — {now_iso}")
    print("─" * 60)
    print(f"Source file:           {args.progress.name}")
    print(f"Total rows expected:   {args.total:,}")
    print(f"Completed:             {done_count:,}  ({done_count/args.total*100:.1f}%)")
    print(f"In-flight:             {len(in_flight)}")
    print(f"Errored (final):       {len(errored)}")
    print(f"Stuck (>120s):         {len(stuck)}")
    if stuck:
        for sid, secs in stuck[:5]:
            print(f"   sense_id {sid}: running {secs}s")
    print()
    print(f"Avg latency:           {avg_lat_ms/1000:.2f}s")
    if eta_sec > 0:
        print(f"ETA:                   ~{eta_sec // 60} min")
    print()
    print("Verdict distribution (so far):")
    for v in ("pass", "regenerate", "human_review"):
        c = verdict_counts.get(v, 0)
        pct = c / done_count * 100 if done_count else 0
        print(f"  {v:14s} {c:5,}  ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
