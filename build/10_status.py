"""Snapshot for the Stage 10 EN audio render.

Reads `audit/10_0_render_en_progress.jsonl` and prints:
  - % done
  - elapsed / ETA
  - in-flight count
  - last few completed clips

Usage:
    .venv/bin/python build/10_status.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JSONL = REPO / "audit" / "10_0_render_en_progress.jsonl"


def fmt_dur(secs: float) -> str:
    if secs <= 0:
        return "0s"
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def main() -> int:
    if not JSONL.exists():
        print(f"No progress file yet at {JSONL}", file=sys.stderr)
        return 1

    events: list[dict] = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Find the most recent run_started marker — that's where the current
    # run begins. Drop any events before it.
    last_run_idx = None
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("event") == "run_started":
            last_run_idx = i
            break
    if last_run_idx is None:
        print("No run_started marker yet — render hasn't begun.")
        return 0
    run_ev = events[last_run_idx]
    events = events[last_run_idx:]

    total = run_ev.get("pending_total", 0)
    started_at = parse_iso(run_ev.get("started_at", ""))
    started_count = sum(1 for e in events if e.get("event") == "started")
    completed_count = sum(1 for e in events if e.get("event") == "completed")
    errored_count = sum(1 for e in events if e.get("event") == "errored")
    in_flight = max(0, started_count - completed_count - errored_count)

    done = completed_count + errored_count
    pct = (done / total * 100) if total else 0.0

    now = datetime.now(timezone.utc)
    elapsed = (now - started_at).total_seconds() if started_at else 0
    rate = done / elapsed if elapsed > 0 else 0
    remaining = max(0, total - done)
    eta_sec = remaining / rate if rate > 0 else 0

    # Recent completed (last 5)
    completed_events = [e for e in events if e.get("event") == "completed"]
    last_n = completed_events[-5:]

    print(f"=== Stage 10 — EN render snapshot ===")
    print(f"  Started:    {run_ev.get('started_at', '?')}")
    print(f"  Total:      {total:,}")
    print(f"  Done:       {done:,} ({pct:.1f}%)")
    print(f"    uploaded: {completed_count:,}")
    print(f"    errored:  {errored_count:,}")
    print(f"  In-flight:  {in_flight}")
    print(f"  Remaining:  {remaining:,}")
    print(f"  Elapsed:    {fmt_dur(elapsed)}")
    print(f"  ETA:        {fmt_dur(eta_sec)}")
    print(f"  Rate:       {rate:.2f} clips/s")
    if last_n:
        print(f"  Last {len(last_n)} completions:")
        for e in last_n:
            ms = e.get("latency_ms", 0)
            print(f"    {e.get('key', '?')}  {ms}ms  ({e.get('decision', '?')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
