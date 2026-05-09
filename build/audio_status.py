"""Standalone snapshot tool for Stage 6/7 audio runs.

Reads `audit/06_audio.jsonl` and prints a current snapshot — useful from a
second terminal while the run is in progress, or after it completes.

Usage:
    .venv/bin/python build/audio_status.py
    .venv/bin/python build/audio_status.py --jsonl audit/06_audio.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "audit" / "06_audio.jsonl"

STUCK_THRESHOLD_SEC = 120.0


def parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 6/7 audio status snapshot")
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--last-n", type=int, default=10, help="Last N transcript events to show")
    args = parser.parse_args()

    p = Path(args.jsonl)
    if not p.exists():
        print(f"(no audit log yet at {p})", file=sys.stderr)
        return 0

    started: dict[str, dict] = {}     # key -> started event
    completed: list[dict] = []         # asr_completed events (chronological)
    errored: list[dict] = []
    started_count = 0
    decisions: Counter[str] = Counter()
    cost_usd = 0.0

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("event")
            sid = ev.get("sense_id", "")
            ctype = ev.get("clip_type", "")
            key = f"{sid}|{ctype}|{ev.get('attempt', '')}"
            if kind == "started":
                started[key] = ev
                started_count += 1
            elif kind == "asr_completed":
                completed.append(ev)
                decisions[ev.get("decision", "?")] += 1
                cost_usd += float(ev.get("cost_usd") or 0.0)
                started.pop(key, None)
            elif kind == "errored":
                errored.append(ev)
                started.pop(key, None)
            elif kind == "regenerated":
                pass

    # Find stuck (still in `started` and timestamp older than threshold)
    now = datetime.now(timezone.utc)
    stuck: list[tuple[str, float]] = []
    for key, ev in started.items():
        ts = parse_iso(ev.get("started_at", ""))
        if ts is None:
            continue
        elapsed = (now - ts).total_seconds()
        if elapsed > STUCK_THRESHOLD_SEC:
            stuck.append((key, elapsed))

    # ETA estimate
    completed_total = len(completed)
    in_flight = len(started)
    if completed:
        first = parse_iso(completed[0].get("completed_at", ""))
        last = parse_iso(completed[-1].get("completed_at", ""))
        if first and last:
            elapsed_run = (last - first).total_seconds()
            rate = completed_total / max(1.0, elapsed_run)
        else:
            rate = 0.0
    else:
        rate = 0.0

    # Print snapshot
    print("Stage 6/7 audio status")
    print("──────────────────────")
    print(f"  events:           started={started_count:,}  asr_completed={completed_total:,}  errored={len(errored):,}")
    print(f"  in-flight:        {in_flight}  stuck (>{int(STUCK_THRESHOLD_SEC)}s): {len(stuck)}")
    if rate:
        print(f"  rate:             {rate:.2f} clips/sec  avg latency {(1.0/rate):.2f}s")
    print(f"  cost so far:      ${cost_usd:.2f}")
    print()
    print("  decision distribution:")
    for k in ("PASS", "REGEN", "HUMAN", "ERR"):
        print(f"    {k:8} {decisions.get(k, 0):,}")
    other = sum(v for k, v in decisions.items() if k not in ("PASS", "REGEN", "HUMAN", "ERR"))
    if other:
        print(f"    other     {other:,}")
    if stuck:
        print()
        print("  stuck clips:")
        for key, secs in stuck[:10]:
            print(f"    {key}  {int(secs)}s")
    if completed:
        print()
        print(f"  last {min(args.last_n, len(completed))} ASR decisions:")
        for ev in completed[-args.last_n:]:
            sid = ev.get("sense_id", "")
            ctype = ev.get("clip_type", "")
            d = ev.get("decision", "?")
            sim = ev.get("levenshtein_similarity", "?")
            inp = (ev.get("input_text") or "")[:30]
            tr = (ev.get("asr_transcript") or "")[:30]
            print(f"    {d:6}  {sid} {ctype:4}  {inp:<32}  ->  {tr:<32}  sim={sim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
