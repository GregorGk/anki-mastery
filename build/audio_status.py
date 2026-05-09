"""Snapshot tool for Stage 6/7 audio runs — current run only.

Auto-detects the current run's start by finding the most recent >5-minute
gap in the JSONL audit log (same heuristic as
build/surgical_voice_swap_recovery.py). Filters every metric and event
list to only the current run, so historical pilot / surgical / closed-loop
events don't pollute the readout.

Reads `data/_audio_manifest.tsv` to determine total expected work for
the current run. Computes:
  - Big % done
  - total / done / remaining counts
  - elapsed time + ETA in human form
  - rate (clips/sec)
  - decision distribution (current run only)
  - last N ASR decisions (current run only)

Usage:
    .venv/bin/python build/audio_status.py
    .venv/bin/python build/audio_status.py --jsonl audit/06_audio.jsonl --last-n 10

Exit code:
    0 always (snapshot tool, no failure semantics)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "audit" / "06_audio.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "_audio_manifest.tsv"

STUCK_THRESHOLD_SEC = 120.0
RUN_GAP_THRESHOLD_SEC = 300.0  # >5 min gap = new run boundary


def parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def event_timestamp(ev: dict) -> datetime | None:
    """Pick the most relevant timestamp from an event."""
    for k in ("started_at", "completed_at", "errored_at", "ts"):
        v = ev.get(k)
        if v:
            return parse_iso(v)
    return None


def find_current_run_start(events: list[dict]) -> tuple[datetime | None, dict | None]:
    """Return (run_start_ts, run_started_event_or_None).

    Preferred: the most recent `run_started` marker event written by the
    orchestrator at launch. This is unambiguous and unaffected by
    timestamp gaps within a run (e.g., a stuck network call).

    Fallback: walk timestamps in reverse and return the first gap >
    threshold (5 min). Used for audit logs from before the marker was
    introduced, or if the orchestrator crashed before writing one.
    """
    # Prefer explicit run_started marker (most recent)
    run_started_events = [e for e in events if e.get("event") == "run_started"]
    if run_started_events:
        latest = max(
            run_started_events,
            key=lambda e: parse_iso(e.get("started_at", "")) or datetime.min.replace(tzinfo=timezone.utc),
        )
        ts = parse_iso(latest.get("started_at", ""))
        if ts is not None:
            return ts, latest

    # Fallback: gap-based detection
    timestamps = sorted(t for ev in events if (t := event_timestamp(ev)))
    if not timestamps:
        return None, None
    if len(timestamps) == 1:
        return timestamps[0], None
    for i in range(len(timestamps) - 1, 0, -1):
        gap_sec = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if gap_sec > RUN_GAP_THRESHOLD_SEC:
            return timestamps[i], None
    return timestamps[0], None


def fmt_duration(seconds: float) -> str:
    """Format seconds as 'HhMm' or 'Mm Ss'."""
    if seconds <= 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def load_manifest_counts(path: Path) -> tuple[int, int]:
    """Return (pending, in_progress) counts from manifest. Total expected
    work for the current run = pending + uploading + done_in_current_run.
    """
    if not path.exists():
        return 0, 0
    pending = 0
    in_progress = 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        for r in reader:
            status = r.get("status", "")
            if status == "pending":
                pending += 1
            elif status == "uploading":
                in_progress += 1
    return pending, in_progress


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 6/7 audio status (current run only)")
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--last-n", type=int, default=10, help="Last N decisions to show")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    manifest_path = Path(args.manifest)

    if not jsonl_path.exists():
        print(f"(no audit log yet at {jsonl_path})", file=sys.stderr)
        return 0

    # Load all events
    all_events: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            all_events.append(ev)

    if not all_events:
        print("(audit log is empty)", file=sys.stderr)
        return 0

    # Detect current run start
    run_start, run_marker = find_current_run_start(all_events)
    if run_start is None:
        print("(no events with timestamps)", file=sys.stderr)
        return 0
    detection_label = "from run_started marker" if run_marker else "from gap heuristic"

    # Filter events to the current run
    current_events = [
        ev for ev in all_events
        if (ts := event_timestamp(ev)) is not None and ts >= run_start
    ]

    # Tally events
    started_keys: set[tuple[str, str]] = set()
    completed_keys: set[tuple[str, str]] = set()
    completed_events: list[dict] = []
    errored_events: list[dict] = []
    in_flight: dict[tuple[str, str], dict] = {}  # key -> last started event
    decisions: Counter[str] = Counter()
    cost_usd_sum = 0.0
    latencies_ms: list[int] = []

    for ev in current_events:
        sid = ev.get("sense_id", "")
        ctype = ev.get("clip_type", "")
        key = (sid, ctype)
        kind = ev.get("event")
        if kind == "started":
            started_keys.add(key)
            in_flight[key] = ev
        elif kind == "asr_completed":
            completed_keys.add(key)
            completed_events.append(ev)
            decisions[ev.get("decision", "?")] += 1
            cost_usd_sum += float(ev.get("cost_usd") or 0.0)
            latency = ev.get("latency_ms")
            if isinstance(latency, (int, float)) and latency > 0:
                latencies_ms.append(int(latency))
            in_flight.pop(key, None)
        elif kind == "errored":
            errored_events.append(ev)
            in_flight.pop(key, None)

    # Determine total work for the current run.
    # Work = manifest pending+uploading (still to do) + already done in current run.
    pending_in_manifest, uploading_in_manifest = load_manifest_counts(manifest_path)
    done = len(completed_keys)
    total = done + pending_in_manifest + uploading_in_manifest
    remaining = max(0, total - done)
    pct_done = (done / total * 100.0) if total else 0.0

    # Stuck detection (in-flight rows with very old started_at)
    now = datetime.now(timezone.utc)
    stuck: list[tuple[tuple[str, str], float]] = []
    for key, ev in in_flight.items():
        ts = event_timestamp(ev)
        if ts is None:
            continue
        elapsed_sec = (now - ts).total_seconds()
        if elapsed_sec > STUCK_THRESHOLD_SEC:
            stuck.append((key, elapsed_sec))

    # Rate / ETA
    elapsed_sec = (now - run_start).total_seconds()
    rate_per_sec = done / elapsed_sec if elapsed_sec > 0 else 0.0
    avg_latency_ms = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0
    eta_sec = (remaining / rate_per_sec) if rate_per_sec > 0 else 0.0

    # Render
    last_completed = completed_events[-args.last_n:] if completed_events else []

    stage_label = (run_marker or {}).get("stage", "Stage 6/7")
    print(f"{stage_label} audio status — current run only ({detection_label})")
    print("─" * 60)
    print(f"  Run started:  {run_start.isoformat()}  (elapsed {fmt_duration(elapsed_sec)})")
    print(f"  Progress:     {done:,} / {total:,}  ({pct_done:.1f}%)")
    print(f"  Remaining:    {remaining:,}  →  ETA {fmt_duration(eta_sec)}")
    print(f"  Rate:         {rate_per_sec:.2f} clips/sec    avg latency  {avg_latency_ms / 1000:.1f}s")
    in_flight_n = len(in_flight)
    stuck_n = len(stuck)
    print(f"  In flight:    {in_flight_n}    stuck (>{int(STUCK_THRESHOLD_SEC)}s):  {stuck_n}")
    print(f"  Cost so far:  ${cost_usd_sum:.2f}    errored:  {len(errored_events)}")
    print()

    # Decision distribution
    print("  Current-run decision distribution:")
    for tag in ("PASS", "REGEN", "HUMAN", "ERR"):
        print(f"    {tag:<8} {decisions.get(tag, 0):,}")
    other_tags = sum(v for k, v in decisions.items() if k not in ("PASS", "REGEN", "HUMAN", "ERR"))
    if other_tags:
        print(f"    other    {other_tags:,}")

    # Stuck warning
    if stuck:
        print()
        print(f"  ⚠ Stuck clips (>{int(STUCK_THRESHOLD_SEC)}s in flight):")
        for (sid, ctype), elapsed_s in stuck[:10]:
            print(f"    {sid} {ctype}  running {int(elapsed_s)}s")

    # Last N decisions
    if last_completed:
        print()
        print(f"  Last {len(last_completed)} ASR decisions:")
        for ev in last_completed:
            sid = ev.get("sense_id", "")
            ctype = ev.get("clip_type", "")
            d = ev.get("decision", "?")
            sim = ev.get("levenshtein_similarity", "?")
            inp = (ev.get("input_text") or "")[:32]
            tr = (ev.get("asr_transcript") or "")[:32]
            lat_ms = ev.get("latency_ms") or 0
            print(f"    {d:<6} {sid} {ctype:<7}  {inp:<33} -> {tr:<33}  sim={sim}  [{lat_ms / 1000:.1f}s]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
