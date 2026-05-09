"""Generic live progress tracker for long-running pipeline stages.

Generalized from Stage 5.5's `ProgressTracker`. Adds:
- Configurable decision axes (not just pass/regenerate/human_review).
- Ring buffer of last N events for stdout snapshot ("Last 3 clips").
- Cost accumulation in USD.
- Tail-friendly format helpers.

Stage 5.5 keeps its own copy in `build/stage_55.py` for backwards
compatibility; new stages should import from this module.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STUCK_THRESHOLD_SEC = 120.0
SUMMARY_INTERVAL_SEC = 30.0
DEFAULT_RING_BUFFER_SIZE = 5


@dataclass
class RingEntry:
    """One human-readable line for the 'Last N events' display."""

    timestamp: str
    rendered: str  # pre-formatted line ready to print


class ProgressTracker:
    """Thread-safe progress tracker with JSONL event log and decision counters.

    Parameters
    ----------
    progress_path:
        Path to JSONL file for lifecycle events.
    total:
        Total number of work items expected (drives % done + ETA).
    decision_axes:
        Names of decision categories to count (e.g. ('pass','regen','human','err')).
        Defaults to common audit-style verdicts.
    stage_label:
        Short tag for the periodic stdout summary (e.g. 'Stage 6').
    ring_buffer_size:
        How many recent events to keep for the snapshot's "Last N" display.
    """

    def __init__(
        self,
        progress_path: Path,
        total: int,
        *,
        decision_axes: tuple[str, ...] = ("pass", "regenerate", "human_review"),
        stage_label: str = "Stage",
        ring_buffer_size: int = DEFAULT_RING_BUFFER_SIZE,
    ) -> None:
        self.progress_path = progress_path
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.total = total
        self.stage_label = stage_label
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self.in_flight: dict[str, float] = {}  # key -> monotonic start time
        self.done = 0
        self.error_count = 0
        self.cost_usd = 0.0
        self.decisions: dict[str, int] = {axis: 0 for axis in decision_axes}
        self.latencies_ms: list[int] = []
        self.start_wall = time.time()
        self.start_iso = datetime.now(timezone.utc).isoformat()
        self.ring: deque[RingEntry] = deque(maxlen=ring_buffer_size)

    # --- JSONL emit ------------------------------------------------------- #

    def _emit(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._file_lock, self.progress_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # --- Lifecycle events ------------------------------------------------- #

    def started(self, key: str, *, attempt: int = 0, **extra) -> None:
        now = time.monotonic()
        with self._lock:
            self.in_flight[key] = now
        payload = {
            "event": "started",
            "key": key,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt,
        }
        payload.update(extra)
        self._emit(payload)

    def completed(
        self,
        key: str,
        *,
        decision: str,
        attempt: int = 0,
        cost_usd: float = 0.0,
        rendered: str | None = None,
        **extra,
    ) -> int:
        """Record a completed work item.

        Returns the latency in milliseconds (handy for the per-clip log).
        """
        now = time.monotonic()
        with self._lock:
            start = self.in_flight.pop(key, now)
            latency_ms = int((now - start) * 1000)
            self.latencies_ms.append(latency_ms)
            self.done += 1
            self.decisions[decision] = self.decisions.get(decision, 0) + 1
            self.cost_usd += cost_usd
            if rendered is not None:
                self.ring.append(
                    RingEntry(
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                        rendered=rendered,
                    )
                )
        payload = {
            "event": "completed",
            "key": key,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "decision": decision,
            "attempt": attempt,
            "cost_usd": cost_usd,
        }
        payload.update(extra)
        self._emit(payload)
        return latency_ms

    def errored(
        self,
        key: str,
        *,
        error_type: str,
        error_msg: str,
        attempt: int = 0,
        rendered: str | None = None,
        **extra,
    ) -> None:
        with self._lock:
            self.in_flight.pop(key, None)
            self.error_count += 1
            if rendered is not None:
                self.ring.append(
                    RingEntry(
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                        rendered=rendered,
                    )
                )
        payload = {
            "event": "errored",
            "key": key,
            "errored_at": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type,
            "error_msg": error_msg[:300],
            "attempt": attempt,
        }
        payload.update(extra)
        self._emit(payload)

    # --- Stuck detection -------------------------------------------------- #

    def stuck_check(self, threshold_sec: float = STUCK_THRESHOLD_SEC) -> list[tuple[str, float]]:
        """Return list of (key, secs_in_flight) for items running > threshold."""
        now = time.monotonic()
        stuck: list[tuple[str, float]] = []
        with self._lock:
            for key, start in self.in_flight.items():
                elapsed = now - start
                if elapsed > threshold_sec:
                    stuck.append((key, elapsed))
        return stuck

    # --- Snapshots --------------------------------------------------------- #

    def snapshot(self) -> dict:
        with self._lock:
            in_flight_count = len(self.in_flight)
            done = self.done
            errored = self.error_count
            decisions = dict(self.decisions)
            lats = list(self.latencies_ms)
            ring = list(self.ring)
            cost_usd = self.cost_usd
        avg_lat = sum(lats) / len(lats) if lats else 0
        p99_lat = sorted(lats)[int(0.99 * len(lats))] if lats else 0
        elapsed = time.time() - self.start_wall
        rate = done / elapsed if elapsed > 0 else 0
        remaining = max(0, self.total - done)
        eta_sec = int(remaining / rate) if rate > 0 else 0
        return {
            "stage_label": self.stage_label,
            "total": self.total,
            "done": done,
            "in_flight": in_flight_count,
            "errored": errored,
            "stuck": len(self.stuck_check()),
            "decisions": decisions,
            "avg_latency_ms": int(avg_lat),
            "p99_latency_ms": int(p99_lat),
            "elapsed_sec": int(elapsed),
            "eta_sec": eta_sec,
            "start_iso": self.start_iso,
            "cost_usd": round(cost_usd, 4),
            "last_events": [{"timestamp": r.timestamp, "rendered": r.rendered} for r in ring],
        }

    # --- Format helpers ---------------------------------------------------- #

    def format_summary(self) -> str:
        s = self.snapshot()
        pct = (s["done"] / s["total"] * 100) if s["total"] else 0.0
        eta_min = s["eta_sec"] // 60
        elapsed_min = s["elapsed_sec"] // 60
        decisions_str = " ".join(
            f"{k}={v}" for k, v in s["decisions"].items()
        )
        head = (
            f"[{s['stage_label']}  {s['done']:,}/{s['total']:,} ({pct:.1f}%) | "
            f"{s['in_flight']} in-flight | "
            f"{s['stuck']} stuck (>{int(STUCK_THRESHOLD_SEC)}s) | "
            f"elapsed {elapsed_min}m | "
            f"avg {s['avg_latency_ms'] / 1000:.1f}s | "
            f"ETA {eta_min}m | "
            f"spent ${s['cost_usd']:.2f}]"
        )
        lines = [head, f"  decisions: {decisions_str} errored={s['errored']}"]
        if s["last_events"]:
            lines.append("  Last events:")
            for ev in s["last_events"]:
                lines.append(f"    {ev['timestamp']}  {ev['rendered']}")
        return "\n".join(lines)


class SummaryPrinter(threading.Thread):
    """Daemon thread that prints `tracker.format_summary()` every interval seconds."""

    def __init__(
        self,
        tracker: ProgressTracker,
        *,
        interval: float = SUMMARY_INTERVAL_SEC,
        stream=sys.stderr,
    ):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.interval = interval
        self.stream = stream
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            print(self.tracker.format_summary(), file=self.stream, flush=True)
            stuck = self.tracker.stuck_check()
            if stuck:
                for key, secs in stuck[:3]:
                    print(
                        f"  [warn] stuck: {key} running {int(secs)}s",
                        file=self.stream,
                        flush=True,
                    )

    def stop(self) -> None:
        self.stop_event.set()
