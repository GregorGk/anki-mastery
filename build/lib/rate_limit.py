"""Shared rate-limit primitives for high-volume LLM stages.

`AnthropicClient` provides retry / backoff / prompt caching, but **does not**
enforce a local requests-per-minute (RPM) ceiling. Stages that fan out > 25
workers (Stage 14 onward) need a thread-safe RPM cap so we stay under
Anthropic's per-workspace limits even under bursty load.

`SlidingWindowRateLimiter` is the chosen primitive: it tracks the wall-clock
timestamps of the last `max_rpm` API calls and forces a worker to sleep when
adding the next call would exceed the rate. Sliding-window (rather than a
token bucket) so the rate is bounded over any 60-second window — token
buckets allow brief overshoot via accumulated tokens, which we don't want
against an external SLA.

Threadsafe by an internal `threading.Lock`. Workers share one limiter.

Usage:
    limiter = SlidingWindowRateLimiter(max_rpm=3200)
    # inside each worker:
    limiter.wait_before_call()
    decision = client.call_tool(...)
"""
from __future__ import annotations

import collections
import threading
import time


class SlidingWindowRateLimiter:
    """Sliding-window RPM limiter, threadsafe.

    Args:
        max_rpm: Maximum requests per rolling 60-second window. `0` disables
            the limiter (calls return immediately) — convenient for tests
            and CLI flags that want an "unbounded" sentinel.
    """

    def __init__(self, max_rpm: int) -> None:
        self.max_rpm = int(max_rpm)
        self._lock = threading.Lock()
        # Wall-clock timestamps (monotonic) of the last `max_rpm` calls.
        self._calls: collections.deque[float] = collections.deque()

    def wait_before_call(self) -> None:
        """Block until adding one more call keeps us under the RPM budget.

        Called at the top of every worker before the API call. Returns
        immediately if `max_rpm <= 0` (unbounded).
        """
        if self.max_rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps older than 60 s — they're outside the window.
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.max_rpm:
                    self._calls.append(now)
                    return
                # Sleep until the oldest call ages out of the window.
                sleep_for = 60.0 - (now - self._calls[0]) + 0.05
            time.sleep(max(sleep_for, 0.05))

    def snapshot(self) -> dict:
        """Return a snapshot for logging. Not strictly consistent under load
        but good enough for periodic progress prints."""
        with self._lock:
            n = len(self._calls)
            if n == 0:
                return {"max_rpm": self.max_rpm, "in_window": 0, "rpm": 0.0}
            now = time.monotonic()
            # Effective realized RPM = calls-in-window / window-span × 60.
            span = max(now - self._calls[0], 1e-3)
            return {
                "max_rpm": self.max_rpm,
                "in_window": n,
                "rpm": n / span * 60.0,
            }
