"""In-memory fixed-window rate limiter (Phase 20 fix #6).

Hand-rolled rather than pulling in a library (e.g. slowapi, a Redis-backed
limiter): this API's routes are plain `def` handlers, which Starlette runs
in a threadpool, so a per-process counter guarded by a lock is enough - the
same single-process assumption every `lru_cache` singleton in
dependencies.py already makes. It is NOT safe across multiple API
processes/replicas (each would keep its own counters, so the effective
limit multiplies by the replica count) - a real limitation, documented in
the README rather than hidden.
"""

import threading
import time


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    def allow(self, client_id: str) -> bool:
        """True if `client_id` may make another request right now, and
        records the attempt either way (a rejected request still counts
        towards the window, so a client can't probe for free)."""
        now = time.monotonic()
        with self._lock:
            window_start, count = self._windows.get(client_id, (now, 0))
            if now - window_start >= self._window_seconds:
                window_start, count = now, 0
            count += 1
            self._windows[client_id] = (window_start, count)
            return count <= self._max_requests
