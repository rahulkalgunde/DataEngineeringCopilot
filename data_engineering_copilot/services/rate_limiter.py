"""Token-bucket rate limiter."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(
        self,
        max_calls: int,
        period_seconds: float,
        refill_interval: float | None = None,
    ) -> None:
        self._max_calls = max_calls
        self._period = period_seconds
        self._refill_interval = refill_interval or period_seconds
        self._tokens = max_calls
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill_count = int(elapsed / self._refill_interval)
        if refill_count > 0:
            self._tokens = min(self._max_calls, self._tokens + refill_count)
            self._last_refill = now

    def allow(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens > 0:
                self._tokens -= 1
                return True
            return False
