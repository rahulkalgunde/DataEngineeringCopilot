"""Redis-backed sliding-window rate limiter with in-memory fallback."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# In-memory fallback when Redis is unavailable
_IN_MEMORY_STORE: dict[str, list[float]] = {}
_IN_MEMORY_LOCK = threading.Lock()
_last_cleanup: float = time.time()
_CLEANUP_INTERVAL: float = 300  # 5 minutes


def _cleanup_in_memory_store() -> None:
    """Remove stale entries from the in-memory rate limit store to prevent unbounded growth."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    cutoff = now - 120  # Keep entries from the last 2 minutes
    empty_keys = []
    for key, timestamps in _IN_MEMORY_STORE.items():
        _IN_MEMORY_STORE[key] = [t for t in timestamps if t > cutoff]
        if not _IN_MEMORY_STORE[key]:
            empty_keys.append(key)
    for key in empty_keys:
        del _IN_MEMORY_STORE[key]

# Per-route defaults
DEFAULT_LIMITS: dict[str, tuple[int, int]] = {
    "/api/v1/ask": (60, 60),  # 60 req / 60 s
    "/api/v1/ingest": (10, 60),  # 10 req / 60 s
}


def _redis_client():
    """Return a Redis client or None if unavailable."""
    try:
        from data_engineering_copilot.workers.progress import get_redis_client

        return get_redis_client()
    except Exception:
        return None


def sliding_window_allow(
    path: str,
    client_ip: str,
    max_calls: int,
    period_seconds: int,
) -> bool:
    """Check whether a request is allowed under the sliding window.

    Uses Redis ZSET if available, falls back to in-memory token bucket.
    """
    redis = _redis_client()
    key = f"ratelimit:{path}:{client_ip}"
    now = time.time()
    window_start = now - period_seconds

    if redis is not None:
        try:
            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, "-inf", window_start)
            pipe.zadd(key, {f"{now}:{client_ip}": now})
            pipe.zcard(key)
            pipe.expire(key, period_seconds)
            results = pipe.execute()
            request_count = results[2]
            return request_count <= max_calls
        except Exception as exc:
            logger.warning("Redis rate limit check failed, falling back to in-memory: %s", exc)

    # In-memory fallback
    _cleanup_in_memory_store()
    with _IN_MEMORY_LOCK:
        entries = _IN_MEMORY_STORE.get(key, [])
        entries = [t for t in entries if t > window_start]
        if len(entries) < max_calls:
            entries.append(now)
            _IN_MEMORY_STORE[key] = entries
            return True
        _IN_MEMORY_STORE[key] = entries
        return False


class RateLimiter:
    """Per-route sliding-window rate limiter backed by Redis.

    Falls back to in-memory when Redis is unavailable.
    """

    def __init__(
        self,
        path: str = "",
        max_calls: int = 60,
        period_seconds: int = 60,
    ) -> None:
        self._path = path
        self._max_calls = max_calls
        self._period_seconds = period_seconds

    def allow(self, client_ip: str = "") -> bool:
        """Check whether a request from *client_ip* is allowed."""
        return sliding_window_allow(
            path=self._path,
            client_ip=client_ip,
            max_calls=self._max_calls,
            period_seconds=self._period_seconds,
        )
