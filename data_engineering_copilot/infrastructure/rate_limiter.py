"""Shared sliding-window rate limiter for LLM API providers.

Both embeddings and LLM generation share the same API key for a given
provider, so they must coordinate to stay under the same RPM / RPD limits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """Sliding-window RPM + daily-counter RPD limiter.

    Designed to be shared by multiple clients that share the same API
    key / rate limit budget (e.g. embeddings + LLM generation).

    Parameters
    ----------
    rpm_limit:
        Maximum requests per minute (sliding window).
    rpd_limit:
        Maximum requests per day (calendar-style window).
        Set to 0 to disable daily limit.
    """

    def __init__(self, rpm_limit: int = 20, rpd_limit: int = 1000) -> None:
        self._rpm_limit = rpm_limit
        self._rpd_limit = rpd_limit
        self._request_timestamps: deque[float] = deque()
        self._daily_count = 0
        self._daily_reset = time.time() + 86400
        self._lock = asyncio.Lock()
        self._last_rpm_hit = 0.0
        self._last_rpd_hit = 0.0

    async def acquire(self) -> None:
        """Block until a request slot is available under both RPM and RPD limits.

        Raises ``RuntimeError`` if the daily limit is exhausted.
        """
        async with self._lock:
            now = time.time()

            # Purge timestamps older than 60 seconds
            while self._request_timestamps and self._request_timestamps[0] < now - 60:
                self._request_timestamps.popleft()

            # Daily reset
            if now >= self._daily_reset:
                self._daily_count = 0
                self._daily_reset = now + 86400

            # RPD limit check (skip if rpd_limit is 0)
            if self._rpd_limit > 0 and self._daily_count >= self._rpd_limit:
                wait = self._daily_reset - now
                if wait > 0:
                    logger.warning(
                        "Daily limit (%d RPD) reached. Sleeping %.0fs until reset.",
                        self._rpd_limit,
                        wait,
                    )
                    self._last_rpd_hit = time.time()
                    await asyncio.sleep(wait)
                    # Reset after sleep
                    self._daily_count = 0
                    self._daily_reset = time.time() + 86400

            # RPM limit check
            if len(self._request_timestamps) >= self._rpm_limit:
                wait = self._request_timestamps[0] + 60 - now
                if wait > 0:
                    logger.debug(
                        "RPM limit (%d) reached. Sleeping %.1fs.",
                        self._rpm_limit,
                        wait,
                    )
                    self._last_rpm_hit = time.time()
                    await asyncio.sleep(wait)
                # Purge again after sleeping
                while self._request_timestamps and self._request_timestamps[0] < time.time() - 60:
                    self._request_timestamps.popleft()

            self._request_timestamps.append(time.time())
            self._daily_count += 1

    async def handle_429(self, response_headers: dict | None = None) -> None:
        """React to a 429 response by waiting for the appropriate ``Retry-After``.

        If the provider includes a ``Retry-After`` header that value is used;
        otherwise a default 60s backoff is applied.
        """
        import contextlib

        retry_after = 60
        if response_headers:
            raw = response_headers.get("Retry-After")
            if raw is not None:
                with contextlib.suppress(ValueError, TypeError):
                    retry_after = float(raw)
        logger.warning("Rate limit 429 hit. Waiting %.0fs before retry.", retry_after)
        await asyncio.sleep(retry_after)

    @property
    def remaining_rpm(self) -> int:
        """Approximate remaining requests available in the current RPM window."""
        now = time.time()
        while self._request_timestamps and self._request_timestamps[0] < now - 60:
            self._request_timestamps.popleft()
        return max(0, self._rpm_limit - len(self._request_timestamps))

    @property
    def remaining_rpd(self) -> int:
        """Approximate remaining requests available today."""
        return max(0, self._rpd_limit - self._daily_count)

    @property
    def stats(self) -> dict:
        return {
            "remaining_rpm": self.remaining_rpm,
            "remaining_rpd": self.remaining_rpd,
            "daily_count": self._daily_count,
            "rpm_limit": self._rpm_limit,
            "rpd_limit": self._rpd_limit,
        }
