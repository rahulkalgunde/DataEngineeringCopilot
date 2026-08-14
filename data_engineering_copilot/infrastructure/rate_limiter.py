"""Shared sliding-window rate limiter for LLM API providers.

Both embeddings and LLM generation share the same API key for a given
provider, so they must coordinate to stay under the same RPM / RPD limits.
"""

from __future__ import annotations

import asyncio
import contextlib
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

    async def try_acquire(self) -> bool:
        """Non-blocking slot acquisition.

        Returns ``True`` and records a slot (shared with ``acquire``) when a
        slot is free under both RPM and RPD limits. Returns ``False`` without
        recording when either limit is exhausted. Never sleeps.

        Used by the LLM router as a pre-flight gate so an over-limit provider
        is skipped without making a paid API call.
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

            if self._rpd_limit > 0 and self._daily_count >= self._rpd_limit:
                return False

            if len(self._request_timestamps) >= self._rpm_limit:
                return False

            self._request_timestamps.append(time.time())
            self._daily_count += 1
            return True

    def wait_until_available(self) -> float:
        """Estimated seconds until the next slot frees (RPM window or RPD reset).

        Returns ``0.0`` when a slot is currently available. Best-effort and
        non-mutating; used for observability (``available_in_seconds``).
        """
        now = time.time()
        if self._rpd_limit > 0 and self._daily_count >= self._rpd_limit and now < self._daily_reset:
            return max(0.0, self._daily_reset - now)
        if len(self._request_timestamps) >= self._rpm_limit:
            oldest = self._request_timestamps[0]
            if oldest > now - 60:
                return max(0.0, oldest + 60 - now)
        return 0.0

    @staticmethod
    def parse_retry_after(response_headers: dict | None = None) -> float:
        """Extract the ``Retry-After`` value from response headers.

        Header keys are matched case-insensitively: ``httpx`` lowercases keys
        when callers pass ``dict(response.headers)``. Defaults to 60s when the
        header is absent or not parseable.
        """
        retry_after = 60
        if response_headers:
            raw = response_headers.get("Retry-After")
            if raw is None:
                raw = response_headers.get("retry-after")
            if raw is not None:
                with contextlib.suppress(ValueError, TypeError):
                    retry_after = float(raw)
        return retry_after

    async def handle_429(self, response_headers: dict | None = None) -> None:
        """React to a 429 response by waiting for the appropriate ``Retry-After``.

        If the provider includes a ``Retry-After`` header that value is used;
        otherwise a default 60s backoff is applied.
        """
        retry_after = self.parse_retry_after(response_headers)
        logger.warning("Rate limit 429 hit. Waiting %.0fs before retry.", retry_after)
        await asyncio.sleep(retry_after)

    @property
    def remaining_rpm(self) -> int:
        """Approximate remaining requests available in the current RPM window.

        Note: popleft on deque is thread-safe under CPython GIL. This is a
        sync property called from async code — no true concurrency (single-
        threaded event loop), so no lock needed.
        """
        now = time.time()
        while self._request_timestamps and self._request_timestamps[0] < now - 60:
            self._request_timestamps.popleft()
        return max(0, self._rpm_limit - len(self._request_timestamps))

    @property
    def remaining_rpd(self) -> int:
        """Approximate remaining requests available today."""
        return max(0, self._rpd_limit - self._daily_count)

    def is_quota_near_limit(self, threshold: float = 0.1) -> bool:
        rpd_limit = self._rpd_limit
        if rpd_limit > 0:
            rpd_remaining = self.remaining_rpd
            if rpd_remaining / rpd_limit < threshold:
                return True
        rpm_limit = self._rpm_limit
        if rpm_limit > 0:
            rpm_remaining = self.remaining_rpm
            if rpm_remaining / rpm_limit < threshold:
                return True
        return False

    @property
    def stats(self) -> dict:
        return {
            "remaining_rpm": self.remaining_rpm,
            "remaining_rpd": self.remaining_rpd,
            "daily_count": self._daily_count,
            "rpm_limit": self._rpm_limit,
            "rpd_limit": self._rpd_limit,
        }
