"""Tests for the infrastructure SlidingWindowRateLimiter pre-flight methods."""

from __future__ import annotations

from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter


class TestTryAcquire:
    async def test_allows_within_rpm_limit(self):
        rl = SlidingWindowRateLimiter(rpm_limit=3, rpd_limit=1000)
        assert await rl.try_acquire() is True
        assert await rl.try_acquire() is True
        assert await rl.try_acquire() is True

    async def test_returns_false_when_rpm_window_exhausted(self):
        rl = SlidingWindowRateLimiter(rpm_limit=1, rpd_limit=1000)
        assert await rl.try_acquire() is True
        assert await rl.try_acquire() is False
        assert await rl.try_acquire() is False

    async def test_does_not_record_when_rejected(self):
        rl = SlidingWindowRateLimiter(rpm_limit=1, rpd_limit=1000)
        await rl.try_acquire()
        assert await rl.try_acquire() is False
        assert rl.remaining_rpm == 0

    async def test_returns_false_when_daily_limit_exhausted(self):
        rl = SlidingWindowRateLimiter(rpm_limit=100, rpd_limit=1)
        assert await rl.try_acquire() is True
        assert await rl.try_acquire() is False

    async def test_shares_window_with_blocking_acquire(self):
        rl = SlidingWindowRateLimiter(rpm_limit=2, rpd_limit=1000)
        await rl.acquire()
        assert await rl.try_acquire() is True
        assert await rl.try_acquire() is False


class TestWaitUntilAvailable:
    def test_zero_when_slot_free(self):
        rl = SlidingWindowRateLimiter(rpm_limit=5, rpd_limit=1000)
        assert rl.wait_until_available() == 0.0

    async def test_positive_when_rpm_window_exhausted(self):
        rl = SlidingWindowRateLimiter(rpm_limit=1, rpd_limit=1000)
        await rl.try_acquire()
        assert rl.wait_until_available() > 0.0

    async def test_positive_when_daily_limit_exhausted(self):
        rl = SlidingWindowRateLimiter(rpm_limit=100, rpd_limit=1)
        await rl.try_acquire()
        assert rl.wait_until_available() > 0.0


class TestParseRetryAfter:
    def test_defaults_to_60_when_header_missing(self):
        assert SlidingWindowRateLimiter.parse_retry_after() == 60.0
        assert SlidingWindowRateLimiter.parse_retry_after({}) == 60.0

    def test_parses_float_header(self):
        assert SlidingWindowRateLimiter.parse_retry_after({"Retry-After": "5.5"}) == 5.5

    def test_parses_integer_header(self):
        assert SlidingWindowRateLimiter.parse_retry_after({"Retry-After": "30"}) == 30.0

    def test_invalid_header_falls_back_to_default(self):
        assert SlidingWindowRateLimiter.parse_retry_after({"Retry-After": "not-a-number"}) == 60.0


class TestStats:
    def test_remaining_rpm_decrements(self):
        rl = SlidingWindowRateLimiter(rpm_limit=3, rpd_limit=1000)
        assert rl.remaining_rpm == 3

    def test_remaining_rpd_decrements(self):
        rl = SlidingWindowRateLimiter(rpm_limit=100, rpd_limit=5)
        assert rl.remaining_rpd == 5

    def test_is_quota_near_limit_fresh(self):
        rl = SlidingWindowRateLimiter(rpm_limit=1, rpd_limit=1)
        assert rl.is_quota_near_limit() is False

    async def test_is_quota_near_limit_when_exhausted(self):
        rl = SlidingWindowRateLimiter(rpm_limit=1, rpd_limit=1000)
        assert rl.is_quota_near_limit() is False
        await rl.try_acquire()
        assert rl.is_quota_near_limit() is True

    def test_stats_shape(self):
        rl = SlidingWindowRateLimiter(rpm_limit=20, rpd_limit=1000)
        stats = rl.stats
        assert stats["rpm_limit"] == 20
        assert stats["rpd_limit"] == 1000
        assert "remaining_rpm" in stats
        assert "remaining_rpd" in stats
