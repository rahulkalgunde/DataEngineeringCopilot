"""Tests for rate limiter and health checks."""

from __future__ import annotations

import time

from data_engineering_copilot.services.health_check import HealthChecker
from data_engineering_copilot.services.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(max_calls=5, period_seconds=1.0)
        assert rl.allow() is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_calls=3, period_seconds=1.0)
        for _ in range(3):
            assert rl.allow() is True
        assert rl.allow() is False

    def test_resets_after_period(self):
        rl = RateLimiter(max_calls=2, period_seconds=0.1)
        assert rl.allow() is True
        assert rl.allow() is True
        assert rl.allow() is False
        time.sleep(0.15)
        assert rl.allow() is True

    def test_token_bucket_refills(self):
        rl = RateLimiter(max_calls=2, period_seconds=0.1, refill_interval=0.05)
        for _ in range(2):
            rl.allow()
        assert rl.allow() is False
        time.sleep(0.12)
        assert rl.allow() is True


class TestHealthChecker:
    def test_healthy_status(self):
        hc = HealthChecker()
        hc.register("qdrant", lambda: True)
        status = hc.check()
        assert status.overall == "healthy"
        assert status.services["qdrant"] is True

    def test_degraded_when_one_fails(self):
        hc = HealthChecker()
        hc.register("qdrant", lambda: True)
        hc.register("redis", lambda: False)
        status = hc.check()
        assert status.overall == "degraded"
        assert status.services["redis"] is False

    def test_unhealthy_when_all_fail(self):
        hc = HealthChecker()
        hc.register("qdrant", lambda: False)
        status = hc.check()
        assert status.overall == "unhealthy"

    def test_exception_counts_as_unhealthy(self):
        hc = HealthChecker()
        hc.register("ollama", lambda: (_ for _ in ()).throw(ConnectionError("down")))
        status = hc.check()
        assert status.services["ollama"] is False
