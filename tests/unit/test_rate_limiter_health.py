"""Tests for rate limiter and health checks."""

from __future__ import annotations

import time

from data_engineering_copilot.services.health_check import HealthChecker
from data_engineering_copilot.services.rate_limiter import sliding_window_allow


class TestSlidingWindowRateLimiter:
    def test_allows_within_limit(self):
        assert sliding_window_allow("/test", "ip1", max_calls=5, period_seconds=60) is True

    def test_blocks_over_limit(self):
        for _ in range(3):
            assert sliding_window_allow("/test", "ip2", max_calls=3, period_seconds=60) is True
        assert sliding_window_allow("/test", "ip2", max_calls=3, period_seconds=60) is False

    def test_resets_after_period(self):
        assert sliding_window_allow("/test", "ip3", max_calls=2, period_seconds=1) is True
        assert sliding_window_allow("/test", "ip3", max_calls=2, period_seconds=1) is True
        assert sliding_window_allow("/test", "ip3", max_calls=2, period_seconds=1) is False
        time.sleep(1.1)
        assert sliding_window_allow("/test", "ip3", max_calls=2, period_seconds=1) is True

    def test_different_ips_independent(self):
        for _ in range(3):
            assert sliding_window_allow("/test", "ipA", max_calls=3, period_seconds=60) is True
        assert sliding_window_allow("/test", "ipA", max_calls=3, period_seconds=60) is False
        assert sliding_window_allow("/test", "ipB", max_calls=3, period_seconds=60) is True


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
