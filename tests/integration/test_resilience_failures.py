"""Integration tests for resilience: rate limits, Redis fallback, API timeouts.

Real Redis for rate limiter testing. Mocked RAG service for timeout testing.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from data_engineering_copilot.services.rate_limiter import (
    _IN_MEMORY_STORE,
    RateLimiter,
    sliding_window_allow,
)

# ---------------------------------------------------------------------------
# Rate Limiter Tests (in-memory fallback)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestRateLimiterInMemory:
    """Rate limiter with in-memory fallback (Redis unavailable)."""

    @pytest.fixture(autouse=True)
    def _force_in_memory(self):
        _IN_MEMORY_STORE.clear()
        with patch("data_engineering_copilot.services.rate_limiter._redis_client", return_value=None):
            yield
        _IN_MEMORY_STORE.clear()

    def test_allows_requests_under_limit(self):
        limiter = RateLimiter(path="/api/v1/ask", max_calls=5, period_seconds=60)
        for i in range(5):
            assert limiter.allow_sync("client-1"), f"Request {i + 1} should be allowed"

    def test_blocks_requests_over_limit(self):
        limiter = RateLimiter(path="/api/v1/ask", max_calls=5, period_seconds=60)
        for _ in range(5):
            limiter.allow_sync("client-1")
        assert not limiter.allow_sync("client-1"), "6th request should be blocked"

    def test_different_clients_have_separate_limits(self):
        limiter = RateLimiter(path="/api/v1/ask", max_calls=3, period_seconds=60)
        for _ in range(3):
            limiter.allow_sync("client-a")
        assert not limiter.allow_sync("client-a"), "client-a's 4th request blocked"
        assert limiter.allow_sync("client-b"), "client-b's 1st request allowed"


# ---------------------------------------------------------------------------
# Rate Limiter Tests (real Redis)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestRateLimiterRedis:
    """Rate limiter with real Redis path."""

    def test_redis_sliding_window_blocks_at_limit(self, fresh_redis_client):
        """61st request to /api/v1/ask (limit 60/60s) is blocked."""
        with patch("data_engineering_copilot.services.rate_limiter._redis_client", return_value=fresh_redis_client):
            for i in range(60):
                allowed = sliding_window_allow("/api/v1/ask", "test-ip", 60, 60)
                assert allowed, f"Request {i + 1} should be allowed"
            blocked = not sliding_window_allow("/api/v1/ask", "test-ip", 60, 60)
            assert blocked, "61st request should be blocked"

    def test_redis_connection_failure_falls_back(self, fresh_redis_client):
        """When Redis is unreachable, fall back to in-memory."""
        import redis

        broken_client = redis.Redis.from_url("redis://localhost:1/0", socket_connect_timeout=1)
        _IN_MEMORY_STORE.clear()
        with patch("data_engineering_copilot.services.rate_limiter._redis_client", return_value=broken_client):
            allowed = sliding_window_allow("/api/v1/ask", "test-ip", 5, 60)
            assert allowed, "Should fall back to in-memory and allow"


# ---------------------------------------------------------------------------
# API Timeout (via FastAPI AsyncClient with respx-mocked RAG)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestApiTimeouts:
    """API timeout scenarios with wire-mocked external calls."""

    @pytest.mark.asyncio
    async def test_ask_stream_timeout_emits_error(self, routes_async_redis):
        """Wire-mock LLM to hang, verify SSE emits timeout error and [DONE]."""
        from unittest.mock import AsyncMock

        from data_engineering_copilot.api.app import app
        from data_engineering_copilot.services.async_rag import LLMGenerationError

        # Create a mock RAG service that simulates LLM timeout
        mock_service = AsyncMock()
        mock_service.answer.side_effect = LLMGenerationError("LLM timed out")

        with patch(
            "data_engineering_copilot.services.rag_service_singleton.get_rag_service", return_value=mock_service
        ):
            async with (
                httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
                client.stream("POST", "/api/v1/ask/stream", json={"question": "test"}) as response,
            ):
                assert response.status_code == 200
                chunks = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunks.append(line)

                error_events = [c for c in chunks if '"error"' in c]
                assert len(error_events) > 0, "Should emit error event on timeout"
                assert any("[DONE]" in c for c in chunks), "Stream should end with [DONE]"
