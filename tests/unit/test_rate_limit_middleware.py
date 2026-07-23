"""Tests for ASGI rate limiter middleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_engineering_copilot.api.middleware import RateLimitMiddleware
from data_engineering_copilot.services.rate_limiter import RateLimiter


class TestRateLimitMiddleware:
    def test_allows_requests_under_limit(self):
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(max_calls=5, period_seconds=1.0))

        @app.get("/test")
        async def test_route():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_blocks_over_limit(self):
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(max_calls=2, period_seconds=60.0))

        @app.get("/test")
        async def test_route():
            return {"ok": True}

        client = TestClient(app)
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 200
        resp = client.get("/test")
        assert resp.status_code == 429
