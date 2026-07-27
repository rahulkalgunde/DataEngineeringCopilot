"""Tests for ASGI rate limiter middleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_engineering_copilot.api.middleware import RateLimitMiddleware


class TestRateLimitMiddleware:
    def test_allows_requests_under_limit(self):
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get("/test")
        async def test_route():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_unconfigured_paths_pass_through(self):
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get("/unconfigured")
        async def test_route():
            return {"ok": True}

        client = TestClient(app)
        for _ in range(20):
            resp = client.get("/unconfigured")
            assert resp.status_code == 200

    def test_blocks_over_limit_on_ask_path(self):
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.post("/api/v1/ask")
        async def ask_route():
            return {"ok": True}

        client = TestClient(app)
        for _ in range(61):
            resp = client.post("/api/v1/ask", json={"question": "test"})
            if resp.status_code == 429:
                break
            assert resp.status_code == 200
        else:
            raise AssertionError("Expected a 429 within 61 requests")
        assert resp.status_code == 429

        # Clean up in-memory state so parallel tests aren't affected
        from data_engineering_copilot.services.rate_limiter import _IN_MEMORY_STORE

        keys_to_remove = [k for k in _IN_MEMORY_STORE if k.startswith("ratelimit:/api/v1/ask:")]
        for k in keys_to_remove:
            del _IN_MEMORY_STORE[k]
