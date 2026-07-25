"""Integration tests for ApiKeyAuthMiddleware.

Tests API key enforcement, Bearer token auth, exempt paths, and dev mode
(no API_KEY set) using a standalone FastAPI app to control the middleware
configuration per test without import-time side effects.

Run with: pytest tests/integration/test_auth_integration.py -v -m integration
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_engineering_copilot.api.auth import ApiKeyAuthMiddleware


def _make_app(api_key: str | None = "test-secret-key-12345") -> FastAPI:
    """Create a minimal FastAPI app with auth middleware for testing."""
    app = FastAPI()
    app.add_middleware(ApiKeyAuthMiddleware, api_key=api_key)

    @app.get("/protected")
    async def protected():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        return {"status": "ready"}

    @app.get("/openapi.json")
    async def openapi():
        return {"openapi": "3.0.0"}

    @app.get("/docs")
    async def docs():
        return {"docs": True}

    @app.get("/redoc")
    async def redoc():
        return {"redoc": True}

    @app.get("/metrics")
    async def metrics():
        return {"metrics": True}

    return app


@pytest.fixture
def auth_client():
    return TestClient(_make_app(api_key="test-secret-key-12345"))


@pytest.fixture
def no_auth_client():
    return TestClient(_make_app(api_key=None))


# ---------------------------------------------------------------------------
# Protected endpoint — correct key
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestApiKeyAuth:
    def test_x_api_key_header_grants_access(self, auth_client):
        resp = auth_client.get("/protected", headers={"X-API-Key": "test-secret-key-12345"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_bearer_token_grants_access(self, auth_client):
        resp = auth_client.get(
            "/protected",
            headers={"Authorization": "Bearer test-secret-key-12345"},
        )
        assert resp.status_code == 200

    def test_wrong_key_returns_401(self, auth_client):
        resp = auth_client.get("/protected", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401
        assert "Invalid or missing API key" in resp.json()["detail"]

    def test_missing_key_returns_401(self, auth_client):
        resp = auth_client.get("/protected")
        assert resp.status_code == 401

    def test_empty_bearer_returns_401(self, auth_client):
        resp = auth_client.get("/protected", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Exempt paths
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestExemptPaths:
    def test_health_exempt(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_ready_exempt(self, auth_client):
        resp = auth_client.get("/ready")
        assert resp.status_code == 200

    def test_docs_exempt(self, auth_client):
        resp = auth_client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_exempt(self, auth_client):
        resp = auth_client.get("/openapi.json")
        assert resp.status_code == 200

    def test_redoc_exempt(self, auth_client):
        resp = auth_client.get("/redoc")
        assert resp.status_code == 200

    def test_metrics_exempt(self, auth_client):
        resp = auth_client.get("/metrics")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Dev mode — no API_KEY set
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestDevMode:
    def test_no_api_key_allows_all_requests(self, no_auth_client):
        resp = no_auth_client.get("/protected")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_no_api_key_with_wrong_key_still_allows(self, no_auth_client):
        resp = no_auth_client.get("/protected", headers={"X-API-Key": "anything"})
        assert resp.status_code == 200
