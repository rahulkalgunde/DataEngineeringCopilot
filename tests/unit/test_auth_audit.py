"""Unit tests for authentication audit logging (auth success/failure events)."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_engineering_copilot.api.auth import ApiKeyAuthMiddleware


class TestAuditLogging:
    def test_success_emits_auth_success(self, caplog):
        app = FastAPI()

        @app.get("/protected")
        async def protected_route():
            return {"ok": True}

        app.add_middleware(ApiKeyAuthMiddleware, api_key="secret-key-123", rbac_enabled=False)
        client = TestClient(app)

        with caplog.at_level(logging.INFO):
            resp = client.get("/protected", headers={"X-API-Key": "secret-key-123"})
        assert resp.status_code == 200
        assert any("auth_success" in record.getMessage() for record in caplog.records)

    def test_failure_emits_auth_failed(self, caplog):
        app = FastAPI()

        @app.get("/protected")
        async def protected_route():
            return {"ok": True}

        app.add_middleware(ApiKeyAuthMiddleware, api_key="secret-key-123")
        client = TestClient(app)

        with caplog.at_level(logging.INFO):
            resp = client.get("/protected", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401
        assert any("auth_failed" in record.getMessage() for record in caplog.records)

    def test_audit_does_not_log_full_key(self, caplog):
        app = FastAPI()

        @app.get("/protected")
        async def protected_route():
            return {"ok": True}

        app.add_middleware(ApiKeyAuthMiddleware, api_key="super-secret-full-key")
        client = TestClient(app)

        with caplog.at_level(logging.INFO):
            client.get("/protected", headers={"X-API-Key": "super-secret-full-key"})
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "super-secret-full-key" not in joined
