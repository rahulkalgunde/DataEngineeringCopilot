"""Tests for FastAPI /health and /ready endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from data_engineering_copilot.api.app import app

    return TestClient(app)


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_head_root_returns_200(self, client):
        response = client.head("/")
        assert response.status_code == 200


class TestVersionEndpoint:
    def test_version_reports_identity_fields(self, client):
        with (
            patch("data_engineering_copilot.api.app._deps_fingerprint_ok", True),
            patch("data_engineering_copilot.api.app._image_built_at", return_value="2026-07-31T10:00:00+00:00"),
            patch.dict("os.environ", {"IMAGE_GIT_SHA": "abc123"}, clear=False),
        ):
            response = client.get("/api/v1/version")
            assert response.status_code == 200
            body = response.json()
            assert body["git_sha"] == "abc123"
            assert body["image_built_at"] == "2026-07-31T10:00:00+00:00"
            assert body["deps_fingerprint_ok"] is True
            assert body["python_version"]

    def test_version_defaults_when_not_in_container(self, client):
        with (
            patch("data_engineering_copilot.api.app._deps_fingerprint_ok", None),
            patch("data_engineering_copilot.api.app._image_built_at", return_value=None),
        ):
            response = client.get("/api/v1/version")
            body = response.json()
            assert body["git_sha"] == "unknown"
            assert body["image_built_at"] is None
            assert body["deps_fingerprint_ok"] is None


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_always_succeeds(self, client):
        with patch("data_engineering_copilot.api.app._check_tcp", return_value=False):
            response = client.get("/health")
            assert response.status_code == 200


class TestReadyEndpoint:
    def test_ready_all_healthy(self, client):
        with patch("data_engineering_copilot.api.app._check_tcp", return_value=True):
            response = client.get("/ready")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "healthy"
            assert body["checks"]["qdrant"] is True
            assert body["checks"]["ollama"] is True
            assert body["checks"]["redis"] is True
            assert body["checks"]["deps"] is True

    def test_ready_qdrant_down_returns_503(self, client):
        def side_effect(host, port, timeout=3.0):
            return host != "localhost" or port != 6333

        with patch("data_engineering_copilot.api.app._check_tcp", side_effect=side_effect):
            response = client.get("/ready")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "degraded"
            assert body["checks"]["qdrant"] is False

    def test_ready_all_down_returns_503(self, client):
        with patch("data_engineering_copilot.api.app._check_tcp", return_value=False):
            response = client.get("/ready")
            assert response.status_code == 503
            body = response.json()
            # deps check is True (not in container), so status is degraded not unhealthy
            assert body["status"] == "degraded"
            assert body["checks"]["qdrant"] is False
            assert body["checks"]["ollama"] is False
            assert body["checks"]["redis"] is False
            assert body["checks"]["deps"] is True

    def test_ready_includes_all_services(self, client):
        with patch("data_engineering_copilot.api.app._check_tcp", return_value=True):
            response = client.get("/ready")
            checks = response.json()["checks"]
            assert set(checks.keys()) == {"qdrant", "ollama", "redis", "deps"}
