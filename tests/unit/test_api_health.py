"""Tests for FastAPI /health and /ready endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from data_engineering_copilot.api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_always_succeeds(self):
        with patch("data_engineering_copilot.api.app._check_tcp", return_value=False):
            response = client.get("/health")
            assert response.status_code == 200


class TestReadyEndpoint:
    def test_ready_all_healthy(self):
        with patch("data_engineering_copilot.api.app._check_tcp", return_value=True):
            response = client.get("/ready")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "healthy"
            assert body["checks"]["qdrant"] is True
            assert body["checks"]["ollama"] is True
            assert body["checks"]["redis"] is True

    def test_ready_qdrant_down_returns_503(self):
        def side_effect(host, port, timeout=3.0):
            return host != "localhost" or port != 6333

        with patch("data_engineering_copilot.api.app._check_tcp", side_effect=side_effect):
            response = client.get("/ready")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "degraded"
            assert body["checks"]["qdrant"] is False

    def test_ready_all_down_returns_503(self):
        with patch("data_engineering_copilot.api.app._check_tcp", return_value=False):
            response = client.get("/ready")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "unhealthy"
            assert all(v is False for v in body["checks"].values())

    def test_ready_includes_all_three_services(self):
        with patch("data_engineering_copilot.api.app._check_tcp", return_value=True):
            response = client.get("/ready")
            checks = response.json()["checks"]
            assert set(checks.keys()) == {"qdrant", "ollama", "redis"}
