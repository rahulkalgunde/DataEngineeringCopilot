"""Tests for api/app.py pure functions."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from data_engineering_copilot.api.app import (
    _check_tcp,
    _check_url,
    _image_built_at,
    app,
    set_trackers,
)

client = TestClient(app)


class TestRoot:
    def test_get_root(self) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_head_root(self) -> None:
        response = client.head("/")
        assert response.status_code == 200


class TestHealth:
    def test_health(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestImageUrl:
    @pytest.mark.asyncio
    async def test_check_url_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _check_url("http://example.com")
            assert result is True

    @pytest.mark.asyncio
    async def test_check_url_failure(self) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("fail"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _check_url("http://example.com")
            assert result is False


class TestCheckTcp:
    @pytest.mark.asyncio
    async def test_check_tcp_success(self) -> None:
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(AsyncMock(), mock_writer)):
            result = await _check_tcp("localhost", 8000, timeout=0.1)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_tcp_failure(self) -> None:
        with patch("asyncio.open_connection", side_effect=OSError("refused")):
            result = await _check_tcp("localhost", 8000, timeout=0.1)
            assert result is False

    @pytest.mark.asyncio
    async def test_check_tcp_timeout(self) -> None:
        with patch("asyncio.open_connection", side_effect=TimeoutError()):
            result = await _check_tcp("localhost", 8000, timeout=0.1)
            assert result is False


class TestImageBuiltAt:
    def test_returns_none_when_file_missing(self) -> None:
        with patch("os.path.getmtime", side_effect=OSError("no such file")):
            assert _image_built_at() is None

    def test_returns_iso_timestamp(self) -> None:
        with patch("os.path.getmtime", return_value=1700000000.0):
            result = _image_built_at()
            assert result is not None
            assert isinstance(result, str)


class TestSetTrackers:
    def test_sets_trackers(self) -> None:
        mock_retrieval = MagicMock()
        mock_token = MagicMock()
        set_trackers(retrieval_tracker=mock_retrieval, token_tracker=mock_token)


class TestVersion:
    def test_version_endpoint(self) -> None:
        mock_detail = MagicMock()
        mock_detail.baked_hash = "abc123"
        mock_detail.live_hash = "def456"
        mock_detail.message = "deps fresh"

        with (
            patch("data_engineering_copilot.infrastructure.dep_check.deps_detail", return_value=mock_detail),
            patch.dict("os.environ", {"IMAGE_GIT_SHA": "sha123"}),
        ):
            response = client.get("/api/v1/version")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "data-engineering-copilot"
        assert data["git_sha"] == "sha123"
        assert data["deps_baked_hash"] == "abc123"
        assert data["deps_live_hash"] == "def456"

    def test_version_without_git_sha(self) -> None:
        mock_detail = MagicMock()
        mock_detail.baked_hash = "abc"
        mock_detail.live_hash = "abc"
        mock_detail.message = ""

        with (
            patch("data_engineering_copilot.infrastructure.dep_check.deps_detail", return_value=mock_detail),
            patch.dict("os.environ", {}, clear=False),
        ):
            response = client.get("/api/v1/version")

        assert response.status_code == 200
        data = response.json()
        assert data["git_sha"] == "unknown"


class TestMetrics:
    def test_metrics_empty(self) -> None:
        set_trackers()
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.text == ""

    def test_metrics_with_retrieval_tracker(self) -> None:
        mock_tracker = MagicMock()
        mock_tracker.get_distribution.return_value = {
            "p50": 0.5,
            "p95": 0.95,
            "p99": 0.99,
            "mean": 0.7,
            "queries": 100,
        }
        set_trackers(retrieval_tracker=mock_tracker)

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "rag_retrieval_score" in response.text
        assert "rag_retrieval_queries_total 100" in response.text

    def test_metrics_with_token_tracker(self) -> None:
        mock_tracker = MagicMock()
        mock_usage = MagicMock()
        mock_usage.total_prompt_tokens = 500
        mock_usage.total_completion_tokens = 200
        mock_usage.total_calls = 50
        mock_tracker.get_usage.return_value = mock_usage
        set_trackers(token_tracker=mock_tracker)

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "rag_token_usage_total" in response.text
        assert "rag_llm_calls_total 50" in response.text

    def test_metrics_with_both_trackers(self) -> None:
        mock_retrieval = MagicMock()
        mock_retrieval.get_distribution.return_value = {
            "p50": 0.5,
            "p95": 0.95,
            "p99": 0.99,
            "mean": 0.7,
            "queries": 10,
        }
        mock_token = MagicMock()
        mock_usage = MagicMock()
        mock_usage.total_prompt_tokens = 100
        mock_usage.total_completion_tokens = 50
        mock_usage.total_calls = 5
        mock_token.get_usage.return_value = mock_usage
        set_trackers(retrieval_tracker=mock_retrieval, token_tracker=mock_token)

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "rag_retrieval_score" in response.text
        assert "rag_token_usage_total" in response.text


class TestReady:
    @pytest.mark.asyncio
    async def test_ready_all_healthy(self) -> None:
        with (
            patch("data_engineering_copilot.api.app._check_tcp", return_value=True),
            patch("data_engineering_copilot.api.app._deps_fingerprint_ok", True),
            patch("data_engineering_copilot.api.app.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.redis_url = "redis://localhost:6379"

            from data_engineering_copilot.api.app import ready

            response = await ready()

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["status"] == "healthy"
        assert body["checks"]["qdrant"] is True
        assert body["checks"]["ollama"] is True
        assert body["checks"]["redis"] is True

    @pytest.mark.asyncio
    async def test_ready_degraded(self) -> None:
        with (
            patch("data_engineering_copilot.api.app._check_tcp", side_effect=[True, False, True]),
            patch("data_engineering_copilot.api.app._deps_fingerprint_ok", True),
            patch("data_engineering_copilot.api.app.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.redis_url = "redis://localhost:6379"

            from data_engineering_copilot.api.app import ready

            response = await ready()

        assert response.status_code == 503
        body = json.loads(response.body)
        assert body["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_ready_all_unhealthy(self) -> None:
        with (
            patch("data_engineering_copilot.api.app._check_tcp", return_value=False),
            patch("data_engineering_copilot.api.app._deps_fingerprint_ok", False),
            patch("data_engineering_copilot.api.app.settings") as mock_settings,
        ):
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.redis_url = "redis://localhost:6379"

            from data_engineering_copilot.api.app import ready

            response = await ready()

        assert response.status_code == 503
        body = json.loads(response.body)
        assert body["status"] == "unhealthy"
