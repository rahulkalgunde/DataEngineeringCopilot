"""Tests for api/app.py."""

from __future__ import annotations

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
        # Verify no exception raised
