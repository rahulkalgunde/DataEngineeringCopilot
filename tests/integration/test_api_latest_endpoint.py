"""Tests for GET /api/v1/ingest/latest endpoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from data_engineering_copilot.api.app import app

    return TestClient(app)


def _async_redis_mock():
    """Return a coroutine that yields an async-mock redis client."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.set = AsyncMock()
    mock_client.setex = AsyncMock()
    mock_client.delete = AsyncMock()
    mock_client.exists = AsyncMock(return_value=True)

    async def _factory():
        return mock_client

    return _factory, mock_client


@pytest.mark.integration
@pytest.mark.api
class TestLatestEndpoint:
    def test_returns_task_status(self, client):
        task_id = "test-latest-123"
        status_data = {
            "task_id": task_id,
            "status": "PROCESSING",
            "source_names": ["Spark"],
            "pages_fetched": 5,
            "chunks_indexed": 20,
            "current_url": "",
            "error": None,
        }
        factory, mock_client = _async_redis_mock()
        mock_client.get.side_effect = lambda key: {
            "ingestion:latest_task_id": task_id.encode(),
            f"ingestion:status:{task_id}": json.dumps(status_data).encode(),
        }.get(key)

        with patch("data_engineering_copilot.api.routes._get_async_redis", factory):
            response = client.get("/api/v1/ingest/latest")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "PROCESSING"

    def test_returns_404_when_no_task(self, client):
        factory, mock_client = _async_redis_mock()
        mock_client.get.return_value = None

        with patch("data_engineering_copilot.api.routes._get_async_redis", factory):
            response = client.get("/api/v1/ingest/latest")

        assert response.status_code == 404

    def test_dispatch_writes_latest_task_key(self, client):
        factory, mock_client = _async_redis_mock()
        with (
            patch("data_engineering_copilot.api.routes._get_async_redis", factory),
            patch("data_engineering_copilot.api.routes.async_ingest_task") as mock_task,
        ):
            mock_task.apply_async.return_value = MagicMock(id="new-task-id", state="PENDING")

            response = client.post(
                "/api/v1/ingest",
                json={
                    "source_names": ["Spark"],
                    "max_pages": 10,
                },
            )

        assert response.status_code == 200
        set_calls = mock_client.set.call_args_list
        latest_key_written = any(call.args[0] == "ingestion:latest_task_id" for call in set_calls)
        assert latest_key_written, "dispatch should write ingestion:latest_task_id to Redis"
