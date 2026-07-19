"""Unit tests for background ingestion progress tracking.

Tests cover:
- IngestionProgressTracker: Redis-backed progress state management
- Worker integration: get_redis_client and callback wiring
- API endpoint: status polling for background ingestion tasks
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.domain.models import IngestionEvent

# ---------------------------------------------------------------------------
# IngestionProgressTracker tests
# ---------------------------------------------------------------------------

class TestIngestionProgressTracker:
    """Tests for the IngestionProgressTracker class that writes
    ingestion progress to Redis as IngestionEvents fire."""

    def _make_tracker(self, mock_redis: MagicMock, task_id: str = "test-task-123"):
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        return IngestionProgressTracker(
            task_id=task_id,
            redis_client=mock_redis,
            source_names=["Apache Spark Documentation"],
        )

    def test_initial_state_written_to_redis(self, mock_redis_client: MagicMock):
        """Tracker publishes an initial PROCESSING state on creation."""
        self._make_tracker(mock_redis_client)

        mock_redis_client.set.assert_called_once()
        call_args = mock_redis_client.set.call_args
        key = call_args[0][0]
        payload = json.loads(call_args[0][1])

        assert key == "ingestion:status:test-task-123"
        assert payload["status"] == "PROCESSING"
        assert payload["pages_fetched"] == 0
        assert payload["chunks_indexed"] == 0
        assert payload["current_url"] == ""
        assert payload["error"] is None

    def test_page_indexed_event_updates_state(self, mock_redis_client: MagicMock):
        """A page_indexed event updates pages_fetched, chunks_indexed, and current_url."""
        tracker = self._make_tracker(mock_redis_client)
        mock_redis_client.reset_mock()

        event = IngestionEvent(
            event_type="page_indexed",
            source_name="Apache Spark Documentation",
            message="Indexed quick-start.html",
            url="https://spark.apache.org/docs/latest/quick-start.html",
            chunks_indexed=12,
            pages_fetched=1,
        )

        tracker.on_event(event)

        mock_redis_client.set.assert_called_once()
        payload = json.loads(mock_redis_client.set.call_args[0][1])

        assert payload["pages_fetched"] == 1
        assert payload["chunks_indexed"] == 12
        assert payload["current_url"] == "https://spark.apache.org/docs/latest/quick-start.html"
        assert payload["status"] == "PROCESSING"

    def test_multiple_events_accumulate_counters(self, mock_redis_client: MagicMock):
        """Sequential events accumulate pages_fetched and chunks_indexed."""
        tracker = self._make_tracker(mock_redis_client)
        mock_redis_client.reset_mock()

        for i in range(1, 4):
            event = IngestionEvent(
                event_type="page_indexed",
                source_name="Apache Spark Documentation",
                message=f"Indexed page {i}",
                url=f"https://spark.apache.org/docs/latest/page{i}.html",
                chunks_indexed=10,
                pages_fetched=i,
            )
            tracker.on_event(event)

        last_payload = json.loads(mock_redis_client.set.call_args[0][1])
        assert last_payload["pages_fetched"] == 3
        assert last_payload["chunks_indexed"] == 10
        assert last_payload["current_url"] == "https://spark.apache.org/docs/latest/page3.html"

    def test_error_event_sets_error_and_status(self, mock_redis_client: MagicMock):
        """An error in the event sets the error field and status to FAILED."""
        tracker = self._make_tracker(mock_redis_client)
        mock_redis_client.reset_mock()

        event = IngestionEvent(
            event_type="error",
            source_name="Apache Spark Documentation",
            message="Connection timeout",
            error="ConnectionTimeout: could not reach server",
        )

        tracker.on_event(event)

        payload = json.loads(mock_redis_client.set.call_args[0][1])
        assert payload["error"] == "ConnectionTimeout: could not reach server"
        assert payload["status"] == "FAILED"

    def test_completion_event_sets_completed_status(self, mock_redis_client: MagicMock):
        """A source_complete or completion event sets status to COMPLETED."""
        tracker = self._make_tracker(mock_redis_client)
        mock_redis_client.reset_mock()

        event = IngestionEvent(
            event_type="ingestion_complete",
            source_name="Apache Spark Documentation",
            message="Ingestion finished",
            pages_fetched=40,
            chunks_indexed=320,
        )

        tracker.on_event(event)

        payload = json.loads(mock_redis_client.set.call_args[0][1])
        assert payload["status"] == "COMPLETED"
        assert payload["pages_fetched"] == 40
        assert payload["chunks_indexed"] == 320

    def test_url_none_does_not_clear_current_url(self, mock_redis_client: MagicMock):
        """When event.url is None, the current_url is not cleared."""
        tracker = self._make_tracker(mock_redis_client)

        # First set a URL
        event_with_url = IngestionEvent(
            event_type="page_indexed",
            source_name="Apache Spark Documentation",
            message="Indexed",
            url="https://spark.apache.org/docs/latest/foo.html",
            chunks_indexed=5,
            pages_fetched=1,
        )
        tracker.on_event(event_with_url)
        mock_redis_client.reset_mock()

        # Then fire event without URL
        event_no_url = IngestionEvent(
            event_type="batch_embedding",
            source_name="",
            message="Embedding 25 chunks...",
        )
        tracker.on_event(event_no_url)

        payload = json.loads(mock_redis_client.set.call_args[0][1])
        assert payload["current_url"] == "https://spark.apache.org/docs/latest/foo.html"

    def test_redis_key_format(self, mock_redis_client: MagicMock):
        """Redis key follows the ingestion:status:{task_id} convention."""
        self._make_tracker(mock_redis_client, task_id="abc-456")

        call_args = mock_redis_client.set.call_args
        assert call_args[0][0] == "ingestion:status:abc-456"

    def test_get_status_returns_current_state(self, mock_redis_client: MagicMock):
        """get_status returns the latest in-memory state dict."""
        tracker = self._make_tracker(mock_redis_client)

        event = IngestionEvent(
            event_type="page_indexed",
            source_name="Apache Spark Documentation",
            message="Indexed",
            url="https://example.com",
            chunks_indexed=5,
            pages_fetched=1,
        )
        tracker.on_event(event)

        state = tracker.get_status()
        assert state["pages_fetched"] == 1
        assert state["chunks_indexed"] == 5
        assert state["task_id"] == "test-task-123"


# ---------------------------------------------------------------------------
# Worker / Redis client tests
# ---------------------------------------------------------------------------

class TestGetRedisClient:
    """Tests for the get_redis_client factory function."""

    @patch("data_engineering_copilot.workers.progress.redis")
    def test_returns_redis_instance(self, mock_redis_module: MagicMock):
        """get_redis_client returns a Redis client from the shared pool."""
        from data_engineering_copilot.workers import progress as progress_mod
        from data_engineering_copilot.workers.progress import get_redis_client

        mock_pool = MagicMock()
        mock_client = MagicMock()
        mock_redis_module.ConnectionPool.from_url.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client

        with patch.object(progress_mod, "_connection_pool", None):
            result = get_redis_client()

        assert result is mock_client

    @patch("data_engineering_copilot.workers.progress.redis")
    def test_uses_settings_url(self, mock_redis_module: MagicMock):
        """Redis client is constructed with the URL from settings."""
        from data_engineering_copilot.workers import progress as progress_mod
        from data_engineering_copilot.workers.progress import get_redis_client

        mock_pool = MagicMock()
        mock_client = MagicMock()
        mock_redis_module.ConnectionPool.from_url.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client

        with patch.object(progress_mod, "_connection_pool", None):
            get_redis_client()

        mock_redis_module.ConnectionPool.from_url.assert_called_once()


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestIngestionStatusEndpoint:
    """Tests for the /api/v1/ingest/status/{task_id} FastAPI endpoint."""

    @patch("data_engineering_copilot.api.routes.get_redis_client")
    def test_returns_200_with_valid_task(self, mock_get_client: MagicMock):
        """Endpoint returns 200 with progress JSON for an existing task."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.routes import router

        app = FastAPI()
        app.include_router(router)

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        expected = {
            "task_id": "task-abc",
            "status": "PROCESSING",
            "pages_fetched": 5,
            "chunks_indexed": 40,
            "current_url": "https://example.com",
            "error": None,
        }
        mock_client.get.return_value = json.dumps(expected).encode()

        client = TestClient(app)
        response = client.get("/api/v1/ingest/status/task-abc")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-abc"
        assert data["status"] == "PROCESSING"
        assert data["pages_fetched"] == 5

    @patch("data_engineering_copilot.api.routes.get_redis_client")
    def test_returns_404_for_missing_task(self, mock_get_client: MagicMock):
        """Endpoint returns 404 when task_id has no Redis record."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.routes import router

        app = FastAPI()
        app.include_router(router)

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = None

        client = TestClient(app)
        response = client.get("/api/v1/ingest/status/nonexistent-task")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("data_engineering_copilot.api.routes.get_redis_client")
    def test_returns_completed_status(self, mock_get_client: MagicMock):
        """Endpoint returns COMPLETED status with final metrics."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.routes import router

        app = FastAPI()
        app.include_router(router)

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        expected = {
            "task_id": "task-xyz",
            "status": "COMPLETED",
            "pages_fetched": 40,
            "chunks_indexed": 320,
            "current_url": "",
            "error": None,
        }
        mock_client.get.return_value = json.dumps(expected).encode()

        client = TestClient(app)
        response = client.get("/api/v1/ingest/status/task-xyz")

        assert response.status_code == 200
        assert response.json()["status"] == "COMPLETED"

    @patch("data_engineering_copilot.api.routes.get_redis_client")
    def test_returns_failed_status_with_error(self, mock_get_client: MagicMock):
        """Endpoint returns FAILED status with error message."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.routes import router

        app = FastAPI()
        app.include_router(router)

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        expected = {
            "task_id": "task-fail",
            "status": "FAILED",
            "pages_fetched": 3,
            "chunks_indexed": 24,
            "current_url": "https://example.com/bad",
            "error": "VectorStoreError: connection refused",
        }
        mock_client.get.return_value = json.dumps(expected).encode()

        client = TestClient(app)
        response = client.get("/api/v1/ingest/status/task-fail")

        assert response.status_code == 200
        assert response.json()["status"] == "FAILED"
        assert "VectorStoreError" in response.json()["error"]

    @patch("data_engineering_copilot.api.routes.get_redis_client")
    @patch("data_engineering_copilot.api.routes.celery_app")
    def test_cancel_endpoint_returns_cancelled_status(
        self, mock_celery: MagicMock, mock_get_client: MagicMock
    ):
        """POST cancel returns 200 and updates Redis status to CANCELLED."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.routes import router

        app = FastAPI()
        app.include_router(router)

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        existing = {
            "task_id": "task-cancel",
            "status": "PROCESSING",
            "pages_fetched": 5,
            "chunks_indexed": 40,
            "current_url": "",
            "error": None,
        }
        mock_client.get.return_value = json.dumps(existing).encode()

        client = TestClient(app)
        response = client.post("/api/v1/ingest/task-cancel/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == "CANCELLED"
        mock_celery.control.revoke.assert_called_once()

    @patch("data_engineering_copilot.api.routes.get_redis_client")
    @patch("data_engineering_copilot.api.routes.celery_app")
    def test_cancel_endpoint_handles_missing_task(
        self, mock_celery: MagicMock, mock_get_client: MagicMock
    ):
        """POST cancel returns 200 even if task not found in Redis."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.routes import router

        app = FastAPI()
        app.include_router(router)

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = None

        client = TestClient(app)
        response = client.post("/api/v1/ingest/nonexistent/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == "CANCELLED"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis_client() -> MagicMock:
    """A MagicMock simulating a Redis client."""
    client = MagicMock()
    client.get.return_value = None
    return client
