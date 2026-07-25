"""Integration tests for FastAPI API endpoints.

Tests /api/v1/ingest, /api/v1/task/{id}, /api/v1/ingest/status/{task_id}
endpoints using the real FastAPI TestClient and a real Redis instance
(testcontainer). Only the Celery worker dispatch is mocked.

Run with: pytest tests/integration/test_api_integration.py -v -m integration
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_test_client(fresh_redis_client):
    """Real Redis client + monkeypatch routes.get_redis_client to use it."""
    import data_engineering_copilot.api.routes as routes_mod

    real_client = fresh_redis_client
    original_fn = routes_mod.get_redis_client
    routes_mod.get_redis_client = lambda: real_client
    yield real_client
    routes_mod.get_redis_client = original_fn


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from data_engineering_copilot.api.app import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/v1/ingest  (real Redis, mocked Celery dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestIngestEndpoint:
    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_returns_task_id(self, mock_delay, redis_test_client, client):
        mock_task = MagicMock()
        mock_task.id = "task-abc-123"
        mock_task.state = "PENDING"
        mock_delay.return_value = mock_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"], "max_pages": 10})
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-abc-123"
        assert body["state"] == "PENDING"
        mock_delay.assert_called_once_with(["Test"], 10)

        raw = redis_test_client.get("ingestion:latest_task_id")
        assert raw is not None
        assert raw.decode() == "task-abc-123"
        raw_status = redis_test_client.get("ingestion:status:task-abc-123")
        assert raw_status is not None
        status = json.loads(raw_status)
        assert status["status"] == "DISPATCHED"
        assert status["source_names"] == ["Test"]

    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_with_no_sources(self, mock_delay, redis_test_client, client):
        mock_task = MagicMock()
        mock_task.id = "task-null-sources"
        mock_task.state = "PENDING"
        mock_delay.return_value = mock_task

        resp = client.post("/api/v1/ingest", json={})
        assert resp.status_code == 200
        mock_delay.assert_called_once_with(None, 0)

    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_with_multiple_sources(self, mock_delay, redis_test_client, client):
        mock_task = MagicMock()
        mock_task.id = "task-multi"
        mock_task.state = "PENDING"
        mock_delay.return_value = mock_task

        resp = client.post(
            "/api/v1/ingest",
            json={"source_names": ["Spark", "Airflow"], "max_pages": 5},
        )
        assert resp.status_code == 200
        mock_delay.assert_called_once_with(["Spark", "Airflow"], 5)

    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_invalid_body(self, mock_delay, redis_test_client, client):
        """Sending a non-dict body should return 422."""
        resp = client.post("/api/v1/ingest", json="invalid")
        assert resp.status_code == 422

    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_returns_409_when_already_running(self, mock_delay, redis_test_client, client):
        """POST /api/v1/ingest should return 409 if another task is PROCESSING."""
        existing_task_id = "task-running-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {
            "task_id": existing_task_id,
            "status": "PROCESSING",
            "source_names": ["Test"],
            "pages_fetched": 5,
            "chunks_indexed": 10,
            "current_url": "https://example.com",
            "error": None,
        }
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 409
        mock_delay.assert_not_called()

    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_returns_409_when_dispatched(self, mock_delay, redis_test_client, client):
        """POST /api/v1/ingest should return 409 if another task is DISPATCHED."""
        existing_task_id = "task-dispatched-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {"task_id": existing_task_id, "status": "DISPATCHED"}
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 409
        mock_delay.assert_not_called()

    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_allowed_when_previous_task_completed(self, mock_delay, redis_test_client, client):
        """POST /api/v1/ingest should succeed if previous task is COMPLETED."""
        existing_task_id = "task-completed-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {"task_id": existing_task_id, "status": "COMPLETED"}
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)

        mock_task = MagicMock()
        mock_task.id = "task-new-001"
        mock_task.state = "PENDING"
        mock_delay.return_value = mock_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        mock_delay.assert_called_once()

    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_allowed_when_latest_task_status_corrupt(self, mock_delay, redis_test_client, client):
        """POST /api/v1/ingest should succeed even if the stored status JSON is corrupt."""
        existing_task_id = "task-corrupt-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        redis_test_client.set(f"ingestion:status:{existing_task_id}", "NOT_JSON", ex=86400)

        mock_task = MagicMock()
        mock_task.id = "task-new-002"
        mock_task.state = "PENDING"
        mock_delay.return_value = mock_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        mock_delay.assert_called_once()

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_allowed_when_stale_zombie(self, mock_delay, mock_ar, redis_test_client, client):
        """Allow dispatch when Redis says PROCESSING but Celery says FAILURE."""
        existing_task_id = "task-zombie-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {"task_id": existing_task_id, "status": "PROCESSING"}
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)

        mock_celery_task = MagicMock()
        mock_celery_task.state = "FAILURE"
        mock_ar.return_value = mock_celery_task

        mock_new_task = MagicMock()
        mock_new_task.id = "task-new-zombie"
        mock_new_task.state = "PENDING"
        mock_delay.return_value = mock_new_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        mock_delay.assert_called_once()

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_allowed_when_stale_revoked(self, mock_delay, mock_ar, redis_test_client, client):
        """Allow dispatch when Redis says DISPATCHED but Celery says REVOKED."""
        existing_task_id = "task-revoked-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {"task_id": existing_task_id, "status": "DISPATCHED"}
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)

        mock_celery_task = MagicMock()
        mock_celery_task.state = "REVOKED"
        mock_ar.return_value = mock_celery_task

        mock_new_task = MagicMock()
        mock_new_task.id = "task-new-revoked"
        mock_new_task.state = "PENDING"
        mock_delay.return_value = mock_new_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        mock_delay.assert_called_once()

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_returns_409_when_stale_but_celery_pending(self, mock_delay, mock_ar, redis_test_client, client):
        """Return 409 when Redis says PROCESSING and Celery says PENDING (task still alive)."""
        existing_task_id = "task-alive-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {"task_id": existing_task_id, "status": "PROCESSING"}
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)

        mock_celery_task = MagicMock()
        mock_celery_task.state = "PENDING"
        mock_ar.return_value = mock_celery_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 409
        mock_delay.assert_not_called()


# ---------------------------------------------------------------------------
# GET /api/v1/task/{task_id}  (mocked Celery AsyncResult only)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestTaskStatusEndpoint:
    @patch("data_engineering_copilot.api.routes.AsyncResult")
    def test_pending_task(self, mock_ar, client):
        mock_task = MagicMock()
        mock_task.state = "PENDING"
        mock_task.ready.return_value = False
        mock_ar.return_value = mock_task

        resp = client.get("/api/v1/task/task-xyz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-xyz"
        assert body["state"] == "PENDING"
        assert body["result"] is None

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    def test_successful_task(self, mock_ar, client):
        mock_task = MagicMock()
        mock_task.state = "SUCCESS"
        mock_task.ready.return_value = True
        mock_task.result = {"chunks": 42}
        mock_ar.return_value = mock_task

        resp = client.get("/api/v1/task/task-done")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "SUCCESS"
        assert body["result"] == {"chunks": 42}

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    def test_failed_task(self, mock_ar, client):
        mock_task = MagicMock()
        mock_task.state = "FAILURE"
        mock_task.ready.return_value = True
        mock_task.result = {"error": "Connection refused"}
        mock_ar.return_value = mock_task

        resp = client.get("/api/v1/task/task-fail")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "FAILURE"
        assert "error" in body["result"]

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    def test_nonexistent_task(self, mock_ar, client):
        mock_task = MagicMock()
        mock_task.state = "PENDING"
        mock_task.ready.return_value = False
        mock_ar.return_value = mock_task

        resp = client.get("/api/v1/task/nonexistent-id")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "nonexistent-id"


# ---------------------------------------------------------------------------
# GET /api/v1/ingest/status/{task_id}  (real Redis)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestIngestionStatusEndpoint:
    def test_returns_status_document(self, redis_test_client, client):
        """GET /api/v1/ingest/status/{id} should return the stored status doc."""
        status_doc = {
            "task_id": "task-001",
            "status": "COMPLETED",
            "source_names": ["Test"],
            "pages_fetched": 10,
            "chunks_indexed": 25,
            "current_url": "https://example.com/page10",
            "error": None,
        }
        redis_test_client.set("ingestion:status:task-001", json.dumps(status_doc), ex=86400)

        resp = client.get("/api/v1/ingest/status/task-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "COMPLETED"
        assert body["pages_fetched"] == 10
        assert body["chunks_indexed"] == 25
        assert body["source_names"] == ["Test"]

    def test_returns_404_when_not_found(self, redis_test_client, client):
        resp = client.get("/api/v1/ingest/status/task-nonexistent")
        assert resp.status_code == 404

    def test_returns_error_when_key_expired(self, redis_test_client, client):
        import time

        redis_test_client.set("ingestion:status:task-expired", "NOT_JSON", ex=1)
        time.sleep(1.1)
        resp = client.get("/api/v1/ingest/status/task-expired")
        assert resp.status_code == 404

    def test_returns_500_when_json_corrupt(self, redis_test_client, client):
        redis_test_client.set("ingestion:status:task-corrupt", "NOT_JSON{{", ex=86400)
        resp = client.get("/api/v1/ingest/status/task-corrupt")
        assert resp.status_code == 500
        assert "corrupted" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /openapi.json and /docs  (no external dependencies)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestAppMetadata:
    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/api/v1/ingest" in schema["paths"]

    def test_docs_endpoint(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
