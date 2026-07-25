"""Integration tests for Celery workers — Redis status updates and cancellation.

Uses real Redis for status tracking. Celery dispatch is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from data_engineering_copilot.api.app import app


@pytest.mark.integration
@pytest.mark.celery
class TestIngestionStatusRedis:
    """Redis status writes for ingestion tasks."""

    def test_initial_status_document_structure(self, fresh_redis_client):
        """Verify the structure of a DISPATCHED status document written to Redis."""
        import data_engineering_copilot.api.routes as routes_mod

        original_fn = routes_mod.get_redis_client
        routes_mod.get_redis_client = lambda: fresh_redis_client

        mock_task = MagicMock()
        mock_task.id = "test-structure-001"
        mock_task.state = "PENDING"

        try:
            with patch("data_engineering_copilot.api.routes.async_ingest_task.delay", return_value=mock_task):
                client = TestClient(app)
                resp = client.post("/api/v1/ingest", json={"source_names": ["Test"], "max_pages": 5})
                assert resp.status_code == 200

                raw = fresh_redis_client.get(f"ingestion:status:{mock_task.id}")
                assert raw is not None
                status = json.loads(raw)
                assert status["task_id"] == mock_task.id
                assert status["status"] == "DISPATCHED"
                assert status["source_names"] == ["Test"]
                assert status["pages_fetched"] == 0
                assert status["chunks_indexed"] == 0
                assert status["error"] is None
        finally:
            routes_mod.get_redis_client = original_fn

    def test_multiple_ingest_dispatch_statuses(self, fresh_redis_client):
        """Two sequential dispatches produce independent status documents."""
        import data_engineering_copilot.api.routes as routes_mod

        original_fn = routes_mod.get_redis_client
        routes_mod.get_redis_client = lambda: fresh_redis_client

        task_ids = iter(["task-seq-001", "task-seq-002"])

        def delay(*args, **kwargs):
            mock = MagicMock()
            mock.id = next(task_ids)
            mock.state = "PENDING"
            return mock

        try:
            with patch("data_engineering_copilot.api.routes.async_ingest_task.delay", side_effect=delay):
                client = TestClient(app)
                resp1 = client.post("/api/v1/ingest", json={"source_names": ["Spark"]})
                assert resp1.status_code == 200
                fresh_redis_client.delete("ingestion:dispatch_lock")

                resp2 = client.post("/api/v1/ingest", json={"source_names": ["Airflow"]})
                assert resp2.status_code == 200

            raw1 = fresh_redis_client.get("ingestion:status:task-seq-001")
            raw2 = fresh_redis_client.get("ingestion:status:task-seq-002")
            assert raw1 is not None, "Task 1 status should exist"
            assert raw2 is not None, "Task 2 status should exist"
            assert json.loads(raw1)["task_id"] == "task-seq-001"
            assert json.loads(raw2)["task_id"] == "task-seq-002"
        finally:
            routes_mod.get_redis_client = original_fn


@pytest.mark.integration
@pytest.mark.celery
class TestTaskCancellation:
    """Task cancellation via API — revoke and status update."""

    def test_cancel_updates_redis_status(self, fresh_redis_client):
        """POST /api/v1/ingest/{task_id}/cancel should set status to CANCELLED."""
        import data_engineering_copilot.api.routes as routes_mod

        task_id = "test-cancel-001"
        initial_status = json.dumps({"task_id": task_id, "status": "PROCESSING"})
        fresh_redis_client.set(f"ingestion:status:{task_id}", initial_status, ex=86400)

        original_fn = routes_mod.get_redis_client
        routes_mod.get_redis_client = lambda: fresh_redis_client
        try:
            with patch("data_engineering_copilot.api.routes.celery_app.control.revoke"):
                client = TestClient(app)
                resp = client.post(f"/api/v1/ingest/{task_id}/cancel")
                assert resp.status_code == 200

                raw = fresh_redis_client.get(f"ingestion:status:{task_id}")
                assert raw is not None
                status = json.loads(raw)
                assert status["status"] == "CANCELLED"
        finally:
            routes_mod.get_redis_client = original_fn
