"""E2E lifecycle tests: dispatch → status → cancel.

Uses in-process FastAPI (ASGITransport) with real testcontainer Redis.
Verifies the full dispatch → Redis status → polling → cancel flow
without requiring a running backend-api or Celery worker.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.e2e
class TestIngestLifecycle:
    """Journey 2: API ingest → Redis status → polling → cancel."""

    @pytest.fixture(autouse=True)
    def _patch_redis(self, e2e_redis_url, e2e_redis, monkeypatch):
        """Patch get_redis_client and flush Redis before each test."""
        import redis

        sync_client = redis.from_url(e2e_redis_url, decode_responses=False)
        import data_engineering_copilot.api.routes as routes_mod

        sync_client.flushdb()
        monkeypatch.setattr(routes_mod, "get_redis_client", lambda: sync_client)
        yield

    async def test_dispatch_creates_redis_status(self, e2e_api_client, e2e_redis):
        """POST /api/v1/ingest writes DISPATCHED status to Redis."""
        resp = await e2e_api_client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        data = resp.json()
        task_id = data["task_id"]

        raw = await e2e_redis.get(f"ingestion:status:{task_id}")
        assert raw is not None, "Redis should have status document"
        status = json.loads(raw)
        assert status["status"] == "DISPATCHED"
        assert status["task_id"] == task_id

    async def test_dispatch_returns_task_id(self, e2e_api_client):
        """POST /api/v1/ingest returns task_id and state."""
        resp = await e2e_api_client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert isinstance(data["task_id"], str)
        assert len(data["task_id"]) > 0

    async def test_status_poll_returns_dispatch_state(self, e2e_api_client, e2e_redis):
        """GET /api/v1/ingest/status/{id} returns the stored status document."""
        resp = await e2e_api_client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        task_id = resp.json()["task_id"]

        status_resp = await e2e_api_client.get(f"/api/v1/ingest/status/{task_id}")
        assert status_resp.status_code == 200
        status = status_resp.json()
        assert status["task_id"] == task_id
        assert status["status"] in ("DISPATCHED", "PROCESSING", "COMPLETED")
        assert "source_names" in status

    async def test_status_returns_404_for_nonexistent(self, e2e_api_client):
        """GET /api/v1/ingest/status/{id} returns 404 for unknown task."""
        resp = await e2e_api_client.get("/api/v1/ingest/status/nonexistent-task-id")
        assert resp.status_code == 404

    async def test_cancel_existing_task(self, e2e_api_client, e2e_redis):
        """POST /api/v1/ingest/{task_id}/cancel sets status to CANCELLED."""
        resp = await e2e_api_client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        task_id = resp.json()["task_id"]

        cancel_resp = await e2e_api_client.post(f"/api/v1/ingest/{task_id}/cancel")
        assert cancel_resp.status_code == 200

        raw = await e2e_redis.get(f"ingestion:status:{task_id}")
        assert raw is not None
        status = json.loads(raw)
        assert status["status"] == "CANCELLED"

    async def test_metrics_endpoint(self, e2e_api_client):
        """GET /metrics returns 200."""
        resp = await e2e_api_client.get("/metrics")
        assert resp.status_code == 200
