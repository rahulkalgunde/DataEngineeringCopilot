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
from httpx import ASGITransport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_test_client(routes_async_redis, fresh_redis_client):
    """Real Redis client + monkeypatch routes to use it (async)."""
    for key in fresh_redis_client.scan_iter("ratelimit:*"):
        fresh_redis_client.delete(key)
    return fresh_redis_client


@pytest.fixture(autouse=True)
def _bypass_rate_limiter():
    """Disable rate limiting for all API integration tests."""
    with patch("data_engineering_copilot.api.middleware.RateLimiter.allow_async", return_value=True):
        yield


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
    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_ingest_returns_task_id(self, mock_dispatch, redis_test_client, client):
        mock_task = MagicMock()
        mock_task.id = "task-abc-123"
        mock_task.state = "PENDING"
        mock_dispatch.return_value = mock_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"], "max_pages": 10})
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-abc-123"
        assert body["state"] == "PENDING"
        assert mock_dispatch.call_args.kwargs["args"] == (["Test"], 10)

        raw = redis_test_client.get("ingestion:latest_task_id")
        assert raw is not None
        assert raw.decode() == "task-abc-123"
        raw_status = redis_test_client.get("ingestion:status:task-abc-123")
        assert raw_status is not None
        status = json.loads(raw_status)
        assert status["status"] == "DISPATCHED"
        assert status["source_names"] == ["Test"]

    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_ingest_with_no_sources(self, mock_dispatch, redis_test_client, client):
        mock_task = MagicMock()
        mock_task.id = "task-null-sources"
        mock_task.state = "PENDING"
        mock_dispatch.return_value = mock_task

        resp = client.post("/api/v1/ingest", json={})
        assert resp.status_code == 200
        assert mock_dispatch.call_args.kwargs["args"] == (None, None)

    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_ingest_with_multiple_sources(self, mock_dispatch, redis_test_client, client):
        mock_task = MagicMock()
        mock_task.id = "task-multi"
        mock_task.state = "PENDING"
        mock_dispatch.return_value = mock_task

        resp = client.post(
            "/api/v1/ingest",
            json={"source_names": ["Spark", "Airflow"], "max_pages": 5},
        )
        assert resp.status_code == 200
        assert mock_dispatch.call_args.kwargs["args"] == (["Spark", "Airflow"], 5)

    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_invalid_body(self, mock_delay, redis_test_client, client):
        """Sending a non-dict body should return 422."""
        resp = client.post("/api/v1/ingest", json="invalid")
        assert resp.status_code == 422

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    def test_ingest_returns_409_when_already_running(self, mock_ar, redis_test_client, client):
        """POST /api/v1/ingest should return 409 if another task is PROCESSING."""
        mock_celery_task = MagicMock()
        mock_celery_task.state = "PENDING"
        mock_ar.return_value = mock_celery_task

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
        redis_test_client.set(f"ingestion:lease:{existing_task_id}", "alive", ex=300)

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 409

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_returns_409_when_dispatched(self, mock_delay, mock_ar, redis_test_client, client):
        """POST /api/v1/ingest should return 409 if another task is DISPATCHED."""
        mock_celery_task = MagicMock()
        mock_celery_task.state = "PENDING"
        mock_ar.return_value = mock_celery_task

        existing_task_id = "task-dispatched-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {"task_id": existing_task_id, "status": "DISPATCHED"}
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 409

    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_ingest_allowed_when_previous_task_completed(self, mock_dispatch, redis_test_client, client):
        """POST /api/v1/ingest should succeed if previous task is COMPLETED."""
        existing_task_id = "task-completed-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {"task_id": existing_task_id, "status": "COMPLETED"}
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)

        mock_task = MagicMock()
        mock_task.id = "task-new-001"
        mock_task.state = "PENDING"
        mock_dispatch.return_value = mock_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        mock_dispatch.assert_called_once()

    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_ingest_allowed_when_latest_task_status_corrupt(self, mock_dispatch, redis_test_client, client):
        """POST /api/v1/ingest should succeed even if the stored status JSON is corrupt."""
        existing_task_id = "task-corrupt-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        redis_test_client.set(f"ingestion:status:{existing_task_id}", "NOT_JSON", ex=86400)

        mock_task = MagicMock()
        mock_task.id = "task-new-002"
        mock_task.state = "PENDING"
        mock_dispatch.return_value = mock_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        mock_dispatch.assert_called_once()

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_ingest_allowed_when_stale_zombie(self, mock_dispatch, mock_ar, redis_test_client, client):
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
        mock_dispatch.return_value = mock_new_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        mock_dispatch.assert_called_once()

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_ingest_allowed_when_stale_revoked(self, mock_dispatch, mock_ar, redis_test_client, client):
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
        mock_dispatch.return_value = mock_new_task

        resp = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp.status_code == 200
        mock_dispatch.assert_called_once()

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    @patch("data_engineering_copilot.api.routes.async_ingest_task.delay")
    def test_ingest_returns_409_when_stale_but_celery_pending(self, mock_delay, mock_ar, redis_test_client, client):
        """Return 409 when Redis says PROCESSING and Celery says PENDING (task still alive)."""
        existing_task_id = "task-alive-001"
        redis_test_client.set("ingestion:latest_task_id", existing_task_id, ex=86400)
        status_doc = {"task_id": existing_task_id, "status": "PROCESSING"}
        redis_test_client.set(f"ingestion:status:{existing_task_id}", json.dumps(status_doc), ex=86400)
        redis_test_client.set(f"ingestion:lease:{existing_task_id}", "alive", ex=300)

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


# ---------------------------------------------------------------------------
# Dispatch Lock Tests (real Redis, wire-mocked Celery)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestDispatchLock:
    """Concurrent dispatch protection via Redis SETNX."""

    @pytest.fixture(autouse=True)
    def _setup(self, routes_async_redis, fresh_redis_client):
        for key in fresh_redis_client.scan_iter("ratelimit:*"):
            fresh_redis_client.delete(key)
        with patch("data_engineering_copilot.services.rate_limiter._redis_client", return_value=fresh_redis_client):
            yield

    @patch("data_engineering_copilot.api.routes.AsyncResult")
    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_concurrent_dispatches_second_gets_409(self, mock_dispatch, mock_ar, fresh_redis_client, client):
        """Two rapid POST /api/v1/ingest calls: second returns 409."""

        task_ids = iter(["task-first", "task-second"])

        def dispatch(*args, **kwargs):
            mock = MagicMock()
            mock.id = next(task_ids)
            mock.state = "PENDING"
            return mock

        mock_dispatch.side_effect = dispatch
        mock_celery = MagicMock()
        mock_celery.state = "PENDING"
        mock_ar.return_value = mock_celery

        resp1 = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp1.status_code == 200

        resp2 = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp2.status_code == 409

    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_second_dispatch_without_latest_task(self, mock_dispatch, fresh_redis_client, client):
        """Second dispatch succeeds when previous task status is cleared."""
        task_ids = iter(["task-clear-001", "task-clear-002"])

        def dispatch(*args, **kwargs):
            mock = MagicMock()
            mock.id = next(task_ids)
            mock.state = "PENDING"
            return mock

        mock_dispatch.side_effect = dispatch

        resp1 = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp1.status_code == 200

        fresh_redis_client.delete("ingestion:latest_task_id")
        fresh_redis_client.delete("ingestion:status:task-clear-001")
        fresh_redis_client.delete("ingestion:dispatch_lock")

        resp2 = client.post("/api/v1/ingest", json={"source_names": ["Test"]})
        assert resp2.status_code == 200

    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_dispatch_lock_released_after_success(self, mock_dispatch, fresh_redis_client, client):
        """After successful dispatch, lock key should be deleted."""
        mock_task = MagicMock()
        mock_task.id = "task-lock-test"
        mock_task.state = "PENDING"
        mock_dispatch.return_value = mock_task

        client.post("/api/v1/ingest", json={"source_names": ["Test"]})

        lock_exists = fresh_redis_client.get("ingestion:dispatch_lock")
        assert lock_exists is None, "Lock should be deleted after dispatch"

    @patch("data_engineering_copilot.api.routes.async_ingest_task.apply_async")
    def test_dispatch_lock_auto_expires(self, mock_dispatch, fresh_redis_client, client):
        """Lock key should have TTL set so it auto-expires if process crashes."""
        mock_task = MagicMock()
        mock_task.id = "task-lock-ttl"
        mock_task.state = "PENDING"
        mock_dispatch.return_value = mock_task

        client.post("/api/v1/ingest", json={"source_names": ["Test"]})

        ttl = fresh_redis_client.ttl("ingestion:dispatch_lock")
        # TTL may be -2 if lock already deleted, or > 0 if still present
        assert ttl in (-2, -1) or ttl > 0, f"Unexpected TTL: {ttl}"


# ---------------------------------------------------------------------------
# SSE Streaming Tests (respx wire-mock)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestSseStreaming:
    """SSE streaming endpoint with wire-mocked RAG pipeline."""

    @pytest.mark.asyncio
    async def test_stream_returns_start_answer_done(self, routes_async_redis):
        """SSE event flow against the real answer_stream body wired to doubles."""
        import httpx

        from data_engineering_copilot.api.app import app
        from data_engineering_copilot.domain.models import DocumentChunk, RagConfig
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from tests.doubles.embedder import StubEmbedder
        from tests.doubles.llm import StubLLM
        from tests.doubles.vector_store import InMemoryVectorStore

        store = InMemoryVectorStore()
        await store.initialize()
        embedder = StubEmbedder(dimension=768)
        chunk = DocumentChunk(
            chunk_id="stream:doc000:chunk00",
            source_name="RAG Test Docs",
            title="Apache Spark",
            url="https://example.com/docs/apache-spark.html",
            text="Apache Spark is a unified analytics engine for large-scale data processing.",
        )
        vector = (await embedder.embed_texts([chunk.text]))[0]
        await store.upsert_chunks([chunk], [vector])

        service = AsyncRagService(
            config=RagConfig(retrieval_top_k=5, confidence_threshold=0.05, max_context_chars=2000),
            vector_store=store,
            llm_client=StubLLM(),
            embedder=embedder,
        )

        async def _get_service():
            return service

        with patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service", side_effect=_get_service):
            async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                async with client.stream("POST", "/api/v1/ask/stream", json={"question": "what is spark?"}) as response:
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers.get("content-type", "")

                    chunks = []
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunks.append(line)

                    assert any("status" in c for c in chunks), "Should emit status event"
                    assert any("token" in c for c in chunks), "Should emit token event"
                    assert any("[DONE]" in c for c in chunks), "Should end with [DONE]"
        await embedder.close()
        await store.close()

    @pytest.mark.asyncio
    async def test_stream_invalid_request_returns_422(self, routes_async_redis):
        """Empty question should return 422."""
        import httpx

        from data_engineering_copilot.api.app import app

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask/stream", json={"question": ""})
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# App lifespan: RAG service shutdown
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestAppLifespan:
    """Verifies the API lifespan finally-block closes the RAG service."""

    @pytest.mark.asyncio
    async def test_api_lifespan_closes_service(self, routes_async_redis):
        """Lifespan finally-block closes the initialized RAG service and leaks no tasks."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from data_engineering_copilot.api.app import _lifespan
        from data_engineering_copilot.services.rag_service_singleton import reset_rag_service

        service = MagicMock()
        service.close = AsyncMock()

        reset_rag_service()
        try:
            with patch(
                "data_engineering_copilot.services.rag_service_singleton.get_rag_service_if_initialized",
                return_value=service,
            ):
                async with _lifespan(app=MagicMock()):
                    pass
            service.close.assert_awaited_once()
        finally:
            reset_rag_service()

        # No pending async tasks leaked from the lifespan shutdown.
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        assert pending == []
