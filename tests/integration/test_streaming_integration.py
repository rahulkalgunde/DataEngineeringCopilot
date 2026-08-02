"""Integration tests for the POST /api/v1/ask/stream SSE endpoint.

Exercises the real HTTP streaming path end-to-end.  Happy-path cases run the
*real* ``AsyncRagService.answer_stream`` body against deterministic doubles
(no LLM/embeddings infra); only explicit error-injection cases swap in a
crashing coroutine so the SSE error contract can be verified.

Run with: pytest tests/integration/test_streaming_integration.py -v -m integration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fake RAG answer object returned by the mocked service (error-injection only)
# ---------------------------------------------------------------------------


@dataclass
class FakeSource:
    source_name: str = "test_source"
    title: str = "Test Page"
    url: str = "https://example.com"
    text: str = "Test content snippet."


@dataclass
class FakeAnswer:
    text: str = "The answer is **42**."
    sources: list = field(default_factory=lambda: [FakeSource()])
    confidence: float = 0.85
    groundedness_score: float = 0.92


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fake_rag_service():
    """Real AsyncRagService wired to deterministic doubles.

    ``answer_stream`` runs its actual body so the SSE endpoint is exercised
    against the real pipeline (retrieval → rerank → context → stream).
    """
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
        config=RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.05,
            max_context_chars=2000,
        ),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
    )
    yield service
    await embedder.close()
    await store.close()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from data_engineering_copilot.api.app import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# SSE format helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(raw_text: str) -> list[dict]:
    """Parse raw SSE text into a list of decoded JSON payloads."""
    events = []
    for line in raw_text.strip().splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: ") :]))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestStreamingEndpoint:
    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_sse_stream_returns_start_answer_done(self, mock_get_service, fake_rag_service, client):
        async def _get_service():
            return fake_rag_service

        mock_get_service.side_effect = _get_service
        resp = client.post("/api/v1/ask/stream", json={"question": "What is Apache Spark?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert resp.headers["cache-control"] == "no-cache"
        events = _parse_sse_events(resp.text)
        types = [e["type"] for e in events]
        assert types[0] == "status"
        assert types[-1] == "done"

    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_answer_event_structure(self, mock_get_service, fake_rag_service, client):
        async def _get_service():
            return fake_rag_service

        mock_get_service.side_effect = _get_service
        resp = client.post("/api/v1/ask/stream", json={"question": "What is Apache Spark?"})
        events = _parse_sse_events(resp.text)
        done = next(e for e in events if e["type"] == "done")
        assert "text" in done
        assert "confidence" in done
        assert isinstance(done["text"], str)
        assert len(done["text"]) > 0

    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_done_marker_always_present(self, mock_get_service, fake_rag_service, client):
        async def _get_service():
            return fake_rag_service

        mock_get_service.side_effect = _get_service
        resp = client.post("/api/v1/ask/stream", json={"question": "test"})
        assert resp.text.rstrip().endswith("data: [DONE]")

    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_streaming_error_emits_error_event(self, mock_get_service, client):
        mock_service = AsyncMock()

        async def _boom(question, source_filter=None):
            raise RuntimeError("LLM crashed")
            yield  # pragma: no cover

        mock_service.answer_stream = _boom
        mock_get_service.return_value = mock_service
        resp = client.post("/api/v1/ask/stream", json={"question": "fail?"})
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert any(e["type"] == "error" for e in events)
        assert all(e["type"] != "answer" for e in events), "No partial answer on failure"

    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_streaming_timeout_emits_timeout_error(self, mock_get_service, client):
        mock_service = AsyncMock()

        async def _timeout(question, source_filter=None):
            raise TimeoutError("timed out")
            yield  # pragma: no cover

        mock_service.answer_stream = _timeout
        mock_get_service.return_value = mock_service
        resp = client.post("/api/v1/ask/stream", json={"question": "timeout?"})
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert any(e["type"] == "error" for e in events)
        assert any("Request timed out" in e.get("message", "") for e in events)

    def test_invalid_request_returns_422(self, client):
        resp = client.post("/api/v1/ask/stream", json={"question": ""})
        assert resp.status_code == 422
