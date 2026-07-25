"""Integration tests for the POST /api/v1/ask/stream SSE endpoint.

Exercises the real HTTP streaming path end-to-end while mocking the
underlying RAG service (which requires LLM/embeddings).

Run with: pytest tests/integration/test_streaming_integration.py -v -m integration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fake RAG answer object returned by the mocked service
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
def fake_rag_service():
    service = AsyncMock()
    service.answer.return_value = FakeAnswer()
    return service


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
        mock_get_service.return_value = fake_rag_service
        resp = client.post("/api/v1/ask/stream", json={"question": "What is the answer?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert resp.headers["cache-control"] == "no-cache"
        events = _parse_sse_events(resp.text)
        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert types[-1] == "answer"

    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_answer_event_structure(self, mock_get_service, fake_rag_service, client):
        mock_get_service.return_value = fake_rag_service
        resp = client.post("/api/v1/ask/stream", json={"question": "What is the answer?"})
        events = _parse_sse_events(resp.text)
        answer = next(e for e in events if e["type"] == "answer")
        assert "text" in answer
        assert "confidence" in answer
        assert "groundedness_score" in answer
        assert isinstance(answer["sources"], list)
        assert answer["confidence"] == 0.85
        assert answer["groundedness_score"] == 0.92
        assert answer["text"] == "The answer is **42**."

    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_done_marker_always_present(self, mock_get_service, fake_rag_service, client):
        mock_get_service.return_value = fake_rag_service
        resp = client.post("/api/v1/ask/stream", json={"question": "test"})
        assert resp.text.rstrip().endswith("data: [DONE]")

    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_streaming_error_emits_error_event(self, mock_get_service, client):
        mock_service = AsyncMock()
        mock_service.answer.side_effect = RuntimeError("LLM crashed")
        mock_get_service.return_value = mock_service
        resp = client.post("/api/v1/ask/stream", json={"question": "fail?"})
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert any(e["type"] == "error" for e in events)
        assert any("LLM crashed" in e.get("message", "") for e in events)

    @patch("data_engineering_copilot.services.rag_service_singleton.get_rag_service")
    def test_streaming_timeout_emits_timeout_error(self, mock_get_service, client):
        mock_service = AsyncMock()
        mock_service.answer.side_effect = TimeoutError("timed out")
        mock_get_service.return_value = mock_service
        resp = client.post("/api/v1/ask/stream", json={"question": "timeout?"})
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert any(e["type"] == "error" for e in events)

    def test_invalid_request_returns_422(self, client):
        resp = client.post("/api/v1/ask/stream", json={"question": ""})
        assert resp.status_code == 422
