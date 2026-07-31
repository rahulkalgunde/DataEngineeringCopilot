"""Tests for /api/v1/ask endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from data_engineering_copilot.domain.models import Answer, DocumentChunk


@pytest.fixture(scope="module")
def client() -> TestClient:
    from data_engineering_copilot.api.app import app

    return TestClient(app)


def _mock_answer():
    return Answer(
        text="Spark SQL is a module for structured data.",
        sources=(
            DocumentChunk(
                chunk_id="c1", source_name="spark-docs", title="Spark Guide", url="http://x", text="Spark SQL."
            ),
        ),
        confidence=0.85,
    )


class TestAskEndpoint:
    def test_ask_returns_200(self, client):
        mock_service = MagicMock()
        mock_service.answer = AsyncMock(return_value=_mock_answer())

        async def _fake_get_rag_service():
            return mock_service

        with patch(
            "data_engineering_copilot.services.rag_service_singleton.get_rag_service",
            side_effect=_fake_get_rag_service,
        ):
            resp = client.post("/api/v1/ask", json={"question": "What is Spark SQL?"})
            assert resp.status_code == 200
            body = resp.json()
            assert "answer" in body
            assert body["confidence"] > 0

    def test_ask_empty_question_returns_422(self, client):
        resp = client.post("/api/v1/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_missing_question_returns_422(self, client):
        resp = client.post("/api/v1/ask", json={})
        assert resp.status_code == 422
