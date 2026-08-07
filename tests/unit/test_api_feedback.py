"""Tests for POST /api/v1/feedback (Phase 7, Task 7.5)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from data_engineering_copilot.api.app import app

    return TestClient(app)


def _patch_tracer():
    from unittest.mock import AsyncMock

    tracer = MagicMock()
    tracer.score = MagicMock()
    tracer.flush_async = AsyncMock()
    return tracer


class TestFeedbackEndpoint:
    def test_feedback_scores_trace(self, client):
        tracer = _patch_tracer()
        with patch(
            "data_engineering_copilot.observability.telemetry.build_telemetry_tracer",
            return_value=tracer,
        ):
            resp = client.post(
                "/api/v1/feedback",
                json={"trace_id": "trace-123", "rating": 1, "comment": "great"},
            )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        tracer.score.assert_called_once()
        kwargs = tracer.score.call_args.kwargs
        assert kwargs["trace_id"] == "trace-123"
        assert kwargs["name"] == "user_feedback"
        assert kwargs["value"] == 1.0
        assert kwargs["data_type"] == "NUMERIC"
        assert kwargs["comment"] == "great"
        tracer.flush_async.assert_awaited_once()

    def test_feedback_thumbs_down(self, client):
        tracer = _patch_tracer()
        with patch(
            "data_engineering_copilot.observability.telemetry.build_telemetry_tracer",
            return_value=tracer,
        ):
            resp = client.post(
                "/api/v1/feedback",
                json={"trace_id": "trace-456", "rating": 0},
            )
        assert resp.status_code == 200
        assert tracer.score.call_args.kwargs["value"] == 0.0

    def test_feedback_rating_validation(self, client):
        for bad_rating in (-1, 2, 5):
            resp = client.post(
                "/api/v1/feedback",
                json={"trace_id": "trace-123", "rating": bad_rating},
            )
            assert resp.status_code == 422

    def test_feedback_missing_trace_id_rejected(self, client):
        resp = client.post("/api/v1/feedback", json={"rating": 1})
        assert resp.status_code == 422

    def test_feedback_fail_open_when_tracer_raises(self, client):
        tracer = _patch_tracer()
        tracer.score.side_effect = RuntimeError("boom")
        with patch(
            "data_engineering_copilot.observability.telemetry.build_telemetry_tracer",
            return_value=tracer,
        ):
            resp = client.post(
                "/api/v1/feedback",
                json={"trace_id": "trace-123", "rating": 1},
            )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
