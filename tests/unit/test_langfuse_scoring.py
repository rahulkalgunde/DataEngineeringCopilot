"""Test Langfuse scoring integration."""

from __future__ import annotations

import pytest

from data_engineering_copilot.observability.telemetry import LangfuseTelemetryTracer


class MockLangfuseClient:
    """Mock Langfuse client for testing."""

    def __init__(self):
        self.scores = []

    def start_observation(self, name: str, **kwargs):
        return MockObservation()

    def score(self, trace_id: str, name: str, value: float, data_type: str = "NUMERIC", **kwargs):
        self.scores.append({"trace_id": trace_id, "name": name, "value": value, "data_type": data_type})

    def flush(self):
        pass


class MockObservation:
    """Mock observation for testing."""

    def update(self, **kwargs):
        return self

    def end(self):
        return self

    def start_observation(self, name: str, **kwargs):
        return MockObservation()


class MockLangfuseCompat:
    """Mock LangfuseCompat for testing."""

    def __init__(self):
        self._client = MockLangfuseClient()
        self.scores = []

    def start_observation(self, name: str, **kwargs):
        return MockObservation()

    def score(self, trace_id: str, name: str, value: float, data_type: str = "NUMERIC", **kwargs):
        self.scores.append({"trace_id": trace_id, "name": name, "value": value, "data_type": data_type})

    def flush(self):
        pass


@pytest.fixture
def mock_langfuse():
    """Create mock Langfuse client."""
    return MockLangfuseCompat()


@pytest.fixture
def telemetry_tracer(mock_langfuse):
    """Create telemetry tracer with mock Langfuse."""
    return LangfuseTelemetryTracer(mock_langfuse)


def test_telemetry_tracer_has_score_method(telemetry_tracer):
    """Test that telemetry tracer has score method."""
    assert hasattr(telemetry_tracer, "score")
    assert callable(telemetry_tracer.score)


def test_telemetry_tracer_score_calls_langfuse(telemetry_tracer, mock_langfuse):
    """Test that telemetry tracer score calls Langfuse score."""
    telemetry_tracer.score(
        trace_id="test-trace-123",
        name="confidence",
        value=0.85,
        data_type="NUMERIC",
    )
    
    assert len(mock_langfuse.scores) == 1
    assert mock_langfuse.scores[0]["trace_id"] == "test-trace-123"
    assert mock_langfuse.scores[0]["name"] == "confidence"
    assert mock_langfuse.scores[0]["value"] == 0.85
    assert mock_langfuse.scores[0]["data_type"] == "NUMERIC"


def test_telemetry_tracer_score_multiple_metrics(telemetry_tracer, mock_langfuse):
    """Test that telemetry tracer can score multiple metrics."""
    telemetry_tracer.score(trace_id="trace-1", name="confidence", value=0.9)
    telemetry_tracer.score(trace_id="trace-1", name="groundedness", value=0.8)
    telemetry_tracer.score(trace_id="trace-1", name="intent_confidence", value=0.7)
    
    assert len(mock_langfuse.scores) == 3
    assert mock_langfuse.scores[0]["name"] == "confidence"
    assert mock_langfuse.scores[1]["name"] == "groundedness"
    assert mock_langfuse.scores[2]["name"] == "intent_confidence"