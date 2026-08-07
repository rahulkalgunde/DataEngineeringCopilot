"""Tests for the Langfuse Metrics API v2 wrapper (Phase 8, Task 8.2)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from data_engineering_copilot.evaluation import langfuse_metrics as mod


class _FakeResponse:
    def __init__(self, data):
        self.data = data


def _fake_client(rows):
    client = MagicMock()
    client._client.api.metrics.metrics.return_value = _FakeResponse(rows)
    return client


def test_fetch_metrics_returns_rows():
    rows = [{"providedModelName": "groq", "sum_totalCost": 1.5}]
    with patch.object(mod, "get_langfuse_instance", return_value=_fake_client(rows)):
        assert mod.fetch_metrics({"view": "observations"}) == rows


def test_fetch_metrics_empty_on_no_client():
    with patch.object(mod, "get_langfuse_instance", return_value=None):
        assert mod.fetch_metrics({"view": "observations"}) == []


def test_fetch_metrics_swallows_errors():
    client = MagicMock()
    client._client.api.metrics.metrics.side_effect = RuntimeError("boom")
    with patch.object(mod, "get_langfuse_instance", return_value=client):
        assert mod.fetch_metrics({"view": "observations"}) == []


def test_base_query_has_required_fields():
    q = mod._base_query(
        "observations",
        [{"measure": "totalCost", "aggregation": "sum"}],
        days=7,
    )
    assert q["view"] == "observations"
    assert q["metrics"] == [{"measure": "totalCost", "aggregation": "sum"}]
    assert "fromTimestamp" in q
    assert "toTimestamp" in q
    assert q["config"] == {"row_limit": 100}
    # fromTimestamp parseable ISO
    from datetime import datetime

    datetime.fromisoformat(q["fromTimestamp"])


def test_cost_by_model_query_shape():
    rows = [{"providedModelName": "groq", "sum_totalCost": 0.5}]
    client = _fake_client(rows)
    with patch.object(mod, "get_langfuse_instance", return_value=client):
        result = mod.cost_by_model(days=3)
    assert result == rows
    _, kwargs = client._client.api.metrics.metrics.call_args
    query = json.loads(kwargs["query"])
    assert query["view"] == "observations"
    assert query["dimensions"] == [{"field": "providedModelName"}]
    assert query["orderBy"] == [{"field": "sum_totalCost", "direction": "desc"}]


def test_daily_volume_latency_query_shape():
    client = _fake_client([{"time_dimension": "2026-08-01", "count_count": 5, "p95_latency": 100}])
    with patch.object(mod, "get_langfuse_instance", return_value=client):
        result = mod.daily_volume_and_latency(days=1)
    assert len(result) == 1
    _, kwargs = client._client.api.metrics.metrics.call_args
    query = json.loads(kwargs["query"])
    assert query["timeDimension"] == {"granularity": "day"}
    assert query["metrics"] == [
        {"measure": "count", "aggregation": "count"},
        {"measure": "latency", "aggregation": "p95"},
    ]


def test_score_summary_filters_by_name():
    client = _fake_client([{"name": "confidence", "avg_value": 0.7, "count_count": 10}])
    with patch.object(mod, "get_langfuse_instance", return_value=client):
        result = mod.score_summary(name="confidence")
    assert len(result) == 1
    _, kwargs = client._client.api.metrics.metrics.call_args
    query = json.loads(kwargs["query"])
    assert query["view"] == "scores-numeric"
    assert query["filters"] == [{"column": "name", "operator": "=", "value": "confidence", "type": "string"}]


def test_query_aliases_has_three_presets():
    assert set(mod.query_aliases()) == {"cost-by-model", "daily-volume-latency", "score-summary"}
