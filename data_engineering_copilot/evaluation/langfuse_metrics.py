"""Langfuse Metrics API v2 wrapper (Phase 8, Task 8.2).

Thin wrapper over ``GET /api/public/v2/metrics`` (via the v4 SDK's
``api.metrics.metrics``) for alerting/reporting: daily aggregates of cost,
latency, volume, and score averages.

Query reference: https://langfuse.com/docs/metrics/features/metrics-api
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 7


def _time_window(days: int) -> tuple[str, str]:
    now = datetime.datetime.now(datetime.UTC)
    start = now - datetime.timedelta(days=days)
    return start.isoformat(), now.isoformat()


def fetch_metrics(query: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a v2 Metrics API query and return the raw rows.

    ``query`` must include ``view`` plus the fields documented in the Metrics
    API reference (``metrics``, ``dimensions``, ``filters``, ``fromTimestamp``,
    ``toTimestamp``, ``orderBy``, ``config``). Returns ``[]`` on error.
    """
    client = get_langfuse_instance()
    if client is None:
        logger.warning("Langfuse unavailable; cannot fetch metrics")
        return []
    try:
        response = client._client.api.metrics.metrics(query=json.dumps(query))
        return list(getattr(response, "data", None) or [])
    except Exception as exc:
        logger.warning("Metrics query failed: %s", exc)
        return []


def _base_query(
    view: str,
    measures: list[dict[str, str]],
    *,
    dimensions: list[dict[str, str]] | None = None,
    filters: list[dict[str, Any]] | None = None,
    time_dimension: dict[str, str] | None = None,
    order_by: list[dict[str, str]] | None = None,
    days: int = DEFAULT_DAYS,
    row_limit: int = 100,
) -> dict[str, Any]:
    from_timestamp, to_timestamp = _time_window(days)
    query: dict[str, Any] = {
        "view": view,
        "metrics": measures,
        "dimensions": dimensions or [],
        "filters": filters or [],
        "fromTimestamp": from_timestamp,
        "toTimestamp": to_timestamp,
        "config": {"row_limit": row_limit},
    }
    if time_dimension:
        query["timeDimension"] = time_dimension
    if order_by:
        query["orderBy"] = order_by
    return query


def cost_by_model(days: int = DEFAULT_DAYS, row_limit: int = 10) -> list[dict[str, Any]]:
    """Total cost grouped by model, most expensive first."""
    return fetch_metrics(
        _base_query(
            "observations",
            [{"measure": "totalCost", "aggregation": "sum"}],
            dimensions=[{"field": "providedModelName"}],
            order_by=[{"field": "sum_totalCost", "direction": "desc"}],
            days=days,
            row_limit=row_limit,
        )
    )


def daily_volume_and_latency(days: int = DEFAULT_DAYS, row_limit: int = 100) -> list[dict[str, Any]]:
    """Daily request count + p95 latency."""
    return fetch_metrics(
        _base_query(
            "observations",
            [
                {"measure": "count", "aggregation": "count"},
                {"measure": "latency", "aggregation": "p95"},
            ],
            time_dimension={"granularity": "day"},
            order_by=[{"field": "time_dimension", "direction": "asc"}],
            days=days,
            row_limit=row_limit,
        )
    )


def score_summary(
    name: str | None = None,
    days: int = DEFAULT_DAYS,
    row_limit: int = 20,
) -> list[dict[str, Any]]:
    """Average numeric score + count, grouped by score name (optionally one)."""
    filters: list[dict[str, Any]] = []
    if name:
        filters.append({"column": "name", "operator": "=", "value": name, "type": "string"})
    return fetch_metrics(
        _base_query(
            "scores-numeric",
            [
                {"measure": "value", "aggregation": "avg"},
                {"measure": "count", "aggregation": "count"},
            ],
            dimensions=[{"field": "name"}],
            filters=filters,
            order_by=[{"field": "avg_value", "direction": "desc"}],
            days=days,
            row_limit=row_limit,
        )
    )


def query_aliases() -> dict[str, str]:
    """Well-known query names for the ``dec langfuse-metrics`` command."""
    return {
        "cost-by-model": "Total cost by model",
        "daily-volume-latency": "Daily request count and p95 latency",
        "score-summary": "Average numeric scores by name",
    }
