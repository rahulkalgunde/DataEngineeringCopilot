"""Integration tests for IngestionProgressTracker against real Redis.

Validates the Redis-backed progress tracker writes correct JSON documents
and handles events accurately under real network conditions.

Run with: pytest tests/integration/test_progress_integration.py -v -m integration
"""

from __future__ import annotations

import json

import pytest

from data_engineering_copilot.domain.models import IngestionEvent
from data_engineering_copilot.workers.progress import (
    _MAX_RECENT_EVENTS,
    IngestionProgressTracker,
)


def _event(
    event_type: str = "page_indexed",
    source_name: str = "test_source",
    *,
    url: str = "https://example.com/page1",
    title: str | None = None,
    chunks_indexed: int = 5,
    pages_fetched: int = 0,
    total_pages_fetched: int = 0,
    error: str | None = None,
    batch_size: int = 0,
) -> IngestionEvent:
    """Build a minimal IngestionEvent for testing."""
    return IngestionEvent(
        event_type=event_type,
        source_name=source_name,
        message=f"{event_type} from {source_name}",
        url=url,
        title=title,
        chunks_indexed=chunks_indexed,
        pages_fetched=pages_fetched,
        error=error,
        total_pages_fetched=total_pages_fetched,
        batch_size=batch_size,
    )


@pytest.fixture
def tracker(fresh_redis_client):
    """Create an IngestionProgressTracker backed by the real Redis testcontainer."""
    return IngestionProgressTracker(
        task_id="test-task-001",
        redis_client=fresh_redis_client,
        source_names=["test_source"],
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestProgressInitialisation:
    def test_initial_state_stored_in_redis(self, fresh_redis_client, tracker):
        raw = fresh_redis_client.get("ingestion:status:test-task-001")
        assert raw is not None
        doc = json.loads(raw)
        assert doc["task_id"] == "test-task-001"
        assert doc["status"] == "PROCESSING"
        assert doc["source_names"] == ["test_source"]
        assert doc["pages_fetched"] == 0
        assert doc["chunks_indexed"] == 0
        assert doc["pages_skipped"] == 0
        assert doc["source_stats"] == {}
        assert doc["recent_events"] == []

    def test_redis_key_has_ttl(self, fresh_redis_client, tracker):
        ttl = fresh_redis_client.ttl("ingestion:status:test-task-001")
        assert ttl > 0


# ---------------------------------------------------------------------------
# Event processing
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestProgressEvents:
    def test_page_indexed_increments_chunks(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_indexed", chunks_indexed=5))
        tracker.on_event(_event("page_indexed", url="https://example.com/page2", chunks_indexed=3))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["chunks_indexed"] == 8

    def test_total_pages_fetched_overwrites(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_indexed", pages_fetched=3))
        tracker.on_event(_event("page_indexed", total_pages_fetched=7))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["pages_fetched"] == 7

    def test_error_event_sets_status_failed(self, fresh_redis_client, tracker):
        tracker.on_event(_event("error", error="Connection timeout"))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["status"] == "FAILED"
        assert doc["error"] == "Connection timeout"

    def test_completion_event_sets_status_completed(self, fresh_redis_client, tracker):
        tracker.on_event(_event("ingestion_complete", chunks_indexed=42))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["status"] == "COMPLETED"
        assert doc["chunks_indexed"] == 42

    def test_pages_skipped_counter(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_skipped"))
        tracker.on_event(_event("page_skipped_duplicate"))
        tracker.on_event(_event("page_skipped_cached"))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["pages_skipped"] == 3

    def test_current_url_updates(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_indexed", url="https://example.com/page1"))
        tracker.on_event(_event("page_indexed", url="https://example.com/page2"))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["current_url"] == "https://example.com/page2"


# ---------------------------------------------------------------------------
# Recent events rolling buffer
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRecentEvents:
    def test_recent_events_capped_at_max(self, fresh_redis_client, tracker):
        for i in range(_MAX_RECENT_EVENTS + 5):
            tracker.on_event(
                _event("page_indexed", url=f"https://example.com/page{i}", chunks_indexed=1)
            )
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert len(doc["recent_events"]) == _MAX_RECENT_EVENTS

    def test_recent_events_order_is_fifo(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_indexed", url="https://example.com/first", chunks_indexed=1))
        tracker.on_event(_event("page_indexed", url="https://example.com/second", chunks_indexed=1))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        urls = [e["url"] for e in doc["recent_events"]]
        assert urls == ["https://example.com/first", "https://example.com/second"]


# ---------------------------------------------------------------------------
# Source stats
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSourceStats:
    def test_per_source_chunks_accumulated(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_indexed", source_name="test_source", chunks_indexed=4))
        tracker.on_event(_event("page_indexed", source_name="test_source", chunks_indexed=6))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["source_stats"]["test_source"]["chunks_indexed"] == 10

    def test_per_source_error_counted(self, fresh_redis_client, tracker):
        tracker.on_event(_event("error", source_name="test_source", error="timeout"))
        tracker.on_event(_event("error", source_name="test_source", error="500"))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["source_stats"]["test_source"]["errors"] == 2

    def test_per_source_pages_skipped(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_skipped", source_name="test_source"))
        tracker.on_event(_event("page_skipped_duplicate", source_name="test_source"))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["source_stats"]["test_source"]["pages_skipped"] == 2

    def test_source_complete_overwrites_cumulative_chunks(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_indexed", source_name="test_source", chunks_indexed=3))
        tracker.on_event(_event("page_indexed", source_name="test_source", chunks_indexed=2))
        tracker.on_event(
            _event("source_complete", source_name="test_source", chunks_indexed=10)
        )
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["source_stats"]["test_source"]["chunks_indexed"] == 10

    def test_unknown_source_not_tracked(self, fresh_redis_client, tracker):
        tracker.on_event(_event("page_indexed", source_name="unknown_source", chunks_indexed=1))
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert "unknown_source" not in doc["source_stats"]


# ---------------------------------------------------------------------------
# mark_completed / mark_failed convenience methods
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConvenienceMethods:
    def test_mark_completed(self, fresh_redis_client, tracker):
        tracker.mark_completed()
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["status"] == "COMPLETED"

    def test_mark_failed(self, fresh_redis_client, tracker):
        tracker.mark_failed("Disk full")
        doc = json.loads(fresh_redis_client.get("ingestion:status:test-task-001"))
        assert doc["status"] == "FAILED"
        assert doc["error"] == "Disk full"

    def test_get_status_returns_copy(self, tracker):
        status = tracker.get_status()
        status["task_id"] = "mutated"
        assert tracker.get_status()["task_id"] == "test-task-001"
