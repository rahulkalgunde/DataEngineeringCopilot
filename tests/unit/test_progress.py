"""Unit tests for IngestionProgressTracker — source_stats, recent_events, pages_skipped."""

from __future__ import annotations

import json
import time

import pytest

from data_engineering_copilot.domain.models import IngestionEvent


class FakeRedis:
    """In-memory Redis stand-in for unit tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    def get(self, key: str) -> bytes | None:
        val = self.store.get(key)
        return val.encode() if val is not None else None


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def tracker(redis: FakeRedis):
    from data_engineering_copilot.workers.progress import IngestionProgressTracker

    return IngestionProgressTracker(
        task_id="test-task-123",
        redis_client=redis,
        source_names=["Apache Spark", "Apache Airflow"],
    )


def _get_state(redis: FakeRedis) -> dict:
    raw = redis.store.get("ingestion:status:test-task-123", "{}")
    return json.loads(raw)


class TestSourceStats:
    """Tests for per-source progress tracking (P2)."""

    def test_initial_source_stats_empty(self, tracker, redis):
        state = _get_state(redis)
        assert state.get("source_stats") == {}

    def test_page_indexed_increments_source(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="page_indexed",
                source_name="Apache Spark",
                message="Indexed page",
                url="https://spark.apache.org/docs/latest/",
                title="Spark Quick Start",
                chunks_indexed=12,
                pages_fetched=1,
            )
        )
        state = _get_state(redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["pages_fetched"] == 1
        assert stats["chunks_indexed"] == 12

    def test_multiple_events_accumulate(self, tracker, redis):
        for i in range(5):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_indexed",
                    source_name="Apache Spark",
                    message=f"Page {i}",
                    url=f"https://spark.apache.org/docs/latest/page{i}.html",
                    title=f"Page {i}",
                    chunks_indexed=3,
                    pages_fetched=1,
                )
            )
        state = _get_state(redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["pages_fetched"] == 5
        assert stats["chunks_indexed"] == 15

    def test_page_skipped_increments_skipped(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped",
                source_name="Apache Spark",
                message="Skipped page",
                url="https://spark.apache.org/docs/latest/",
            )
        )
        state = _get_state(redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["pages_skipped"] == 1
        assert stats["pages_fetched"] == 0

    def test_page_skipped_duplicate_increments_skipped(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped_duplicate",
                source_name="Apache Airflow",
                message="Duplicate",
                url="https://airflow.apache.org/docs/",
            )
        )
        state = _get_state(redis)
        stats = state["source_stats"]["Apache Airflow"]
        assert stats["pages_skipped"] == 1

    def test_error_increments_source_errors(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="fetch_error",
                source_name="Apache Spark",
                message="Connection refused",
                url="https://spark.apache.org/bad",
                error="Connection refused",
            )
        )
        state = _get_state(redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["errors"] == 1

    def test_current_url_tracked_per_source(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="fetch_success",
                source_name="Apache Spark",
                message="Fetched",
                url="https://spark.apache.org/docs/latest/guide.html",
            )
        )
        state = _get_state(redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["current_url"] == "https://spark.apache.org/docs/latest/guide.html"

    def test_different_sources_tracked_separately(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="page_indexed",
                source_name="Apache Spark",
                message="Indexed",
                url="https://spark.apache.org/docs/latest/",
                chunks_indexed=10,
                pages_fetched=1,
            )
        )
        tracker.on_event(
            IngestionEvent(
                event_type="page_indexed",
                source_name="Apache Airflow",
                message="Indexed",
                url="https://airflow.apache.org/docs/stable/",
                chunks_indexed=5,
                pages_fetched=1,
            )
        )
        state = _get_state(redis)
        assert state["source_stats"]["Apache Spark"]["chunks_indexed"] == 10
        assert state["source_stats"]["Apache Airflow"]["chunks_indexed"] == 5

    def test_unknown_source_does_not_crash(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="page_indexed",
                source_name="Unknown Source",
                message="Indexed",
                url="https://unknown.com/",
                chunks_indexed=1,
                pages_fetched=1,
            )
        )
        state = _get_state(redis)
        assert "Unknown Source" not in state.get("source_stats", {})


class TestRecentEvents:
    """Tests for rolling event feed (P1)."""

    def test_initial_recent_events_empty(self, tracker, redis):
        state = _get_state(redis)
        assert state.get("recent_events") == []

    def test_events_appended(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="page_indexed",
                source_name="Apache Spark",
                message="Indexed page",
                url="https://spark.apache.org/docs/latest/",
                title="Quick Start",
                chunks_indexed=12,
            )
        )
        state = _get_state(redis)
        events = state["recent_events"]
        assert len(events) == 1
        assert events[0]["type"] == "page_indexed"
        assert events[0]["source"] == "Apache Spark"
        assert events[0]["url"] == "https://spark.apache.org/docs/latest/"
        assert events[0]["title"] == "Quick Start"
        assert events[0]["chunks"] == 12

    def test_events_rolling_max_15(self, tracker, redis):
        for i in range(20):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_indexed",
                    source_name="Apache Spark",
                    message=f"Page {i}",
                    url=f"https://spark.apache.org/docs/latest/page{i}.html",
                    chunks_indexed=1,
                )
            )
        state = _get_state(redis)
        events = state["recent_events"]
        assert len(events) == 15
        # Most recent events should be kept
        assert events[-1]["url"].endswith("page19.html")
        assert events[0]["url"].endswith("page5.html")

    def test_event_has_timestamp(self, tracker, redis):
        before = time.time()
        tracker.on_event(
            IngestionEvent(
                event_type="fetch_success",
                source_name="Apache Spark",
                message="Fetched",
                url="https://spark.apache.org/docs/latest/",
            )
        )
        after = time.time()
        state = _get_state(redis)
        events = state["recent_events"]
        assert len(events) == 1
        assert before <= events[0]["ts"] <= after

    def test_error_events_included(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="fetch_error",
                source_name="Apache Spark",
                message="Connection refused",
                url="https://spark.apache.org/bad",
                error="Connection refused",
            )
        )
        state = _get_state(redis)
        events = state["recent_events"]
        assert events[0]["type"] == "fetch_error"
        assert events[0]["error"] == "Connection refused"

    def test_batch_events_included(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="batch_embedding",
                source_name="",
                message="Embedding 256 chunks...",
                batch_size=256,
            )
        )
        state = _get_state(redis)
        events = state["recent_events"]
        assert events[0]["type"] == "batch_embedding"
        assert events[0]["batch_size"] == 256

    def test_source_start_events_included(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="source_start",
                source_name="Apache Spark",
                message="Crawling Apache Spark",
            )
        )
        state = _get_state(redis)
        events = state["recent_events"]
        assert events[0]["type"] == "source_start"
        assert events[0]["source"] == "Apache Spark"


class TestPagesSkipped:
    """Tests for top-level pages_skipped counter (P6)."""

    def test_initial_pages_skipped_zero(self, tracker, redis):
        state = _get_state(redis)
        assert state.get("pages_skipped", 0) == 0

    def test_page_skipped_increments_top_level(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped",
                source_name="Apache Spark",
                message="Skipped",
                url="https://spark.apache.org/",
            )
        )
        state = _get_state(redis)
        assert state["pages_skipped"] == 1

    def test_page_skipped_duplicate_increments_top_level(self, tracker, redis):
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped_duplicate",
                source_name="Apache Spark",
                message="Duplicate",
                url="https://spark.apache.org/",
            )
        )
        state = _get_state(redis)
        assert state["pages_skipped"] == 1

    def test_pages_skipped_accumulates(self, tracker, redis):
        for _ in range(3):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_skipped",
                    source_name="Apache Spark",
                    message="Skipped",
                    url="https://spark.apache.org/",
                )
            )
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped_duplicate",
                source_name="Apache Airflow",
                message="Duplicate",
                url="https://airflow.apache.org/",
            )
        )
        state = _get_state(redis)
        assert state["pages_skipped"] == 4
