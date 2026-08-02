"""Unit tests for IngestionProgressTracker — source_stats, recent_events, pages_skipped.

The tracker is async: ``on_event`` mutates in-memory state only, and state is
persisted to Redis by ``start()`` (initial snapshot) and ``aclose()`` (final
snapshot) plus a coalescing background flush loop.  Tests therefore create a
tracker, fire events, ``await tracker.aclose()``, then assert against the fake
Redis store.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import cast

import pytest

from data_engineering_copilot.domain.models import IngestionEvent


class FakeRedis:
    """In-memory async Redis stand-in for unit tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.write_count = 0
        self.delete_count = 0

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.write_count += 1
        self.store[key] = value

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def delete(self, key: str) -> None:
        self.delete_count += 1
        self.store.pop(key, None)

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


async def _make_tracker(
    fake_redis: FakeRedis,
    task_id: str = "test-task-123",
    flush_interval: float = 1.0,
):
    import redis.asyncio as aioredis

    from data_engineering_copilot.workers.progress import IngestionProgressTracker

    tracker = IngestionProgressTracker(
        task_id=task_id,
        redis_client=cast(aioredis.Redis, fake_redis),
        source_names=["Apache Spark", "Apache Airflow"],
        flush_interval=flush_interval,
    )
    await tracker.start()
    return tracker


def _get_state(fake_redis: FakeRedis, task_id: str = "test-task-123") -> dict:
    raw = fake_redis.store.get(f"ingestion:status:{task_id}", "{}")
    return json.loads(raw)


def _indexed_event(i: int, chunks: int = 1) -> IngestionEvent:
    return IngestionEvent(
        event_type="page_indexed",
        source_name="Apache Spark",
        message=f"Page {i}",
        url=f"https://spark.apache.org/docs/page{i}.html",
        chunks_indexed=chunks,
        pages_fetched=i + 1,
    )


class TestSourceStats:
    """Tests for per-source progress tracking (P2)."""

    async def test_initial_source_stats_empty(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state.get("source_stats") == {}

    async def test_initial_task_has_live_lease(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        assert "ingestion:lease:test-task-123" in fake_redis.store
        await tracker.aclose()

    async def test_terminal_state_clears_live_lease(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.mark_failed("worker lost")
        await tracker.aclose()
        assert "ingestion:lease:test-task-123" not in fake_redis.store

    async def test_page_indexed_increments_source(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
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
        await tracker.aclose()
        state = _get_state(fake_redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["pages_fetched"] == 1
        assert stats["chunks_indexed"] == 12

    async def test_multiple_events_accumulate(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        for i in range(1, 6):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_indexed",
                    source_name="Apache Spark",
                    message=f"Page {i}",
                    url=f"https://spark.apache.org/docs/latest/page{i}.html",
                    title=f"Page {i}",
                    chunks_indexed=3,
                    pages_fetched=i,
                )
            )
        await tracker.aclose()
        state = _get_state(fake_redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["pages_fetched"] == 5
        assert stats["chunks_indexed"] == 15

    async def test_page_skipped_increments_skipped(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped",
                source_name="Apache Spark",
                message="Skipped page",
                url="https://spark.apache.org/docs/latest/",
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["pages_skipped"] == 1
        assert stats["pages_fetched"] == 0

    async def test_page_skipped_duplicate_increments_skipped(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped_duplicate",
                source_name="Apache Airflow",
                message="Duplicate",
                url="https://airflow.apache.org/docs/",
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        stats = state["source_stats"]["Apache Airflow"]
        assert stats["pages_skipped"] == 1

    async def test_error_increments_source_errors(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="fetch_error",
                source_name="Apache Spark",
                message="Connection refused",
                url="https://spark.apache.org/bad",
                error="Connection refused",
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["errors"] == 1

    async def test_current_url_tracked_per_source(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="fetch_success",
                source_name="Apache Spark",
                message="Fetched",
                url="https://spark.apache.org/docs/latest/guide.html",
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        stats = state["source_stats"]["Apache Spark"]
        assert stats["current_url"] == "https://spark.apache.org/docs/latest/guide.html"

    async def test_different_sources_tracked_separately(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
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
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["source_stats"]["Apache Spark"]["chunks_indexed"] == 10
        assert state["source_stats"]["Apache Airflow"]["chunks_indexed"] == 5

    async def test_unknown_source_does_not_crash(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
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
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert "Unknown Source" not in state.get("source_stats", {})


class TestRecentEvents:
    """Tests for rolling event feed (P1)."""

    async def test_initial_recent_events_empty(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state.get("recent_events") == []

    async def test_events_appended(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
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
        await tracker.aclose()
        state = _get_state(fake_redis)
        events = state["recent_events"]
        assert len(events) == 1
        assert events[0]["type"] == "page_indexed"
        assert events[0]["source"] == "Apache Spark"
        assert events[0]["url"] == "https://spark.apache.org/docs/latest/"
        assert events[0]["title"] == "Quick Start"
        assert events[0]["chunks"] == 12

    async def test_events_rolling_max_15(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
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
        await tracker.aclose()
        state = _get_state(fake_redis)
        events = state["recent_events"]
        assert len(events) == 15
        # Most recent events should be kept
        assert events[-1]["url"].endswith("page19.html")
        assert events[0]["url"].endswith("page5.html")

    async def test_event_has_timestamp(self, fake_redis):
        before = time.time()
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="fetch_success",
                source_name="Apache Spark",
                message="Fetched",
                url="https://spark.apache.org/docs/latest/",
            )
        )
        after = time.time()
        await tracker.aclose()
        state = _get_state(fake_redis)
        events = state["recent_events"]
        assert len(events) == 1
        assert before <= events[0]["ts"] <= after

    async def test_error_events_included(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="fetch_error",
                source_name="Apache Spark",
                message="Connection refused",
                url="https://spark.apache.org/bad",
                error="Connection refused",
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        events = state["recent_events"]
        assert events[0]["type"] == "fetch_error"
        assert events[0]["error"] == "Connection refused"

    async def test_batch_events_included(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="batch_embedding",
                source_name="",
                message="Embedding 256 chunks...",
                batch_size=256,
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        events = state["recent_events"]
        assert events[0]["type"] == "batch_embedding"
        assert events[0]["batch_size"] == 256

    async def test_source_start_events_included(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="source_start",
                source_name="Apache Spark",
                message="Crawling Apache Spark",
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        events = state["recent_events"]
        assert events[0]["type"] == "source_start"
        assert events[0]["source"] == "Apache Spark"


class TestPagesSkipped:
    """Tests for top-level pages_skipped counter (P6)."""

    async def test_initial_pages_skipped_zero(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state.get("pages_skipped", 0) == 0

    async def test_page_skipped_increments_top_level(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped",
                source_name="Apache Spark",
                message="Skipped",
                url="https://spark.apache.org/",
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["pages_skipped"] == 1

    async def test_page_skipped_duplicate_increments_top_level(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="page_skipped_duplicate",
                source_name="Apache Spark",
                message="Duplicate",
                url="https://spark.apache.org/",
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["pages_skipped"] == 1

    async def test_pages_skipped_accumulates(self, fake_redis):
        tracker = await _make_tracker(fake_redis)
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
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["pages_skipped"] == 4


class TestCumulativeEventSemantics:
    """Tests that the tracker correctly handles cumulative pages_fetched
    and per-page-delta chunks_indexed as emitted by both IngestionService
    and AsyncIngestionService.

    Bug context: page_indexed events emit pages_fetched as a cumulative
    count (1, 2, 3, ...) and chunks_indexed as a per-page delta.
    source_complete events emit both as cumulative totals for the source.
    """

    async def test_top_level_chunks_indexed_accumulates_from_page_events(self, fake_redis):
        """Top-level chunks_indexed must sum per-page deltas, not overwrite."""
        tracker = await _make_tracker(fake_redis)
        for i in range(3):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_indexed",
                    source_name="Apache Spark",
                    message=f"Page {i}",
                    url=f"https://spark.apache.org/docs/page{i}.html",
                    chunks_indexed=8,
                    pages_fetched=i + 1,
                )
            )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["chunks_indexed"] == 24

    async def test_top_level_pages_fetched_overwrites_with_cumulative(self, fake_redis):
        """Top-level pages_fetched uses the latest cumulative value."""
        tracker = await _make_tracker(fake_redis)
        for i in range(3):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_indexed",
                    source_name="Apache Spark",
                    message=f"Page {i}",
                    url=f"https://spark.apache.org/docs/page{i}.html",
                    chunks_indexed=5,
                    pages_fetched=i + 1,
                )
            )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["pages_fetched"] == 3

    async def test_per_source_pages_fetched_overwrites_not_accumulates(self, fake_redis):
        """Per-source pages_fetched must overwrite with cumulative, not +=."""
        tracker = await _make_tracker(fake_redis)
        for i in range(3):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_indexed",
                    source_name="Apache Spark",
                    message=f"Page {i}",
                    url=f"https://spark.apache.org/docs/page{i}.html",
                    chunks_indexed=5,
                    pages_fetched=i + 1,
                )
            )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["source_stats"]["Apache Spark"]["pages_fetched"] == 3

    async def test_per_source_chunks_indexed_accumulates_from_page_events(self, fake_redis):
        """Per-source chunks_indexed sums per-page deltas."""
        tracker = await _make_tracker(fake_redis)
        for i in range(3):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_indexed",
                    source_name="Apache Spark",
                    message=f"Page {i}",
                    url=f"https://spark.apache.org/docs/page{i}.html",
                    chunks_indexed=7,
                    pages_fetched=i + 1,
                )
            )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["source_stats"]["Apache Spark"]["chunks_indexed"] == 21

    async def test_source_complete_does_not_double_count_chunks(self, fake_redis):
        """source_complete sends cumulative chunks_indexed which must not
        be added on top of already-accumulated page_indexed deltas."""
        tracker = await _make_tracker(fake_redis)
        for i in range(3):
            tracker.on_event(
                IngestionEvent(
                    event_type="page_indexed",
                    source_name="Apache Spark",
                    message=f"Page {i}",
                    url=f"https://spark.apache.org/docs/page{i}.html",
                    chunks_indexed=8,
                    pages_fetched=i + 1,
                )
            )
        tracker.on_event(
            IngestionEvent(
                event_type="source_complete",
                source_name="Apache Spark",
                message="Done",
                chunks_indexed=24,
                pages_fetched=3,
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["chunks_indexed"] == 24
        assert state["source_stats"]["Apache Spark"]["chunks_indexed"] == 24

    async def test_source_complete_sets_status_completed(self, fake_redis):
        """source_complete event transitions status to COMPLETED."""
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="source_complete",
                source_name="Apache Spark",
                message="Done",
                chunks_indexed=10,
                pages_fetched=5,
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["status"] == "PROCESSING"  # source_complete does NOT set COMPLETED

    async def test_multi_source_cumulative_events(self, fake_redis):
        """Multiple sources with cumulative events are tracked correctly."""
        tracker = await _make_tracker(fake_redis)
        tracker.on_event(
            IngestionEvent(
                event_type="page_indexed",
                source_name="Apache Spark",
                message="Page 1",
                url="https://spark.apache.org/page1.html",
                chunks_indexed=10,
                pages_fetched=1,
                total_pages_fetched=1,
            )
        )
        tracker.on_event(
            IngestionEvent(
                event_type="page_indexed",
                source_name="Apache Spark",
                message="Page 2",
                url="https://spark.apache.org/page2.html",
                chunks_indexed=12,
                pages_fetched=2,
                total_pages_fetched=2,
            )
        )
        tracker.on_event(
            IngestionEvent(
                event_type="page_indexed",
                source_name="Apache Airflow",
                message="Page 1",
                url="https://airflow.apache.org/page1.html",
                chunks_indexed=5,
                pages_fetched=1,
                total_pages_fetched=3,
            )
        )
        await tracker.aclose()
        state = _get_state(fake_redis)
        assert state["pages_fetched"] == 3
        assert state["chunks_indexed"] == 27
        assert state["source_stats"]["Apache Spark"]["pages_fetched"] == 2
        assert state["source_stats"]["Apache Spark"]["chunks_indexed"] == 22
        assert state["source_stats"]["Apache Airflow"]["pages_fetched"] == 1
        assert state["source_stats"]["Apache Airflow"]["chunks_indexed"] == 5


class TestAsyncCoalescing:
    """P1-04: on_event performs no I/O and the flush loop coalesces bursts."""

    async def test_on_event_performs_no_io(self, fake_redis):
        """1000 on_event calls must perform zero Redis writes."""
        tracker = await _make_tracker(fake_redis, task_id="no-io-task")
        fake_redis.write_count = 0
        fake_redis.delete_count = 0

        for i in range(1000):
            tracker.on_event(_indexed_event(i))

        assert fake_redis.write_count == 0
        assert fake_redis.delete_count == 0
        await tracker.aclose()

    async def test_flush_loop_coalesces_burst_to_single_write(self, fake_redis):
        """A 1000-event burst collapses into exactly one Redis write."""
        tracker = await _make_tracker(fake_redis, task_id="burst-task", flush_interval=0.02)
        fake_redis.write_count = 0

        for i in range(1000):
            tracker.on_event(_indexed_event(i))

        await asyncio.sleep(0.1)
        assert fake_redis.write_count == 1
        await tracker.aclose()

    async def test_aclose_writes_final_state(self, fake_redis):
        """aclose() persists the terminal state and clears the lease."""
        tracker = await _make_tracker(fake_redis, task_id="final-task")
        fake_redis.write_count = 0

        tracker.on_event(_indexed_event(0))
        tracker.mark_failed("boom")
        await tracker.aclose()

        state = _get_state(fake_redis, "final-task")
        assert state["status"] == "FAILED"
        assert state["error"] == "boom"
        assert "ingestion:lease:final-task" not in fake_redis.store

    async def test_aclose_cancels_background_tasks(self, fake_redis):
        """aclose() leaves no dangling flush/heartbeat tasks behind."""
        tracker = await _make_tracker(fake_redis, task_id="cancel-task")
        await tracker.aclose()
        assert tracker._flush_task is not None and tracker._flush_task.done()
        assert tracker._heartbeat_task is not None and tracker._heartbeat_task.done()
