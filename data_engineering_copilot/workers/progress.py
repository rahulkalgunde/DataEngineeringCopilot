"""Redis-backed progress tracker for background ingestion tasks.

Provides ``IngestionProgressTracker`` which listens to ``IngestionEvent``
callbacks and maintains a JSON progress document in Redis so that frontend
clients can poll ``/api/v1/ingest/status/{task_id}`` without blocking.

Design: ``on_event`` callbacks are pure in-memory mutations — they update the
local progress dict and mark it dirty without touching the network.  A
background ``_flush_loop`` (started by :meth:`IngestionProgressTracker.start`)
coalesces dirty snapshots and writes at most one Redis document per
``flush_interval`` seconds, so high-frequency event streams (thousands of
events per page) cost a handful of writes instead of one per event.
:meth:`IngestionProgressTracker.aclose` cancels the background tasks and
performs a final write so a terminal state (COMPLETED/FAILED) is never lost.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from typing import Any

import redis
import redis.asyncio as aioredis
import structlog

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import IngestionEvent

log = structlog.get_logger(__name__)

# Redis key TTL: 24 hours.  Prevents stale keys from accumulating when tasks
# complete or are abandoned.
_STATUS_KEY_TTL_SECONDS = 86400

# A short liveness lease distinguishes an active task from a progress record
# left behind after a worker process is killed.
_LEASE_KEY_PREFIX = "ingestion:lease"
_LEASE_TTL_SECONDS = 300

# How often the tracker refreshes the liveness lease.
_HEARTBEAT_INTERVAL_SECONDS = 30

# Maximum number of recent events to retain in the rolling buffer.
_MAX_RECENT_EVENTS = 15

# Event types that count as "page skipped" for the top-level counter.
_SKIP_EVENT_TYPES = {"page_skipped", "page_skipped_duplicate", "page_skipped_cached"}

# Shared connection pool to avoid opening a new TCP connection on every call.
_connection_pool: redis.ConnectionPool | None = None


def get_redis_client() -> redis.Redis:
    """Return a synchronous Redis client connected via a shared connection pool.

    Used by CLI commands and the rate limiter, which run outside an event
    loop.  The progress tracker itself uses an async client (see
    ``IngestionProgressTracker``).
    """
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=False,
        )
    return redis.Redis(connection_pool=_connection_pool)


class IngestionProgressTracker:
    """Maintains a JSON progress document in Redis for a single ingestion task.

    The tracker is stateful: it keeps an in-memory copy of the progress dict
    and writes it to Redis on a coalescing schedule.  Callers inject an async
    Redis client (reuse ``factory.get_shared_redis_client``) via the
    constructor; no Redis I/O happens until :meth:`start` is awaited.
    """

    REDIS_KEY_PREFIX = "ingestion:status"

    def __init__(
        self,
        task_id: str,
        redis_client: aioredis.Redis,
        source_names: list[str] | None = None,
        flush_interval: float = 1.0,
    ) -> None:
        self._task_id = task_id
        self._redis = redis_client
        self._redis_key = f"{self.REDIS_KEY_PREFIX}:{task_id}"
        self._flush_interval = flush_interval

        resolved_names = source_names if source_names else [s.name for s in settings.sources]
        self._state: dict[str, Any] = {
            "task_id": task_id,
            "status": "PROCESSING",
            "started_at": time.time(),
            "source_names": resolved_names,
            "pages_fetched": 0,
            "chunks_indexed": 0,
            "current_url": "",
            "error": None,
            "source_stats": {},
            "recent_events": [],
            "pages_skipped": 0,
        }
        self._dirty = asyncio.Event()
        self._stop = asyncio.Event()
        self._flush_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._terminal = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def redis_key(self) -> str:
        return self._redis_key

    @property
    def lease_key(self) -> str:
        return f"{_LEASE_KEY_PREFIX}:{self._task_id}"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Write the initial snapshot and start the flush/heartbeat tasks.

        Must be awaited from the event loop that will also receive ``on_event``
        callbacks (ingestion runs on the same loop).
        """
        self._loop = asyncio.get_running_loop()
        await self._refresh_lease()
        self._dirty.set()
        await self._flush_once()
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def aclose(self) -> None:
        """Stop background tasks and persist the latest snapshot.

        The final write guarantees a terminal state (COMPLETED/FAILED) is never
        lost, even if the flush loop was cancelled mid-flight.
        """
        self._stop.set()
        self._dirty.set()
        for task in (self._flush_task, self._heartbeat_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        await self._flush_once()

    # ------------------------------------------------------------------
    # public API (sync, in-memory only)
    # ------------------------------------------------------------------

    def on_event(self, event: IngestionEvent) -> None:
        """Callback wired into ``IngestionService.ingest(on_event=...)``.

        Mutates the in-memory snapshot and marks it dirty.  Performs no
        network I/O; the flush loop persists the snapshot later.
        """
        # --- pages_fetched: use global total when available, else per-source ---
        if event.total_pages_fetched:
            self._state["pages_fetched"] = event.total_pages_fetched
        elif event.pages_fetched:
            self._state["pages_fetched"] = event.pages_fetched

        # --- chunks_indexed: page_indexed events carry per-page deltas ---
        # Accumulate only page_indexed deltas at top-level (source_complete
        # carries the cumulative per-source total which would double-count).
        # ingestion_complete carries the grand total and overwrites.
        if event.event_type == "page_indexed" and event.chunks_indexed:
            self._state["chunks_indexed"] += event.chunks_indexed
        elif event.event_type == "ingestion_complete" and event.chunks_indexed:
            self._state["chunks_indexed"] = event.chunks_indexed

        if event.url:
            self._state["current_url"] = event.url
        if event.error:
            self._state["error"] = str(event.error)

        if event.event_type == "error":
            self._state["status"] = "FAILED"
        elif event.event_type == "ingestion_complete":
            self._state["status"] = "COMPLETED"

        # --- pages_skipped top-level counter ---
        if event.event_type in _SKIP_EVENT_TYPES:
            self._state["pages_skipped"] = self._state.get("pages_skipped", 0) + 1

        # --- per-source stats ---
        source = event.source_name
        known_sources = set(self._state.get("source_names", []))
        if source and source in known_sources:
            stats = self._state["source_stats"]
            if source not in stats:
                stats[source] = {
                    "pages_fetched": 0,
                    "chunks_indexed": 0,
                    "pages_skipped": 0,
                    "errors": 0,
                    "current_url": "",
                }
            src = stats[source]
            # pages_fetched is cumulative — overwrite, don't accumulate
            if event.pages_fetched:
                src["pages_fetched"] = event.pages_fetched
            # chunks_indexed: accumulate per-page deltas; use cumulative
            # from source_complete as the authoritative final count
            if event.event_type == "page_indexed" and event.chunks_indexed:
                src["chunks_indexed"] += event.chunks_indexed
            elif event.event_type == "source_complete" and event.chunks_indexed:
                src["chunks_indexed"] = event.chunks_indexed
            if event.url:
                src["current_url"] = event.url
            if event.error:
                src["errors"] = src.get("errors", 0) + 1
            if event.event_type in _SKIP_EVENT_TYPES:
                src["pages_skipped"] = src.get("pages_skipped", 0) + 1

        # --- recent events (rolling max 15) ---
        recent: list[dict[str, Any]] = self._state["recent_events"]
        recent.append(
            {
                "type": event.event_type,
                "source": event.source_name,
                "url": event.url or "",
                "title": event.title or "",
                "chunks": event.chunks_indexed,
                "ts": time.time(),
                "error": event.error or "",
                "batch_size": event.batch_size,
            }
        )
        if len(recent) > _MAX_RECENT_EVENTS:
            self._state["recent_events"] = recent[-_MAX_RECENT_EVENTS:]

        self._dirty.set()

    def mark_completed(self) -> None:
        """Mark the task completed.  Persisted by the flush loop / ``aclose``."""
        self._state["status"] = "COMPLETED"
        self._terminal = True
        self._request_flush()

    def mark_failed(self, error: str) -> None:
        """Mark the task failed with *error*.  Persisted by the flush loop."""
        self._state["status"] = "FAILED"
        self._state["error"] = error
        self._terminal = True
        self._request_flush()

    def get_status(self) -> dict[str, Any]:
        return dict(self._state)

    # ------------------------------------------------------------------
    # background tasks
    # ------------------------------------------------------------------

    def _request_flush(self) -> None:
        """Wake the flush loop thread-safely from any thread."""
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._dirty.set)
        else:
            self._dirty.set()

    async def _flush_loop(self) -> None:
        """Coalescing writer: at most one Redis write per ``flush_interval``."""
        while True:
            await self._dirty.wait()
            if self._stop.is_set():
                return
            self._dirty.clear()
            # Drain any further events that arrive while we wait so a burst
            # collapses into a single write.
            try:
                while True:
                    await asyncio.wait_for(self._dirty.wait(), timeout=self._flush_interval)
                    self._dirty.clear()
            except TimeoutError:
                pass
            if self._stop.is_set():
                return
            try:
                await self._flush_once()
            except Exception as exc:
                log.warning("progress_tracker.flush_failed", task_id=self._task_id, error=str(exc))

    async def _heartbeat_loop(self) -> None:
        """Refresh the liveness lease every ``_HEARTBEAT_INTERVAL_SECONDS``."""
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=_HEARTBEAT_INTERVAL_SECONDS)
                return
            except TimeoutError:
                pass
            try:
                await self._refresh_lease()
            except Exception as exc:
                log.warning("progress_tracker.heartbeat_failed", task_id=self._task_id, error=str(exc))

    async def _refresh_lease(self) -> None:
        await self._redis.set(
            self.lease_key,
            json.dumps({"task_id": self._task_id, "heartbeat_at": time.time()}),
            ex=_LEASE_TTL_SECONDS,
        )

    async def _flush_once(self) -> None:
        await self._redis.set(self._redis_key, json.dumps(self._state), ex=_STATUS_KEY_TTL_SECONDS)
        if self._terminal:
            await self._redis.delete(self.lease_key)
        self._dirty.clear()
