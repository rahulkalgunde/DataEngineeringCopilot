"""Redis-backed progress tracker for background ingestion tasks.

Provides ``IngestionProgressTracker`` which listens to ``IngestionEvent``
callbacks and atomically writes JSON progress snapshots to Redis so that
frontend clients can poll ``/api/v1/ingest/status/{task_id}`` without
blocking.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import IngestionEvent

logger = logging.getLogger(__name__)

# Redis key TTL: 24 hours.  Prevents stale keys from accumulating when tasks
# complete or are abandoned.
_STATUS_KEY_TTL_SECONDS = 86400

# Maximum number of recent events to retain in the rolling buffer.
_MAX_RECENT_EVENTS = 15

# Event types that count as "page skipped" for the top-level counter.
_SKIP_EVENT_TYPES = {"page_skipped", "page_skipped_duplicate", "page_skipped_cached"}

# Shared connection pool to avoid opening a new TCP connection on every call.
_connection_pool: redis.ConnectionPool | None = None


def get_redis_client() -> redis.Redis:
    """Return a Redis client connected via a shared connection pool."""
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
    and writes the entire document to Redis on every ``on_event`` call.  This
    keeps the implementation simple while ensuring that the latest snapshot is
    always available for polling.
    """

    REDIS_KEY_PREFIX = "ingestion:status"

    def __init__(
        self,
        task_id: str,
        redis_client: redis.Redis,
        source_names: list[str] | None = None,
    ) -> None:
        self._task_id = task_id
        self._redis = redis_client
        self._redis_key = f"{self.REDIS_KEY_PREFIX}:{task_id}"

        resolved_names = source_names if source_names else [s.name for s in settings.sources]
        self._state: dict[str, Any] = {
            "task_id": task_id,
            "status": "PROCESSING",
            "source_names": resolved_names,
            "pages_fetched": 0,
            "chunks_indexed": 0,
            "current_url": "",
            "error": None,
            "source_stats": {},
            "recent_events": [],
            "pages_skipped": 0,
        }
        self._sync()

    @property
    def redis_key(self) -> str:
        return self._redis_key

    def on_event(self, event: IngestionEvent) -> None:
        """Callback wired into ``IngestionService.ingest(on_event=...)``."""
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
        recent.append({
            "type": event.event_type,
            "source": event.source_name,
            "url": event.url or "",
            "title": event.title or "",
            "chunks": event.chunks_indexed,
            "ts": time.time(),
            "error": event.error or "",
            "batch_size": event.batch_size,
        })
        if len(recent) > _MAX_RECENT_EVENTS:
            self._state["recent_events"] = recent[-_MAX_RECENT_EVENTS:]

        self._sync()

    def mark_completed(self) -> None:
        self._state["status"] = "COMPLETED"
        self._sync()

    def mark_failed(self, error: str) -> None:
        self._state["status"] = "FAILED"
        self._state["error"] = error
        self._sync()

    def get_status(self) -> dict[str, Any]:
        return dict(self._state)

    def _sync(self) -> None:
        self._redis.set(self._redis_key, json.dumps(self._state), ex=_STATUS_KEY_TTL_SECONDS)
