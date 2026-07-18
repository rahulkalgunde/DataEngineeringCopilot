"""Redis-backed progress tracker for background ingestion tasks.

Provides ``IngestionProgressTracker`` which listens to ``IngestionEvent``
callbacks and atomically writes JSON progress snapshots to Redis so that
frontend clients can poll ``/api/v1/ingest/status/{task_id}`` without
blocking.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import IngestionEvent

logger = logging.getLogger(__name__)


def get_redis_client() -> redis.Redis:
    """Return a Redis client connected to the application Redis instance."""
    return redis.Redis.from_url(settings.redis_url, decode_responses=False)


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

        self._state: dict[str, Any] = {
            "task_id": task_id,
            "status": "PROCESSING",
            "source_names": source_names or [],
            "pages_fetched": 0,
            "chunks_indexed": 0,
            "current_url": "",
            "error": None,
        }
        self._sync()

    @property
    def redis_key(self) -> str:
        return self._redis_key

    def on_event(self, event: IngestionEvent) -> None:
        """Callback wired into ``IngestionService.ingest(on_event=...)``."""
        if event.pages_fetched:
            self._state["pages_fetched"] = event.pages_fetched
        if event.chunks_indexed:
            self._state["chunks_indexed"] = event.chunks_indexed
        if event.url:
            self._state["current_url"] = event.url
        if event.error:
            self._state["error"] = str(event.error)

        if event.event_type == "error":
            self._state["status"] = "FAILED"
        elif event.event_type == "ingestion_complete":
            self._state["status"] = "COMPLETED"

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
        self._redis.set(self._redis_key, json.dumps(self._state))
