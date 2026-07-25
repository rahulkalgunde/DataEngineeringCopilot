"""Async Redis-backed URL registry for non-blocking crawl state persistence.

Stores ``url → html_hash`` per documentation source asynchronously so that
ingestion runs do not block the asyncio event loop.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class AsyncUrlRegistry:
    """Per-source URL state store backed by Redis hashes with async/await support."""

    def __init__(self, redis_client: Any, source_name: str) -> None:
        if redis_client is not None and not hasattr(redis_client, "hget"):
            raise TypeError(
                f"redis_client must implement SyncRedisProtocol (got {type(redis_client).__name__})"
            )
        self._redis = redis_client
        self._key = f"crawl:url_registry:{source_name}"
        self._source_name = source_name

    async def get_html_hash(self, url: str) -> str | None:
        """Return the stored ``html_hash`` for *url* asynchronously, or ``None``."""
        if self._redis is None:
            return None
        try:
            raw = self._redis.hget(self._key, url)
        except (ConnectionError, TimeoutError, OSError) as exc:
            log.warning("async_url_registry.get_html_hash failed", url=url, error=str(exc), exc_info=True)
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            record = json.loads(raw)
            return record.get("html_hash")
        except (json.JSONDecodeError, AttributeError):
            return None

    async def set_html_hash(self, url: str, html_hash: str) -> None:
        """Store or update the *html_hash* for *url* asynchronously."""
        if self._redis is None:
            return
        record = json.dumps(
            {
                "html_hash": html_hash,
                "discovered_at": time.time(),
            }
        )
        try:
            self._redis.hset(self._key, url, record)
        except (ConnectionError, TimeoutError, OSError) as exc:
            log.warning("async_url_registry.set_html_hash failed", url=url, error=str(exc), exc_info=True)

    async def clear(self) -> None:
        """Remove all entries for this source asynchronously."""
        if self._redis is None:
            return
        try:
            self._redis.delete(self._key)
        except (ConnectionError, TimeoutError, OSError) as exc:
            log.warning("async_url_registry.clear failed", source=self._source_name, error=str(exc), exc_info=True)
        log.info("async_url_registry.cleared", source=self._source_name)
