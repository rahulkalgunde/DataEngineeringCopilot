"""Async Redis-backed URL registry for non-blocking crawl state persistence.

Stores ``url → html_hash`` per documentation source asynchronously so that
ingestion runs do not block the asyncio event loop.
"""

from __future__ import annotations

import json
import time

import redis.exceptions
import structlog

log = structlog.get_logger(__name__)


class AsyncUrlRegistry:
    """Per-source URL state store backed by asyncio Redis hashes."""

    def __init__(self, redis_client: object | None, source_name: str) -> None:
        self._redis = redis_client
        self._key = f"crawl:url_registry:{source_name}"
        self._source_name = source_name

    async def get_html_hash(self, url: str) -> str | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.hget(self._key, url)  # type: ignore[union-attr]
        except redis.exceptions.RedisError as exc:
            log.warning(
                "async_url_registry.get_html_hash failed",
                url=url,
                error_type=type(exc).__name__,
                error=str(exc),
            )
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
        if self._redis is None:
            return
        record = json.dumps(
            {
                "html_hash": html_hash,
                "discovered_at": time.time(),
            }
        )
        try:
            await self._redis.hset(self._key, url, record)  # type: ignore[union-attr]
        except redis.exceptions.RedisError as exc:
            log.warning(
                "async_url_registry.set_html_hash failed",
                url=url,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def clear(self) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.delete(self._key)  # type: ignore[union-attr]
        except redis.exceptions.RedisError as exc:
            log.warning(
                "async_url_registry.clear failed",
                source=self._source_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        log.info("async_url_registry.cleared", source=self._source_name)
