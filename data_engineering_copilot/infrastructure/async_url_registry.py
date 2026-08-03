"""Async Redis-backed URL registry for non-blocking crawl state persistence.

Stores ``url → html_hash`` per documentation source asynchronously so that
ingestion runs do not block the asyncio event loop.
"""

from __future__ import annotations

import json
import time

import redis.exceptions
import structlog

from data_engineering_copilot.domain.protocols import SyncRedisProtocol

log = structlog.get_logger(__name__)


class AsyncUrlRegistry:
    """Per-source URL state store backed by asyncio Redis hashes."""

    # Crawler-state TTL: 7 days per the architecture guidelines (§5.1).
    _TTL_SECONDS = 604800

    def __init__(self, redis_client: SyncRedisProtocol | None, source_name: str) -> None:
        if redis_client is not None and not hasattr(redis_client, "hset"):
            raise TypeError(f"redis_client must implement SyncRedisProtocol (got {type(redis_client).__name__})")
        self._redis = redis_client
        self._key = f"crawl:url_registry:{source_name}"
        self._source_name = source_name

    async def get_html_hash(self, url: str) -> str | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.hget(self._key, url)
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
            await self._redis.hset(self._key, url, record)
            await self._redis.expire(self._key, self._TTL_SECONDS)
        except redis.exceptions.RedisError as exc:
            log.warning(
                "async_url_registry.set_html_hash failed",
                url=url,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def mget_html_hashes(self, urls: list[str]) -> dict[str, str | None]:
        """Return a dict mapping each URL to its stored hash (or None).

        Uses a Redis pipeline to fetch all URLs in a single round trip.
        """
        if self._redis is None or not urls:
            return {u: None for u in urls}
        try:
            pipe = self._redis.pipeline(transaction=False)
            for url in urls:
                pipe.hget(self._key, url)
            raw_results = await pipe.execute()
            result: dict[str, str | None] = {}
            for url, raw in zip(urls, raw_results, strict=False):
                if raw is None:
                    result[url] = None
                else:
                    raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    try:
                        record = json.loads(raw_str)
                        result[url] = record.get("html_hash")
                    except (json.JSONDecodeError, AttributeError):
                        result[url] = None
            return result
        except redis.exceptions.RedisError as exc:
            log.warning(
                "async_url_registry.mget_html_hashes failed",
                url_count=len(urls),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return {u: None for u in urls}

    async def mset_html_hashes(self, url_hash_pairs: list[tuple[str, str]]) -> None:
        """Batch set HTML hashes for multiple URLs using a Redis pipeline."""
        if self._redis is None or not url_hash_pairs:
            return
        records = []
        for url, html_hash in url_hash_pairs:
            records.append((url, json.dumps({"html_hash": html_hash, "discovered_at": time.time()})))
        try:
            pipe = self._redis.pipeline(transaction=False)
            for url, record in records:
                pipe.hset(self._key, url, record)
            pipe.expire(self._key, self._TTL_SECONDS)
            await pipe.execute()
        except redis.exceptions.RedisError as exc:
            log.warning(
                "async_url_registry.mset_html_hashes failed",
                url_count=len(url_hash_pairs),
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def clear(self) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.delete(self._key)
        except redis.exceptions.RedisError as exc:
            log.warning(
                "async_url_registry.clear failed",
                source=self._source_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        log.info("async_url_registry.cleared", source=self._source_name)
