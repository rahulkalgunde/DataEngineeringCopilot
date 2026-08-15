from __future__ import annotations

import contextlib
import logging
from typing import Self

import redis.asyncio as aioredis
import redis.exceptions

log = logging.getLogger(__name__)


class CrawlCache:
    """Ephemeral Redis hash-based cache for HTTP conditional-GET headers.

    Each URL is keyed by its SHA-256 hash. Fields stored: status, etag, last_modified.
    """

    # Content-hash TTL: 30 days per the architecture guidelines (§5.1).
    _TTL_SECONDS = 2592000

    def __init__(self, redis_url: str, prefix: str = "crawl:", redis_client: aioredis.Redis | None = None) -> None:
        self.prefix = prefix
        self.redis_url = redis_url
        self._redis: aioredis.Redis | None
        if redis_client is not None:
            self._redis = redis_client
            self._owns_client = False
        else:
            self._redis = aioredis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20,
            )
            self._owns_client = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def ping(self) -> None:
        """Validate Redis connectivity. Raises on failure."""
        if self._redis is None:
            raise ConnectionError("Redis client not initialized for CrawlCache")
        try:
            await self._redis.ping()
        except redis.exceptions.RedisError as exc:
            raise ConnectionError(
                f"Redis connection failed for CrawlCache: {exc}. "
                f"Check REDIS_URL (password required if Redis has requirepass)."
            ) from exc

    async def get_headers(self, url_hash: str) -> dict[str, str] | None:
        """Return cached headers {status, etag, last_modified} or None."""
        if self._redis is None:
            return None
        try:
            key = f"{self.prefix}{url_hash}"
            data = await self._redis.hgetall(key)
            if not data:
                return None
            return {str(k): (v.decode() if isinstance(v, bytes) else v) for k, v in data.items()}
        except redis.exceptions.RedisError as exc:
            log.warning(
                "CrawlCache.get_headers failed",
                extra={"url_hash": url_hash, "error_type": type(exc).__name__, "error": str(exc)},
            )
            return None

    async def set_headers(
        self,
        url_hash: str,
        status: int,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        """Cache response headers for a URL."""
        if self._redis is None:
            return
        try:
            key = f"{self.prefix}{url_hash}"
            mapping: dict[str, str] = {"status": str(status)}
            if etag:
                mapping["etag"] = etag
            if last_modified:
                mapping["last_modified"] = last_modified
            await self._redis.hset(key, mapping=mapping)  # type: ignore[arg-type]  # aioredis stub FieldT invariance on injected client
            await self._redis.expire(key, self._TTL_SECONDS)
        except redis.exceptions.RedisError as exc:
            log.warning(
                "CrawlCache.set_headers failed",
                extra={"url_hash": url_hash, "error_type": type(exc).__name__, "error": str(exc)},
            )

    async def close(self) -> None:
        # Only close the client if this instance owns it. Shared clients are
        # closed once at process shutdown, not per-component.
        if not self._owns_client or self._redis is None:
            return
        with contextlib.suppress(redis.exceptions.RedisError):
            await self._redis.close()
            self._redis = None


class NoOpCrawlCache(CrawlCache):
    """Cache-disabled stand-in that never stores or reads crawl headers.

    Used when ``crawl_cache_enabled=False`` so the crawler still receives a
    valid ``CrawlCache``-shaped object without touching Redis: every operation
    becomes a no-op (or returns ``None``), forcing a full fetch per URL.
    """

    def __init__(self, redis_url: str = "", *, redis_client=None) -> None:
        self.prefix = "crawl:"
        self.redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._owns_client = False

    async def ping(self) -> None:
        return None

    async def get_headers(self, url_hash: str) -> dict[str, str] | None:
        return None

    async def set_headers(
        self,
        url_hash: str,
        status: int,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        return None
