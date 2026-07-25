from __future__ import annotations

import contextlib
import logging

import redis.asyncio as aioredis
import redis.exceptions

log = logging.getLogger(__name__)


class CrawlCache:
    """Ephemeral Redis hash-based cache for HTTP conditional-GET headers.

    Each URL is keyed by its SHA-256 hash. Fields stored: status, etag, last_modified.
    """

    def __init__(self, redis_url: str, prefix: str = "crawl:") -> None:
        self.prefix = prefix
        self.redis_url = redis_url
        self._redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            max_connections=20,
        )

    async def ping(self) -> None:
        """Validate Redis connectivity. Raises on failure."""
        try:
            await self._redis.ping()
        except redis.exceptions.RedisError as exc:
            raise ConnectionError(
                f"Redis connection failed for CrawlCache: {exc}. "
                f"Check REDIS_URL (password required if Redis has requirepass)."
            ) from exc

    async def get_headers(self, url_hash: str) -> dict[str, str] | None:
        """Return cached headers {status, etag, last_modified} or None."""
        try:
            key = f"{self.prefix}{url_hash}"
            data = await self._redis.hgetall(key)
            if not data:
                return None
            return data
        except redis.exceptions.RedisError as exc:
            log.warning(
                "CrawlCache.get_headers failed",
                url_hash=url_hash,
                error_type=type(exc).__name__,
                error=str(exc),
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
        try:
            key = f"{self.prefix}{url_hash}"
            mapping: dict[str, str] = {"status": str(status)}
            if etag:
                mapping["etag"] = etag
            if last_modified:
                mapping["last_modified"] = last_modified
            await self._redis.hset(key, mapping=mapping)
        except redis.exceptions.RedisError as exc:
            log.warning(
                "CrawlCache.set_headers failed",
                url_hash=url_hash,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def close(self) -> None:
        with contextlib.suppress(redis.exceptions.RedisError):
            await self._redis.close()
