from __future__ import annotations

import logging

import redis.asyncio as aioredis

log = logging.getLogger(__name__)


class CrawlCache:
    """Ephemeral Redis hash-based cache for HTTP conditional-GET headers.

    Each URL is keyed by its SHA-256 hash. Fields stored: status, etag, last_modified.
    """

    def __init__(self, redis_url: str, prefix: str = "crawl:") -> None:
        self.prefix = prefix
        self._redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            max_connections=20,
        )

    async def get_headers(self, url_hash: str) -> dict[str, str] | None:
        """Return cached headers {status, etag, last_modified} or None."""
        key = f"{self.prefix}{url_hash}"
        data = await self._redis.hgetall(key)
        if not data:
            return None
        return data

    async def set_headers(
        self,
        url_hash: str,
        status: int,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        """Cache response headers for a URL."""
        key = f"{self.prefix}{url_hash}"
        mapping: dict[str, str] = {"status": str(status)}
        if etag:
            mapping["etag"] = etag
        if last_modified:
            mapping["last_modified"] = last_modified
        await self._redis.hset(key, mapping=mapping)

    async def close(self) -> None:
        await self._redis.close()
