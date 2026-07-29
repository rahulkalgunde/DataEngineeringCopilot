"""Integration tests for CrawlCache with real Redis."""

from __future__ import annotations

import hashlib

import pytest

from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


@pytest.fixture
def url_hash():
    return hashlib.sha256(b"https://example.com/test").hexdigest()[:16]


async def test_crawl_cache_set_and_get_with_real_redis(redis_url, url_hash):
    cache = CrawlCache(redis_url)
    try:
        await cache.set_headers(url_hash, status=200, etag='"abc123"', last_modified="Thu, 01 Jan 2026 00:00:00 GMT")
        result = await cache.get_headers(url_hash)
        assert result is not None
        assert result["status"] == "200"
        assert result["etag"] == '"abc123"'
        assert result["last_modified"] == "Thu, 01 Jan 2026 00:00:00 GMT"
    finally:
        await cache.close()


async def test_crawl_cache_returns_none_when_redis_down(url_hash):
    cache = CrawlCache("redis://localhost:9999/0")
    try:
        result = await cache.get_headers(url_hash)
        assert result is None
    finally:
        await cache.close()


async def test_crawl_cache_close_handles_error():
    cache = CrawlCache("redis://localhost:9999/0")
    await cache.close()


@pytest.mark.serial
async def test_crawl_cache_set_headers_without_optional_fields(redis_url, url_hash):
    cache = CrawlCache(redis_url)
    try:
        await cache.set_headers(url_hash, status=304)
        result = await cache.get_headers(url_hash)
        assert result is not None
        assert result["status"] == "304"
        assert "etag" not in result
        assert "last_modified" not in result
    finally:
        await cache.close()
