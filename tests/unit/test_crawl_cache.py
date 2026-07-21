from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache


@pytest.fixture
def cache():
    with patch("data_engineering_copilot.infrastructure.crawl_cache.aioredis") as mock_redis:
        mock_conn = AsyncMock()
        mock_redis.from_url.return_value = mock_conn
        c = CrawlCache(redis_url="redis://localhost:6379/1", prefix="test:")
        c._redis = mock_conn
        yield c, mock_conn


@pytest.mark.asyncio
async def test_set_and_get_headers_roundtrip(cache):
    c, mock_conn = cache
    mock_conn.hgetall = AsyncMock(return_value={"status": "200", "etag": '"abc"'})
    await c.set_headers("hash1", status=200, etag='"abc"')
    mock_conn.hset.assert_awaited_once()
    result = await c.get_headers("hash1")
    assert result is not None
    assert result["status"] == "200"
    assert result["etag"] == '"abc"'


@pytest.mark.asyncio
async def test_get_headers_miss(cache):
    c, mock_conn = cache
    mock_conn.hgetall = AsyncMock(return_value={})
    result = await c.get_headers("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_etag_only(cache):
    c, mock_conn = cache
    mock_conn.hgetall = AsyncMock(return_value={"status": "304", "etag": '"xyz"'})
    result = await c.get_headers("hash2")
    assert result is not None
    assert "last_modified" not in result


@pytest.mark.asyncio
async def test_last_modified_only(cache):
    c, mock_conn = cache
    mock_conn.hgetall = AsyncMock(return_value={"status": "200", "last_modified": "Wed, 01 Jan 2025 00:00:00 GMT"})
    await c.set_headers("hash3", status=200, last_modified="Wed, 01 Jan 2025 00:00:00 GMT")
    result = await c.get_headers("hash3")
    assert result is not None
    assert result["last_modified"] == "Wed, 01 Jan 2025 00:00:00 GMT"
    assert "etag" not in result
