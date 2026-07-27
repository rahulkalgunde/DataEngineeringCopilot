"""Integration tests for AsyncUrlRegistry with real Redis."""

from __future__ import annotations

import pytest

from data_engineering_copilot.infrastructure.async_url_registry import AsyncUrlRegistry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


@pytest.fixture
async def async_redis_client(redis_url):
    import redis.asyncio as aioredis

    client = aioredis.from_url(redis_url, decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def sync_redis_client(redis_url):
    import redis

    client = redis.from_url(redis_url, decode_responses=False)
    yield client
    client.close()


async def test_async_url_registry_roundtrip(async_redis_client):
    registry = AsyncUrlRegistry(async_redis_client, "test_source")
    url = "https://example.com/page.html"
    html_hash = "abc123def456"

    await registry.set_html_hash(url, html_hash)
    result = await registry.get_html_hash(url)

    assert result == html_hash


async def test_async_url_registry_returns_none_for_missing(async_redis_client):
    registry = AsyncUrlRegistry(async_redis_client, "test_source")
    result = await registry.get_html_hash("https://example.com/nonexistent")
    assert result is None


async def test_async_url_registry_clear(async_redis_client):
    registry = AsyncUrlRegistry(async_redis_client, "test_clear_source")
    await registry.set_html_hash("https://example.com/a", "hash_a")
    await registry.set_html_hash("https://example.com/b", "hash_b")

    await registry.clear()

    result_a = await registry.get_html_hash("https://example.com/a")
    result_b = await registry.get_html_hash("https://example.com/b")
    assert result_a is None
    assert result_b is None


async def test_async_url_registry_handles_none_redis():
    registry = AsyncUrlRegistry(None, "test_source")
    result = await registry.get_html_hash("https://example.com")
    assert result is None
    await registry.set_html_hash("https://example.com", "hash")
    await registry.clear()


async def test_async_url_registry_rejects_bad_redis_client():
    with pytest.raises(TypeError, match="must implement SyncRedisProtocol"):
        AsyncUrlRegistry("not_a_redis_client", "test_source")
