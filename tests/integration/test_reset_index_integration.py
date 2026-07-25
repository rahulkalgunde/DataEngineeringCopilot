"""Integration tests for reset-index cache clearing."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


@pytest.fixture
def sync_redis_client(redis_url):
    import redis

    client = redis.from_url(redis_url, decode_responses=False)
    yield client
    client.close()


async def test_reset_index_clears_all_cache_keys(sync_redis_client):
    sync_redis_client.hset("crawl:url_registry:SourceA", "url1", '{"html_hash":"h1"}')
    sync_redis_client.hset("crawl:url_registry:SourceB", "url2", '{"html_hash":"h2"}')
    sync_redis_client.hset("crawl:header:abc123", "status", "200")
    sync_redis_client.hset("crawl:header:def456", "etag", '"xyz"')

    from data_engineering_copilot.cli import reset_index

    with pytest.raises(SystemExit):
        reset_index()

    remaining = list(sync_redis_client.scan_iter("crawl:*"))
    assert len(remaining) == 0


async def test_reset_index_enables_fresh_crawl(sync_redis_client):
    sync_redis_client.hset("crawl:header:old_page", "etag", '"old_etag"')

    from data_engineering_copilot.cli import reset_index

    with pytest.raises(SystemExit):
        reset_index()

    result = sync_redis_client.hget("crawl:header:old_page", "etag")
    assert result is None
