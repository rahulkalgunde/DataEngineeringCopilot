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


async def test_reset_index_clears_all_cache_keys(sync_redis_client, monkeypatch, redis_url):
    sync_redis_client.hset("crawl:url_registry:SourceA", "url1", '{"html_hash":"h1"}')
    sync_redis_client.hset("crawl:url_registry:SourceB", "url2", '{"html_hash":"h2"}')
    sync_redis_client.hset("crawl:header:abc123", "status", "200")
    sync_redis_client.hset("crawl:header:def456", "etag", '"xyz"')

    # Monkeypatch settings to use testcontainer Redis
    from data_engineering_copilot import cli
    from data_engineering_copilot.config.settings import AppSettings

    test_settings = AppSettings(
        redis_url=redis_url,
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        embedding_provider="ollama",
        crawl_db_url="",
    )
    monkeypatch.setattr(cli, "settings", test_settings)

    # Patch get_redis_client in workers.progress (where reset_index imports it)
    import data_engineering_copilot.workers.progress as progress_mod

    monkeypatch.setattr(progress_mod, "get_redis_client", lambda: sync_redis_client)

    cli.reset_index()

    remaining = list(sync_redis_client.scan_iter("crawl:*"))
    assert len(remaining) == 0


async def test_reset_index_enables_fresh_crawl(sync_redis_client, monkeypatch, redis_url):
    sync_redis_client.hset("crawl:header:old_page", "etag", '"old_etag"')

    # Monkeypatch settings to use testcontainer Redis
    from data_engineering_copilot import cli
    from data_engineering_copilot.config.settings import AppSettings

    test_settings = AppSettings(
        redis_url=redis_url,
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        embedding_provider="ollama",
        crawl_db_url="",
    )
    monkeypatch.setattr(cli, "settings", test_settings)

    # Patch get_redis_client in workers.progress (where reset_index imports it)
    import data_engineering_copilot.workers.progress as progress_mod

    monkeypatch.setattr(progress_mod, "get_redis_client", lambda: sync_redis_client)

    cli.reset_index()

    result = sync_redis_client.hget("crawl:header:old_page", "etag")
    assert result is None
