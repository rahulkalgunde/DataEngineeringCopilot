"""Integration tests for reset-index / reset-qdrant cache and index clearing."""

from __future__ import annotations

import asyncio

import pytest

from data_engineering_copilot import cli
from data_engineering_copilot.config.settings import AppSettings

pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def sync_redis_client(redis_url):
    import redis

    client = redis.from_url(redis_url, decode_responses=False)
    yield client
    client.close()


def _patch_cli_settings(monkeypatch, redis_url, qdrant_url="http://localhost:6333", crawl_db_url=""):
    test_settings = AppSettings(
        redis_url=redis_url,
        qdrant_url=qdrant_url,
        collection_name="test_collection",
        embedding_provider="ollama",
        llm_provider="ollama",
        answer_llm_provider="ollama",
        rewrite_llm_provider="ollama",
        groundedness_llm_provider="ollama",
        intent_llm_provider="ollama",
        enrichment_llm_provider="",
        evaluation_llm_provider="ollama",
        code_llm_provider="",
        crawl_db_url=crawl_db_url,
        skip_provider_check=True,
    )
    monkeypatch.setattr(cli, "settings", test_settings)


def _patch_redis(monkeypatch, sync_redis_client):
    import data_engineering_copilot.workers.progress as progress_mod

    monkeypatch.setattr(progress_mod, "get_redis_client", lambda: sync_redis_client)


async def test_reset_index_clears_all_cache_keys(sync_redis_client, monkeypatch, redis_url):
    sync_redis_client.hset("crawl:url_registry:SourceA", "url1", '{"html_hash":"h1"}')
    sync_redis_client.hset("crawl:url_registry:SourceB", "url2", '{"html_hash":"h2"}')
    sync_redis_client.hset("crawl:header:abc123", "status", "200")
    sync_redis_client.hset("crawl:header:def456", "etag", '"xyz"')

    _patch_cli_settings(monkeypatch, redis_url)
    _patch_redis(monkeypatch, sync_redis_client)
    # No Qdrant available here; the collection-recreate step is not the focus.
    monkeypatch.setattr(cli, "_recreate_qdrant_collection", lambda: None)
    monkeypatch.setattr(cli, "_delete_bm25_cache", lambda: None)

    cli.reset_index()

    remaining = list(sync_redis_client.scan_iter("crawl:*"))
    assert len(remaining) == 0


async def test_reset_index_enables_fresh_crawl(sync_redis_client, monkeypatch, redis_url):
    sync_redis_client.hset("crawl:header:old_page", "etag", '"old_etag"')

    _patch_cli_settings(monkeypatch, redis_url)
    _patch_redis(monkeypatch, sync_redis_client)
    monkeypatch.setattr(cli, "_recreate_qdrant_collection", lambda: None)
    monkeypatch.setattr(cli, "_delete_bm25_cache", lambda: None)

    cli.reset_index()

    result = sync_redis_client.hget("crawl:header:old_page", "etag")
    assert result is None


def test_reset_index_full_rebuild(qdrant_url, redis_url, pg_dsn, tmp_path, monkeypatch):
    """Full clean rebuild: Qdrant collection recreated empty, BM25 gone, Redis
    crawl keys cleared, PG frontier tables dropped.

    Sync (not async) because ``reset_index`` runs its PG reset via
    ``asyncio.run``, which fails inside a live event loop.  Each async store
    lifecycle is kept within a single ``asyncio.run`` to respect loop-bound
    clients (asyncpg, httpx).
    """
    import redis as redis_lib

    from data_engineering_copilot.domain.models import DocumentChunk

    _patch_cli_settings(
        monkeypatch,
        redis_url=redis_url,
        qdrant_url=qdrant_url,
        crawl_db_url=pg_dsn,
    )

    bm25_path = tmp_path / ".bm25_cache" / "test_collection.json"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_text("{}")
    monkeypatch.setattr(cli, "_bm25_cache_path", lambda: bm25_path)

    # Seed Redis crawl keys
    redis_client = redis_lib.from_url(redis_url, decode_responses=False)
    redis_client.hset("crawl:url_registry:SourceA", "url1", '{"html_hash":"h1"}')
    _patch_redis(monkeypatch, redis_client)

    # Seed the Qdrant collection with a chunk (single loop for loop-bound client)
    chunk = DocumentChunk(
        chunk_id="c_https://example.com/keep",
        source_name="test",
        title="Title",
        url="https://example.com/keep",
        text="some indexed content",
    )

    async def _seed_qdrant() -> None:
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        store = AsyncQdrantVectorStore(url=qdrant_url, collection_name="test_collection", embedding_dimension=768)
        await store.initialize()
        await store.upsert_chunks([chunk], [[0.1] * 768])
        assert await store.count() == 1
        await store.close()

    asyncio.run(_seed_qdrant())

    # Seed the PG frontier (single loop)
    async def _seed_pg() -> None:
        from data_engineering_copilot.infrastructure.crawl_db import PostgresCrawlFrontierDB

        frontier = PostgresCrawlFrontierDB(pg_dsn)
        await frontier.initialize()
        await frontier.discover(url="https://example.com/keep", source_name="test", parent_hash=None, depth=0)
        await frontier.close()

    asyncio.run(_seed_pg())

    try:
        cli.reset_index()

        # Qdrant collection recreated and empty
        async def _verify_qdrant() -> None:
            from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

            fresh = AsyncQdrantVectorStore(url=qdrant_url, collection_name="test_collection", embedding_dimension=768)
            await fresh.initialize()
            assert await fresh.count() == 0
            await fresh.close()

        asyncio.run(_verify_qdrant())

        # BM25 cache removed
        assert not bm25_path.exists()

        # Redis crawl keys cleared
        remaining = list(redis_client.scan_iter("crawl:*"))
        assert len(remaining) == 0

        # PG frontier tables dropped
        async def _verify_pg() -> set[str]:
            import asyncpg

            conn = await asyncpg.connect(pg_dsn)
            try:
                rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            finally:
                await conn.close()
            return {row["tablename"] for row in rows}

        table_names = asyncio.run(_verify_pg())
        assert "crawl_frontier" not in table_names
        assert "sitemap_edges" not in table_names
    finally:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, prefer_grpc=False)
        client.delete_collection(collection_name="test_collection")
        client.close()
        redis_client.close()


def test_reset_qdrant_recreates_collection_and_deletes_bm25(qdrant_url, tmp_path, monkeypatch):
    """reset-qdrant recreates the collection (empty) and removes the BM25 file."""
    from data_engineering_copilot.domain.models import DocumentChunk

    monkeypatch.setattr(
        cli,
        "settings",
        AppSettings(
            qdrant_url=qdrant_url,
            collection_name="test_collection",
            embedding_provider="ollama",
            llm_provider="ollama",
            answer_llm_provider="ollama",
            rewrite_llm_provider="ollama",
            groundedness_llm_provider="ollama",
            intent_llm_provider="ollama",
            enrichment_llm_provider="",
            evaluation_llm_provider="ollama",
            code_llm_provider="",
            redis_url="redis://localhost:6379/0",
            skip_provider_check=True,
        ),
    )
    bm25_path = tmp_path / ".bm25_cache" / "test_collection.json"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_text("{}")
    monkeypatch.setattr(cli, "_bm25_cache_path", lambda: bm25_path)

    chunk = DocumentChunk(
        chunk_id="c_https://example.com/keep",
        source_name="test",
        title="Title",
        url="https://example.com/keep",
        text="some indexed content",
    )

    async def _seed() -> None:
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        store = AsyncQdrantVectorStore(url=qdrant_url, collection_name="test_collection", embedding_dimension=768)
        await store.initialize()
        await store.upsert_chunks([chunk], [[0.1] * 768])
        assert await store.count() == 1
        await store.close()

    asyncio.run(_seed())

    try:
        cli.reset_qdrant()

        assert not bm25_path.exists()

        async def _verify() -> None:
            from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

            fresh = AsyncQdrantVectorStore(url=qdrant_url, collection_name="test_collection", embedding_dimension=768)
            await fresh.initialize()
            assert await fresh.count() == 0
            await fresh.close()

        asyncio.run(_verify())
    finally:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, prefer_grpc=False)
        client.delete_collection(collection_name="test_collection")
        client.close()
