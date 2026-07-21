from __future__ import annotations

import os
import tempfile

import pytest
import pytest_asyncio

from data_engineering_copilot.infrastructure.crawl_db import (
    CrawlFrontierDB,
)


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest_asyncio.fixture
async def frontier(db_path):
    f = CrawlFrontierDB(db_path)
    await f.initialize()
    yield f
    await f.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(frontier):
    assert frontier._db is not None
    cursor = await frontier._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in await cursor.fetchall()}
    assert "crawl_frontier" in tables
    assert "sitemap_edges" in tables


@pytest.mark.asyncio
async def test_discover_new_url(frontier):
    url_hash = await frontier.discover(
        url="https://spark.apache.org/docs/latest/",
        source_name="Apache Spark",
        parent_hash=None,
        depth=0,
    )
    assert url_hash is not None
    assert len(url_hash) == 64
    record = await frontier.get_record(url_hash)
    assert record is not None
    assert record.state == "DISCOVERED"
    assert record.url == "https://spark.apache.org/docs/latest/"
    assert record.source_name == "Apache Spark"
    assert record.depth == 0


@pytest.mark.asyncio
async def test_discover_duplicate_returns_none(frontier):
    h1 = await frontier.discover("https://example.com", "test", None, 0)
    h2 = await frontier.discover("https://example.com", "test", None, 0)
    assert h1 is not None
    assert h2 is None


@pytest.mark.asyncio
async def test_claim_success(frontier):
    url_hash = await frontier.discover("https://example.com", "test", None, 0)
    assert url_hash is not None
    record = await frontier.claim(url_hash)
    assert record is not None
    assert record.state == "FETCHING"
    assert record.attempts == 1


@pytest.mark.asyncio
async def test_claim_already_fetching(frontier):
    url_hash = await frontier.discover("https://example.com", "test", None, 0)
    await frontier.claim(url_hash)
    result = await frontier.claim(url_hash)
    assert result is None


@pytest.mark.asyncio
async def test_mark_processed(frontier):
    url_hash = await frontier.discover("https://example.com", "test", None, 0)
    await frontier.claim(url_hash)
    await frontier.mark_processed(url_hash)
    record = await frontier.get_record(url_hash)
    assert record is not None
    assert record.state == "PROCESSED"


@pytest.mark.asyncio
async def test_mark_failed(frontier):
    url_hash = await frontier.discover("https://example.com", "test", None, 0)
    await frontier.claim(url_hash)
    await frontier.mark_failed(url_hash, "HTTP 500")
    record = await frontier.get_record(url_hash)
    assert record is not None
    assert record.state == "FAILED"
    assert record.last_error == "HTTP 500"


@pytest.mark.asyncio
async def test_reset_stranded(frontier):
    h1 = await frontier.discover("https://a.com", "test", None, 0)
    h2 = await frontier.discover("https://b.com", "test", None, 0)
    await frontier.claim(h1)
    await frontier.claim(h2)
    count = await frontier.reset_stranded()
    assert count == 2
    r1 = await frontier.get_record(h1)
    r2 = await frontier.get_record(h2)
    assert r1 is not None and r1.state == "DISCOVERED"
    assert r2 is not None and r2.state == "DISCOVERED"


@pytest.mark.asyncio
async def test_get_pending_ordering(frontier):
    await frontier.discover("https://a.com/deep/1", "test", None, 2)
    await frontier.discover("https://b.com/shallow", "test", None, 0)
    await frontier.discover("https://c.com/mid", "test", None, 1)
    pending = await frontier.get_pending("test", limit=10)
    assert len(pending) == 3
    depths = [r.depth for r in pending]
    assert depths == sorted(depths)


@pytest.mark.asyncio
async def test_add_edge_and_get_edges(frontier):
    h_parent = await frontier.discover("https://parent.com", "test", None, 0)
    h_child = await frontier.discover("https://child.com", "test", h_parent, 1)
    assert h_parent is not None and h_child is not None
    edges = await frontier.get_edges(h_parent)
    assert h_child in edges


@pytest.mark.asyncio
async def test_stats(frontier):
    await frontier.discover("https://a.com", "src1", None, 0)
    await frontier.discover("https://b.com", "src1", None, 0)
    h = await frontier.discover("https://c.com", "src2", None, 0)
    await frontier.claim(h)
    await frontier.mark_processed(h)
    stats = await frontier.stats()
    assert stats.get("DISCOVERED") == 2
    assert stats.get("PROCESSED") == 1
    stats_src1 = await frontier.stats("src1")
    assert stats_src1.get("DISCOVERED") == 2
