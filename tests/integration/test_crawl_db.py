"""Tests for PostgresCrawlFrontierDB.

These tests require a running PostgreSQL instance, provided by a session-scoped
Postgres testcontainer. Marked ``serial`` because every test drops and recreates
the same ``crawl_frontier`` schema.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from data_engineering_copilot.infrastructure.crawl_db import (
    PostgresCrawlFrontierDB,
)

pytestmark = [pytest.mark.integration, pytest.mark.serial]


@pytest_asyncio.fixture
async def frontier(pg_dsn):
    f = PostgresCrawlFrontierDB(pg_dsn)
    try:
        await f.initialize()
    except Exception:
        pytest.skip("PostgreSQL unreachable")
    yield f
    await f.drop_all()
    await f.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(frontier):
    assert frontier._pool is not None
    async with frontier._pool.acquire() as conn:
        tables = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        table_names = {row["tablename"] for row in tables}
        assert "crawl_frontier" in table_names
        assert "sitemap_edges" in table_names


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
async def test_discover_processed_not_rediscovered(frontier):
    """Bug fix: PROCESSED pages must NOT be re-discovered by discover()."""
    h1 = await frontier.discover("https://example.com", "test", None, 0)
    await frontier.claim(h1)
    await frontier.mark_processed(h1)

    # Attempt to re-discover the same URL from a different parent
    h2 = await frontier.discover("https://example.com", "test", h1, 1)
    assert h2 is None

    # Verify the page stayed PROCESSED (not reset to DISCOVERED)
    record = await frontier.get_record(h1)
    assert record.state == "PROCESSED"


@pytest.mark.asyncio
async def test_discover_failed_can_be_rediscovered(frontier):
    """FAILED pages CAN be re-discovered (retry transient failures)."""
    h1 = await frontier.discover("https://example.com", "test", None, 0)
    await frontier.claim(h1)
    await frontier.mark_failed(h1, "HTTP 500")

    # Re-discover should work for FAILED pages
    h2 = await frontier.discover("https://example.com", "test", h1, 1)
    assert h2 is not None

    record = await frontier.get_record(h1)
    assert record.state == "DISCOVERED"


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


@pytest.mark.asyncio
async def test_rediscover_children(frontier):
    """Test that rediscover_children sets PROCESSED children back to DISCOVERED."""
    h_parent = await frontier.discover("https://parent.com", "test", None, 0)
    h_child1 = await frontier.discover("https://child1.com", "test", h_parent, 1)
    h_child2 = await frontier.discover("https://child2.com", "test", h_parent, 1)

    await frontier.claim(h_child1)
    await frontier.mark_processed(h_child1)
    await frontier.claim(h_child2)
    await frontier.mark_processed(h_child2)

    r1 = await frontier.get_record(h_child1)
    r2 = await frontier.get_record(h_child2)
    assert r1.state == "PROCESSED"
    assert r2.state == "PROCESSED"

    rediscovered = await frontier.rediscover_children(h_parent, "test", 2)
    assert rediscovered == 2

    r1 = await frontier.get_record(h_child1)
    r2 = await frontier.get_record(h_child2)
    assert r1.state == "DISCOVERED"
    assert r1.depth == 2
    assert r2.state == "DISCOVERED"
    assert r2.depth == 2


@pytest.mark.asyncio
async def test_rediscover_children_skips_discovered(frontier):
    """Test that rediscover_children doesn't change children already in DISCOVERED state."""
    h_parent = await frontier.discover("https://parent.com", "test", None, 0)
    h_child = await frontier.discover("https://child.com", "test", h_parent, 1)

    r = await frontier.get_record(h_child)
    assert r.state == "DISCOVERED"

    rediscovered = await frontier.rediscover_children(h_parent, "test", 2)
    assert rediscovered == 0

    r = await frontier.get_record(h_child)
    assert r.state == "DISCOVERED"
    assert r.depth == 1


@pytest.mark.asyncio
async def test_rediscover_children_no_edges(frontier):
    """Test that rediscover_children returns 0 when parent has no edges."""
    h_parent = await frontier.discover("https://parent.com", "test", None, 0)
    rediscovered = await frontier.rediscover_children(h_parent, "test", 2)
    assert rediscovered == 0


@pytest.mark.asyncio
async def test_rediscover_children_includes_failed(frontier):
    """Test that rediscover_children also re-discovers FAILED children."""
    h_parent = await frontier.discover("https://parent.com", "test", None, 0)
    h_child = await frontier.discover("https://child.com", "test", h_parent, 1)

    await frontier.claim(h_child)
    await frontier.mark_failed(h_child, "HTTP 500")

    r = await frontier.get_record(h_child)
    assert r.state == "FAILED"

    rediscovered = await frontier.rediscover_children(h_parent, "test", 2)
    assert rediscovered == 1

    r = await frontier.get_record(h_child)
    assert r.state == "DISCOVERED"
    assert r.depth == 2
