"""Tests for AsyncDocumentationCrawler.

Uses aresponses for transport-level aiohttp mocking and AsyncMock for
internal method testing.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache
from data_engineering_copilot.infrastructure.crawl_db import CrawlFrontierDB, CrawlRecord


def _make_record(url: str = "https://example.com", state: str = "DISCOVERED") -> CrawlRecord:
    return CrawlRecord(
        url_hash=CrawlFrontierDB.hash_url(url),
        url=url,
        source_name="test",
        state=state,
        parent_hash=None,
        depth=0,
        etag=None,
        last_modified=None,
        attempts=0,
        last_error=None,
        created_at=0.0,
        updated_at=0.0,
    )


def _make_source():
    return DocumentationSource(
        name="test",
        start_urls=("https://example.com",),
        allowed_domains=("example.com",),
        url_prefixes=("https://example.com/",),
    )


def _make_context_response(**attrs):
    """Create a mock that works as `async with session.get(...) as resp:`."""
    resp = AsyncMock()
    for k, v in attrs.items():
        setattr(resp, k, v)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


@pytest.fixture
def mock_frontier():
    f = AsyncMock(spec=CrawlFrontierDB)
    f.hash_url = CrawlFrontierDB.hash_url
    return f


@pytest.fixture
def mock_cache():
    c = AsyncMock(spec=CrawlCache)
    return c


@pytest.fixture
def crawler(mock_frontier, mock_cache):
    return AsyncDocumentationCrawler(
        frontier=mock_frontier,
        cache=mock_cache,
        timeout_seconds=5,
        delay_seconds=0.0,
        concurrency=5,
        max_concurrency=20,
        max_retries=2,
        conditional_get=True,
    )


class TestPhase1Head:
    """Tests for _phase1_head (conditional GET via HEAD request)."""

    @pytest.mark.asyncio
    async def test_head_304_returns_true(self, crawler, mock_cache):
        record = _make_record()
        mock_cache.get_headers = AsyncMock(return_value={"status": "200", "etag": '"abc"'})

        mock_resp = _make_context_response(status=304)
        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=mock_resp)

        result = await crawler._phase1_head(mock_session, record, {"etag": '"abc"'})
        assert result is True

    @pytest.mark.asyncio
    async def test_head_200_returns_false_and_updates_cache(self, crawler, mock_cache):
        record = _make_record()

        mock_resp = _make_context_response(
            status=200,
            headers={"ETag": '"new"', "Last-Modified": "Tue, 01 Jan 2025 00:00:00 GMT"},
        )
        mock_session = MagicMock()
        mock_session.head = MagicMock(return_value=mock_resp)

        result = await crawler._phase1_head(mock_session, record, {"etag": '"old"'})
        assert result is False
        mock_cache.set_headers.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_head_exception_returns_false(self, crawler, mock_cache):
        record = _make_record()
        mock_cache.get_headers = AsyncMock(return_value={"etag": '"abc"'})

        mock_session = MagicMock()
        mock_session.head = MagicMock(side_effect=Exception("network error"))

        result = await crawler._phase1_head(mock_session, record, {"etag": '"abc"'})
        assert result is False


class TestPhase2Get:
    """Tests for _phase2_get (full GET request with retry)."""

    @pytest.mark.asyncio
    async def test_get_success(self, crawler, mock_frontier, mock_cache):
        record = _make_record()
        mock_cache.set_headers = AsyncMock()

        mock_resp = _make_context_response(
            status=200,
            headers={"Content-Type": "text/html", "ETag": '"v1"'},
        )
        mock_resp.text = AsyncMock(
            return_value="<html><body>Hello world test content enough words here to pass the check easily.</body></html>"
        )
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        html = await crawler._phase2_get(mock_session, record)
        assert html is not None
        assert "Hello world" in html

    @pytest.mark.asyncio
    async def test_get_non_html_skips(self, crawler, mock_frontier, mock_cache):
        record = _make_record()

        mock_resp = _make_context_response(
            status=200,
            headers={"Content-Type": "application/json"},
        )
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        html = await crawler._phase2_get(mock_session, record)
        assert html is None

    @pytest.mark.asyncio
    async def test_get_http_error_returns_none(self, crawler, mock_frontier, mock_cache):
        record = _make_record()

        mock_resp = _make_context_response(
            status=500,
            headers={"Content-Type": "text/html"},
        )
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        html = await crawler._phase2_get(mock_session, record)
        assert html is None
        mock_frontier.mark_failed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_retries_on_exception(self, crawler, mock_frontier, mock_cache):
        record = _make_record()
        mock_cache.set_headers = AsyncMock()

        mock_resp = _make_context_response(
            status=200,
            headers={"Content-Type": "text/html", "ETag": '"v1"'},
        )
        mock_resp.text = AsyncMock(return_value="<html><body>Success after retry.</body></html>")

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=[Exception("timeout"), mock_resp])

        html = await crawler._phase2_get(mock_session, record)
        assert html is not None
        assert "Success after retry" in html
        assert mock_session.get.call_count == 2


class TestExtractAndDiscover:
    @pytest.mark.asyncio
    async def test_discovers_links(self, crawler, mock_frontier):
        record = _make_record()
        mock_frontier.discover = AsyncMock(return_value="child_hash_123")
        mock_frontier.add_edge = AsyncMock()
        source = _make_source()
        html = '<html><body><a href="/docs/new-page">link</a></body></html>'
        await crawler._extract_and_discover(record, html, source)
        mock_frontier.discover.assert_awaited_once()
        mock_frontier.add_edge.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_filters_disallowed_domains(self, crawler, mock_frontier):
        record = _make_record()
        mock_frontier.discover = AsyncMock(return_value="child_hash")
        source = _make_source()
        html = '<html><body><a href="https://evil.com/malicious">bad link</a></body></html>'
        await crawler._extract_and_discover(record, html, source)
        mock_frontier.discover.assert_not_awaited()


class TestDomainPoliteness:
    @pytest.mark.asyncio
    async def test_enforce_delay_updates_timestamp(self, crawler):
        from data_engineering_copilot.infrastructure.async_crawler import _DomainState

        state = _DomainState(semaphore=asyncio.Semaphore(1), last_request_time=0.0)
        crawler.delay_seconds = 0.1
        await crawler._enforce_delay(state)
        assert state.last_request_time > 0.0


class TestDedupeKey:
    def test_strips_index_html(self, crawler):
        assert crawler._dedupe_key("https://example.com/page/index.html") == "https://example.com/page"

    def test_strips_trailing_slash(self, crawler):
        assert crawler._dedupe_key("https://example.com/page/") == "https://example.com/page"

    def test_keeps_root_slash(self, crawler):
        assert crawler._dedupe_key("https://example.com/") == "https://example.com/"


class TestCrawlWithAresponses:
    """Integration-style tests using aresponses for transport-level mocking."""

    @pytest.mark.asyncio
    async def test_crawl_fetches_single_page(self, mock_frontier, mock_cache, aresponses):
        aresponses.add(
            "example.com",
            "/sitemap.xml",
            "GET",
            aresponses.Response(status=404, text="Not Found"),
        )
        aresponses.add(
            "example.com",
            "/",
            "GET",
            aresponses.Response(
                status=200,
                headers={"Content-Type": "text/html", "ETag": '"v1"'},
                text="<html><body><p>Hello world</p></body></html>",
            ),
        )

        mock_frontier.get_pending = AsyncMock(return_value=[_make_record("https://example.com/")])
        mock_frontier.mark_processed = AsyncMock()
        mock_frontier._db = "not_none"
        mock_cache.get_headers = AsyncMock(return_value=None)
        mock_cache.set_headers = AsyncMock()

        crawler = AsyncDocumentationCrawler(
            frontier=mock_frontier,
            cache=mock_cache,
            timeout_seconds=5,
            delay_seconds=0.0,
            concurrency=1,
            max_concurrency=5,
            max_retries=1,
            conditional_get=False,
        )

        source = _make_source()
        docs = []
        async for doc in crawler.crawl(source, max_pages=1):
            docs.append(doc)

        assert len(docs) == 1
        assert "Hello world" in docs[0].html
        mock_frontier.mark_processed.assert_awaited()

    @pytest.mark.asyncio
    async def test_crawl_skips_304_cached(self, mock_frontier, mock_cache, aresponses):
        aresponses.add(
            "example.com",
            "/sitemap.xml",
            "GET",
            aresponses.Response(status=404, text="Not Found"),
        )
        aresponses.add(
            "example.com",
            "/",
            "HEAD",
            aresponses.Response(status=304),
        )

        mock_frontier.get_pending = AsyncMock(return_value=[_make_record("https://example.com/")])
        mock_frontier.mark_processed = AsyncMock()
        mock_frontier._db = "not_none"
        mock_cache.get_headers = AsyncMock(return_value={"status": "200", "etag": '"v1"'})
        mock_cache.set_headers = AsyncMock()

        crawler = AsyncDocumentationCrawler(
            frontier=mock_frontier,
            cache=mock_cache,
            timeout_seconds=5,
            delay_seconds=0.0,
            concurrency=1,
            max_concurrency=5,
            max_retries=1,
            conditional_get=True,
        )

        source = _make_source()
        docs = []
        async for doc in crawler.crawl(source, max_pages=1):
            docs.append(doc)

        assert len(docs) == 0
        mock_frontier.mark_processed.assert_awaited()
