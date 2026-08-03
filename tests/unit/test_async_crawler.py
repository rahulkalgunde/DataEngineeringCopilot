"""Tests for AsyncDocumentationCrawler.

Uses respx for transport-level httpx mocking and AsyncMock for
internal method testing.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import respx
from httpx import Response

from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache
from data_engineering_copilot.infrastructure.crawl_db import CrawlRecord, PostgresCrawlFrontierDB


def _make_record(url: str = "https://example.com", state: str = "DISCOVERED") -> CrawlRecord:
    return CrawlRecord(
        url_hash=PostgresCrawlFrontierDB.hash_url(url),
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


def _make_httpx_response(**attrs) -> MagicMock:
    """Create a mock httpx.Response-like object with the given attributes."""
    resp = MagicMock(spec=Response)
    for k, v in attrs.items():
        setattr(resp, k, v)
    return resp


@pytest.fixture
def mock_frontier():
    f = AsyncMock(spec=PostgresCrawlFrontierDB)
    f.hash_url = PostgresCrawlFrontierDB.hash_url
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
    )


class TestAlwaysGet:
    """Tests for the always-GET crawl strategy (no conditional-GET fast path)."""

    def test_no_conditional_get_flag(self, crawler):
        assert not hasattr(crawler, "conditional_get")
        assert not hasattr(crawler, "_phase1_head")

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetches_page_with_cached_headers(self, crawler, mock_frontier, mock_cache):
        """Even when cache headers exist, the page is fetched (never skipped on 304)."""
        respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404, text="Not Found"))
        respx.get("https://example.com/").mock(
            return_value=Response(
                200,
                headers={"Content-Type": "text/html", "ETag": '"v1"'},
                text="<html><body><p>Hello world</p></body></html>",
            )
        )

        mock_frontier.get_pending = AsyncMock(return_value=[_make_record("https://example.com/")])
        mock_frontier.claim = AsyncMock(return_value=_make_record("https://example.com/"))
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
        )

        source = _make_source()
        docs = []
        async for doc in crawler.crawl(source, max_pages=1):
            docs.append(doc)

        assert len(docs) == 1
        assert "Hello world" in docs[0].html
        # The crawler must NOT mark the page processed — the ingestion worker
        # owns the PROCESSED transition after a successful index.
        mock_frontier.mark_processed.assert_not_awaited()


class TestPhase2Get:
    """Tests for _phase2_get (full GET request with retry)."""

    @pytest.mark.asyncio
    async def test_get_success(self, crawler, mock_frontier, mock_cache):
        record = _make_record()
        mock_cache.set_headers = AsyncMock()

        mock_resp = _make_httpx_response(
            status_code=200,
            headers={"Content-Type": "text/html", "ETag": '"v1"'},
            text="<html><body>Hello world test content enough words here to pass the check easily.</body></html>",
        )
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_resp)

        result = await crawler._phase2_get(mock_session, record)
        assert result is not None
        html, content_type = result
        assert "Hello world" in html
        assert content_type == "text/html"

    @pytest.mark.asyncio
    async def test_get_non_html_skips(self, crawler, mock_frontier, mock_cache):
        record = _make_record()

        mock_resp = _make_httpx_response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            text="{}",
        )
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_resp)

        result = await crawler._phase2_get(mock_session, record)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_text_plain_accepted(self, crawler, mock_frontier, mock_cache):
        record = _make_record()
        mock_cache.set_headers = AsyncMock()

        mock_resp = _make_httpx_response(
            status_code=200,
            headers={"Content-Type": "text/plain"},
            text="Some plain text content here for RST document processing pipeline.",
        )
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_resp)

        result = await crawler._phase2_get(mock_session, record)
        assert result is not None
        html, content_type = result
        assert content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_get_http_error_returns_none(self, crawler, mock_frontier, mock_cache):
        record = _make_record()

        mock_resp = _make_httpx_response(
            status_code=500,
            headers={"Content-Type": "text/html"},
            text="error",
        )
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_resp)

        html = await crawler._phase2_get(mock_session, record)
        assert html is None
        mock_frontier.mark_failed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_retries_on_exception(self, crawler, mock_frontier, mock_cache):
        record = _make_record()
        mock_cache.set_headers = AsyncMock()

        mock_resp = _make_httpx_response(
            status_code=200,
            headers={"Content-Type": "text/html", "ETag": '"v1"'},
            text="<html><body>Success after retry.</body></html>",
        )

        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[Exception("timeout"), mock_resp])

        result = await crawler._phase2_get(mock_session, record)
        assert result is not None
        html, content_type = result
        assert "Success after retry" in html
        assert content_type == "text/html"
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_redirect_followed_manually(self, crawler, mock_frontier, mock_cache):
        record = _make_record()
        mock_cache.set_headers = AsyncMock()

        redirect_resp = _make_httpx_response(
            status_code=301,
            headers={"Location": "https://example.com/final", "Content-Type": "text/html"},
            text="",
        )
        final_resp = _make_httpx_response(
            status_code=200,
            headers={"Content-Type": "text/html"},
            text="<html><body>Redirected content here.</body></html>",
        )

        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[redirect_resp, final_resp])

        result = await crawler._phase2_get(mock_session, record)
        assert result is not None
        html, content_type = result
        assert "Redirected content" in html


class TestExtractAndDiscover:
    @pytest.mark.asyncio
    async def test_discovers_links(self, crawler, mock_frontier):
        record = _make_record()
        mock_frontier.discover = AsyncMock(return_value="child_hash_123")
        mock_frontier.add_edge = AsyncMock()
        source = _make_source()
        html = '<html><body><a href="/docs/new-page">link</a></body></html>'
        mock_session = MagicMock()
        crawler._get_robots_parser = AsyncMock(return_value=MagicMock(can_fetch=MagicMock(return_value=True)))
        await crawler._extract_and_discover(mock_session, record, html, source)
        mock_frontier.discover.assert_awaited_once()
        mock_frontier.add_edge.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_filters_disallowed_domains(self, crawler, mock_frontier):
        record = _make_record()
        mock_frontier.discover = AsyncMock(return_value="child_hash")
        source = _make_source()
        html = '<html><body><a href="https://evil.com/malicious">bad link</a></body></html>'
        mock_session = MagicMock()
        crawler._get_robots_parser = AsyncMock(return_value=MagicMock(can_fetch=MagicMock(return_value=True)))
        await crawler._extract_and_discover(mock_session, record, html, source)
        mock_frontier.discover.assert_not_awaited()


class TestDomainPoliteness:
    @pytest.mark.asyncio
    async def test_enforce_delay_updates_timestamp(self, crawler):
        from data_engineering_copilot.infrastructure.async_crawler import _DomainState

        state = _DomainState(semaphore=asyncio.Semaphore(1), last_request_time=0.0)
        crawler.delay_seconds = 0.1
        await crawler._enforce_delay(state)
        assert state.last_request_time > 0.0


class TestCleanUrl:
    def test_strips_fragment(self, crawler):
        assert crawler._clean_url("https://example.com/page/#section") == "https://example.com/page"

    def test_lowercases_scheme_and_host(self, crawler):
        assert crawler._clean_url("HTTPS://Example.COM/path") == "https://example.com/path"

    def test_strips_tracking_params(self, crawler):
        url = "https://example.com/page?utm_source=twitter&utm_medium=social&real=1"
        assert crawler._clean_url(url) == "https://example.com/page?real=1"

    def test_strips_trailing_slash(self, crawler):
        assert crawler._clean_url("https://example.com/page/") == "https://example.com/page"

    def test_keeps_root_slash(self, crawler):
        assert crawler._clean_url("https://example.com/") == "https://example.com/"


class TestDedupeKey:
    def test_strips_index_html(self, crawler):
        assert crawler._dedupe_key("https://example.com/page/index.html") == "https://example.com/page"

    def test_strips_trailing_slash(self, crawler):
        assert crawler._dedupe_key("https://example.com/page/") == "https://example.com/page"

    def test_keeps_root_slash(self, crawler):
        assert crawler._dedupe_key("https://example.com/") == "https://example.com/"


class TestCrawlWithRespx:
    """Integration-style tests using respx for transport-level mocking."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_crawl_fetches_single_page(self, mock_frontier, mock_cache):
        respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404, text="Not Found"))
        respx.get("https://example.com/").mock(
            return_value=Response(
                200,
                headers={"Content-Type": "text/html", "ETag": '"v1"'},
                text="<html><body><p>Hello world</p></body></html>",
            )
        )

        mock_frontier.get_pending = AsyncMock(return_value=[_make_record("https://example.com/")])
        mock_frontier.claim = AsyncMock(return_value=_make_record("https://example.com/"))
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
        )

        source = _make_source()
        docs = []
        async for doc in crawler.crawl(source, max_pages=1):
            docs.append(doc)

        assert len(docs) == 1
        mock_frontier.claim.assert_awaited()
        mock_frontier.mark_processed.assert_not_awaited()
        assert "Hello world" in docs[0].html

    @pytest.mark.asyncio
    @respx.mock
    async def test_crawl_gets_page_even_if_server_would_304(self, mock_frontier, mock_cache):
        """The crawler always issues a full GET; cached (304-eligible) pages are
        still fetched so the ingestion pipeline can verify against Qdrant."""
        respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404, text="Not Found"))
        respx.get("https://example.com/").mock(
            return_value=Response(
                200,
                headers={"Content-Type": "text/html", "ETag": '"v1"'},
                text="<html><body><p>Still present</p></body></html>",
            )
        )

        mock_frontier.get_pending = AsyncMock(return_value=[_make_record("https://example.com/")])
        mock_frontier.claim = AsyncMock(return_value=_make_record("https://example.com/"))
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
        )

        source = _make_source()
        docs = []
        async for doc in crawler.crawl(source, max_pages=1):
            docs.append(doc)

        assert len(docs) == 1
        assert "Still present" in docs[0].html
        mock_frontier.mark_processed.assert_not_awaited()


class TestSeedFrontier:
    """Tests for _seed_frontier seeding semantics."""

    @pytest.mark.asyncio
    async def test_seeds_sitemap_urls_and_start_urls(self, crawler, mock_frontier):
        source = _make_source()
        crawler._try_sitemap = AsyncMock(
            return_value=[
                "https://example.com/docs/a.html",
                "https://example.com/docs/b.html",
            ]
        )

        await crawler._seed_frontier(source, max_pages=0)

        discovered = {call.kwargs["url"] for call in mock_frontier.discover.await_args_list}
        assert discovered == {
            "https://example.com/docs/a.html",
            "https://example.com/docs/b.html",
            "https://example.com",
        }

    @pytest.mark.asyncio
    async def test_seeds_start_urls_when_sitemap_unavailable(self, crawler, mock_frontier):
        source = _make_source()
        crawler._try_sitemap = AsyncMock(return_value=None)

        await crawler._seed_frontier(source, max_pages=0)

        mock_frontier.discover.assert_awaited_once_with(
            url="https://example.com",
            source_name="test",
            parent_hash=None,
            depth=0,
        )

    @pytest.mark.asyncio
    async def test_start_url_duplicate_in_sitemap_seeded_once(self, crawler, mock_frontier):
        source = _make_source()
        crawler._try_sitemap = AsyncMock(
            return_value=[
                "https://example.com",
                "https://example.com/docs/a.html",
            ]
        )

        await crawler._seed_frontier(source, max_pages=0)

        discovered = [call.kwargs["url"] for call in mock_frontier.discover.await_args_list]
        assert discovered.count("https://example.com") == 1
        assert len(discovered) == 2
