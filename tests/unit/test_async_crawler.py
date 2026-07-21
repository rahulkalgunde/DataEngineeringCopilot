from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.infrastructure.async_crawler import (
    AsyncDocumentationCrawler,
)
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


@pytest.mark.asyncio
async def test_phase1_head_304_skips(crawler, mock_frontier, mock_cache):
    record = _make_record()
    mock_cache.get_headers = AsyncMock(return_value={"status": "200", "etag": '"abc"'})
    mock_frontier.mark_processed = AsyncMock()

    with patch("data_engineering_copilot.infrastructure.async_crawler.aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 304
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.head = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = await crawler._phase1_head(mock_session, record, {"etag": '"abc"'})
        assert result is True


@pytest.mark.asyncio
async def test_phase1_head_200_updates_cache(crawler, mock_cache):
    record = _make_record()
    with patch("data_engineering_copilot.infrastructure.async_crawler.aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"ETag": '"new"', "Last-Modified": "Tue, 01 Jan 2025 00:00:00 GMT"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.head = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = await crawler._phase1_head(mock_session, record, {"etag": '"old"'})
        assert result is False
        mock_cache.set_headers.assert_awaited_once()


@pytest.mark.asyncio
async def test_phase2_get_success(crawler, mock_frontier, mock_cache):
    record = _make_record()
    mock_frontier.mark_failed = AsyncMock()
    mock_cache.set_headers = AsyncMock()

    with patch("data_engineering_copilot.infrastructure.async_crawler.aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/html", "ETag": '"v1"'}
        mock_resp.text = AsyncMock(
            return_value="<html><body>Hello world test content enough words here to pass the check easily.</body></html>"
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        html = await crawler._phase2_get(mock_session, record)
        assert html is not None
        assert "Hello world" in html


@pytest.mark.asyncio
async def test_phase2_get_non_html_skips(crawler, mock_frontier, mock_cache):
    record = _make_record()
    mock_frontier.mark_failed = AsyncMock()

    with patch("data_engineering_copilot.infrastructure.async_crawler.aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        html = await crawler._phase2_get(mock_session, record)
        assert html is None


@pytest.mark.asyncio
async def test_extract_and_discover(crawler, mock_frontier):
    record = _make_record()
    mock_frontier.discover = AsyncMock(return_value="child_hash_123")
    mock_frontier.add_edge = AsyncMock()
    source = _make_source()
    html = '<html><body><a href="/docs/new-page">link</a></body></html>'
    await crawler._extract_and_discover(record, html, source)
    mock_frontier.discover.assert_awaited_once()
    mock_frontier.add_edge.assert_awaited_once()


@pytest.mark.asyncio
async def test_domain_politeness_delay(crawler):
    from data_engineering_copilot.infrastructure.async_crawler import _DomainState

    state = _DomainState(semaphore=asyncio.Semaphore(1), last_request_time=0.0)
    state.last_request_time = 0.0
    crawler.delay_seconds = 0.1
    await crawler._enforce_delay(state)
    assert state.last_request_time > 0.0


def test_dedupe_key(crawler):
    assert crawler._dedupe_key("https://example.com/page/index.html") == "https://example.com/page"
    assert crawler._dedupe_key("https://example.com/page/") == "https://example.com/page"
    assert crawler._dedupe_key("https://example.com/") == "https://example.com/"
