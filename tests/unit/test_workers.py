"""Tests for Celery worker tasks."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


def test_celery_app():
    from data_engineering_copilot.workers.celery_app import celery_app

    assert celery_app.main == "data_engineering_copilot"


@pytest.mark.asyncio
@patch("crawl4ai.AsyncWebCrawler")
async def test_run_async_crawl(mock_crawler_class):
    from data_engineering_copilot.workers.tasks import _run_async_crawl

    mock_crawler = AsyncMock()
    mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
    mock_crawler.arun.return_value = "result1"

    results = await _run_async_crawl(["http://test.com"])
    assert results == ["result1"]
    mock_crawler.arun.assert_called_once_with(url="http://test.com")
