"""Tests for Celery worker tasks."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.workers.tasks import _run_async_crawl, execute_background_ingestion


def test_celery_app():
    from data_engineering_copilot.workers.celery_app import celery_app

    assert celery_app.main == "data_engineering_copilot"


@pytest.mark.asyncio
@patch("data_engineering_copilot.workers.tasks.AsyncWebCrawler")
async def test_run_async_crawl(mock_crawler_class):
    mock_crawler = AsyncMock()
    mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
    mock_crawler.arun.return_value = "result1"

    results = await _run_async_crawl(["http://test.com"])
    assert results == ["result1"]
    mock_crawler.arun.assert_called_once_with(url="http://test.com")


@patch("data_engineering_copilot.workers.tasks.AsyncWebCrawler")
@patch("data_engineering_copilot.workers.tasks.AsyncOllamaEmbeddings")
@patch("data_engineering_copilot.workers.tasks.DocumentChunker")
@patch("data_engineering_copilot.workers.tasks.AsyncQdrantVectorStore")
def test_execute_background_ingestion(mock_qdrant, mock_chunker_class, mock_embedder, mock_crawler_class):
    mock_crawler = AsyncMock()
    mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
    mock_doc_magic = MagicMock()
    mock_doc_magic.success = True
    mock_doc_magic.markdown = "Test markdown"
    mock_doc_magic.title = "Test Title"
    mock_doc_magic.url = "http://test.com"
    mock_crawler.arun.return_value = mock_doc_magic

    mock_chunker = mock_chunker_class.return_value
    mock_chunk = MagicMock()
    mock_chunk.text = "Test markdown"
    mock_chunker.chunk.return_value = [mock_chunk]

    mock_embed = mock_embedder.return_value
    mock_embed.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])

    mock_vector_store = mock_qdrant.return_value
    mock_vector_store.initialize = AsyncMock(return_value=None)
    mock_vector_store.upsert_chunks = AsyncMock()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = execute_background_ingestion(["http://test.com"])
    finally:
        loop.close()

    assert result == {"status": "INGESTION_COMPLETED", "processed_count": 1}
    mock_chunker.chunk.assert_called_once()
    mock_embed.embed_texts.assert_called_once_with(["Test markdown"])
    mock_vector_store.upsert_chunks.assert_called_once()


@patch("data_engineering_copilot.workers.tasks.AsyncWebCrawler")
@patch("data_engineering_copilot.workers.tasks.AsyncOllamaEmbeddings")
@patch("data_engineering_copilot.workers.tasks.DocumentChunker")
@patch("data_engineering_copilot.workers.tasks.AsyncQdrantVectorStore")
def test_execute_background_ingestion_failure(mock_qdrant, mock_chunker_class, mock_embedder, mock_crawler_class):
    mock_crawler = AsyncMock()
    mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
    mock_doc_magic = MagicMock()
    mock_doc_magic.success = False
    mock_crawler.arun.return_value = mock_doc_magic

    mock_vector_store = mock_qdrant.return_value
    mock_vector_store.initialize = AsyncMock(return_value=None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = execute_background_ingestion(["http://test.com"])
    finally:
        loop.close()

    assert result == {"status": "INGESTION_COMPLETED", "processed_count": 0}
