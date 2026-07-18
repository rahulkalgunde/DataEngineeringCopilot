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


@patch("data_engineering_copilot.workers.tasks.OllamaEmbeddings")
@patch("data_engineering_copilot.workers.tasks.DocumentChunker")
@patch("data_engineering_copilot.workers.tasks.QdrantVectorStore")
def test_execute_background_ingestion(mock_qdrant, mock_chunker_class, mock_embedder):
    mock_doc = MagicMock()
    mock_doc.success = True
    mock_doc.markdown = "Test markdown"
    mock_doc.title = "Test Title"
    mock_doc.url = "http://test.com"

    mock_chunker = mock_chunker_class.return_value
    mock_chunk = MagicMock()
    mock_chunk.text = "Test markdown"
    mock_chunker.chunk.return_value = [mock_chunk]

    mock_embed = mock_embedder.return_value
    mock_embed.embed_texts.return_value = [[0.1, 0.2]]

    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_until_complete.return_value = [mock_doc]

        result = execute_background_ingestion(["http://test.com"])

        assert result == {"status": "INGESTION_COMPLETED", "processed_count": 1}
        mock_chunker.chunk.assert_called_once()
        mock_embed.embed_texts.assert_called_once_with(["Test markdown"])
        mock_qdrant.return_value.upsert_chunks.assert_called_once()


@patch("data_engineering_copilot.workers.tasks.OllamaEmbeddings")
@patch("data_engineering_copilot.workers.tasks.DocumentChunker")
@patch("data_engineering_copilot.workers.tasks.QdrantVectorStore")
def test_execute_background_ingestion_failure(mock_qdrant, mock_chunker_class, mock_embedder):
    mock_doc = MagicMock()
    mock_doc.success = False

    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value.run_until_complete.return_value = [mock_doc]

        result = execute_background_ingestion(["http://test.com"])

        assert result == {"status": "INGESTION_COMPLETED", "processed_count": 0}
