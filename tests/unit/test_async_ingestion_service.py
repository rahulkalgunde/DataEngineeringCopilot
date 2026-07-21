from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument, RawDocument


@pytest.fixture
def mock_settings():
    return AppSettings(
        max_pages_per_source=10,
        ingestion_batch_chunk_size=2,
        processing_concurrency=2,
        embedding_batch_size=32,
        sources=(
            DocumentationSource(
                name="test",
                start_urls=("https://example.com",),
                allowed_domains=("example.com",),
                url_prefixes=("https://example.com/",),
            ),
        ),
    )


@pytest.fixture
def mock_crawler():
    c = MagicMock()
    c.crawl = MagicMock()
    c.frontier = MagicMock()
    c.frontier.close = AsyncMock()
    return c


@pytest.fixture
def mock_parser():
    p = MagicMock()
    p.parse = MagicMock()
    return p


@pytest.fixture
def mock_chunker():
    c = MagicMock()
    c.chunk = MagicMock()
    return c


@pytest.fixture
def mock_embeddings():
    e = MagicMock()
    e.embed_texts = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    return e


@pytest.fixture
def mock_vector_store():
    v = MagicMock()
    v.upsert_chunks = MagicMock()
    v.get_content_hash_for_url = MagicMock(return_value=None)
    return v


@pytest.fixture
def service(mock_settings, mock_crawler, mock_parser, mock_chunker, mock_embeddings, mock_vector_store):
    from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

    return AsyncIngestionService(
        settings=mock_settings,
        crawler=mock_crawler,
        parser=mock_parser,
        chunker=mock_chunker,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )


def _make_raw(source_name="test", url="https://example.com/doc", html="<p>hello</p>"):
    return RawDocument(source_name=source_name, url=url, html=html)


class TestAsyncIngestionServiceInit:
    def test_init_accepts_components(self, mock_settings, mock_crawler, mock_parser, mock_chunker, mock_embeddings, mock_vector_store):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        s = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=mock_parser,
            chunker=mock_chunker,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
        )
        assert s.settings is mock_settings
        assert s.crawler is mock_crawler


class TestAsyncIngestionServiceIngest:
    @pytest.mark.asyncio
    async def test_single_page_indexed(self, service, mock_crawler, mock_parser, mock_chunker):
        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw])
        mock_crawler.crawl.return_value = async_iter
        mock_parser.parse.return_value = ParsedDocument(
            source_name="test", url="https://example.com/doc", title="Test", text="hello world"
        )
        mock_chunker.chunk.return_value = [
            DocumentChunk(chunk_id="c1", source_name="test", title="Test", url="https://example.com/doc", text="hello world"),
        ]

        total = await service.ingest()

        assert total == 1
        mock_crawler.crawl.assert_called()
        mock_parser.parse.assert_called_once_with(raw)
        mock_chunker.chunk.assert_called_once()
        assert service.embeddings.embed_texts.called

    @pytest.mark.asyncio
    async def test_on_event_callback(self, service, mock_crawler, mock_parser, mock_chunker):
        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw])
        mock_crawler.crawl.return_value = async_iter
        mock_parser.parse.return_value = ParsedDocument(
            source_name="test", url="https://example.com/doc", title="Test", text="hello world"
        )
        mock_chunker.chunk.return_value = [
            DocumentChunk(chunk_id="c1", source_name="test", title="Test", url="https://example.com/doc", text="hello world"),
        ]

        events = []
        total = await service.ingest(on_event=events.append)

        assert total == 1
        assert len(events) > 0
        assert events[0].event_type == "source_start"
        assert any(e.event_type == "page_indexed" for e in events)

    @pytest.mark.asyncio
    async def test_skips_none_parsed(self, service, mock_crawler, mock_parser):
        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw, raw])
        mock_crawler.crawl.return_value = async_iter
        mock_parser.parse.return_value = None  # skip

        total = await service.ingest()
        assert total == 0

    @pytest.mark.asyncio
    async def test_content_hash_dedup(self, service, mock_crawler, mock_parser, mock_chunker, mock_vector_store):
        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw])
        mock_crawler.crawl.return_value = async_iter
        mock_parser.parse.return_value = ParsedDocument(
            source_name="test", url="https://example.com/doc", title="Test", text="same text"
        )
        mock_vector_store.get_content_hash_for_url.return_value = (
            "sha256:somehash"
        )

        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService
        original_hash = AsyncIngestionService._compute_content_hash
        AsyncIngestionService._compute_content_hash = staticmethod(lambda text: "sha256:somehash")

        try:
            total = await service.ingest()
            assert total == 0  # skipped by dedup
        finally:
            AsyncIngestionService._compute_content_hash = original_hash

    @pytest.mark.asyncio
    async def test_respects_source_names(self, service, mock_crawler, mock_parser, mock_chunker):
        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw])
        mock_crawler.crawl.return_value = async_iter
        mock_parser.parse.return_value = ParsedDocument(
            source_name="test", url="https://example.com/doc", title="Test", text="hello"
        )
        mock_chunker.chunk.return_value = [
            DocumentChunk(chunk_id="c1", source_name="test", title="Test", url="https://example.com/doc", text="hello"),
        ]

        total = await service.ingest(source_names=["test"])
        assert total > 0


class TestAsyncIngestionServiceWorkerPool:
    @pytest.mark.asyncio
    async def test_executor_created_on_init(self, service):
        from concurrent.futures import ThreadPoolExecutor

        assert hasattr(service, "_executor")
        assert isinstance(service._executor, ThreadPoolExecutor)

    @pytest.mark.asyncio
    async def test_processing_concurrency_from_settings(self, service):
        assert service._processing_concurrency == 2

    @pytest.mark.asyncio
    async def test_multi_page_batch_flush(self, service, mock_crawler, mock_parser, mock_chunker):
        urls = [f"https://example.com/{i}" for i in range(3)]
        raws = [_make_raw(url=url) for url in urls]
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter(raws)
        mock_crawler.crawl.return_value = async_iter
        mock_parser.parse.side_effect = [
            ParsedDocument(source_name="test", url=url, title=f"Doc{i}", text=f"text{i}")
            for i, url in enumerate(urls)
        ]

        def _chunk(parsed):
            return [
                DocumentChunk(
                    chunk_id=f"c_{parsed.url}",
                    source_name="test",
                    title=parsed.title,
                    url=parsed.url,
                    text=parsed.text,
                )
            ]

        mock_chunker.chunk.side_effect = _chunk

        total = await service.ingest()

        assert total == 3
        assert service.embeddings.embed_texts.call_count >= 2
        assert service.vector_store.upsert_chunks.call_count >= 2
