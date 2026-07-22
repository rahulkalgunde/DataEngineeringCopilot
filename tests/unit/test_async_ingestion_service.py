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
        parse_concurrency=2,
        chunk_concurrency=2,
        embed_concurrency=2,
        store_concurrency=1,
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


# Module-level picklable functions for ProcessPoolExecutor tests
def _picklable_parse(raw_doc):
    return ParsedDocument(
        source_name=raw_doc.source_name,
        title="Parsed",
        url=raw_doc.url,
        text="parsed content",
    )


def _picklable_chunk(parsed):
    return [
        DocumentChunk(
            chunk_id=f"c_{parsed.url}",
            source_name=parsed.source_name,
            title=parsed.title,
            url=parsed.url,
            text=parsed.text,
        )
    ]


def _picklable_parse_skip(raw_doc):
    """Module-level picklable function that always returns None (skip)."""
    return None


class TestAsyncIngestionServiceInit:
    def test_init_accepts_components(
        self, mock_settings, mock_crawler, mock_parser, mock_chunker, mock_embeddings, mock_vector_store
    ):
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
    async def test_single_page_indexed(self, mock_settings, mock_crawler):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock()
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = MagicMock(return_value=None)

        service = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw])
        mock_crawler.crawl.return_value = async_iter

        total = await service.ingest()

        assert total == 1
        mock_crawler.crawl.assert_called()

    @pytest.mark.asyncio
    async def test_on_event_callback(self, mock_settings, mock_crawler):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock()
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = MagicMock(return_value=None)

        service = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw])
        mock_crawler.crawl.return_value = async_iter

        events = []
        total = await service.ingest(on_event=events.append)

        assert total == 1
        assert len(events) > 0
        assert events[0].event_type == "source_start"
        assert any(e.event_type == "page_indexed" for e in events)

    @pytest.mark.asyncio
    async def test_skips_none_parsed(self, mock_settings, mock_crawler):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse_skip
        chunker_mock = MagicMock()
        embeddings_mock = MagicMock()
        vector_store_mock = MagicMock()

        service = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw, raw])
        mock_crawler.crawl.return_value = async_iter

        total = await service.ingest()
        assert total == 0

    @pytest.mark.asyncio
    async def test_content_hash_dedup(self, mock_settings, mock_crawler, mock_vector_store):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock()
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()

        service = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=mock_vector_store,
        )

        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw])
        mock_crawler.crawl.return_value = async_iter

        mock_vector_store.get_content_hash_for_url.return_value = "sha256:somehash"

        original_hash = AsyncIngestionService._compute_content_hash
        AsyncIngestionService._compute_content_hash = staticmethod(lambda text: "sha256:somehash")

        try:
            total = await service.ingest()
            assert total == 0  # skipped by dedup
        finally:
            AsyncIngestionService._compute_content_hash = original_hash

    @pytest.mark.asyncio
    async def test_respects_source_names(self, mock_settings, mock_crawler):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock()
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = MagicMock(return_value=None)

        service = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter([raw])
        mock_crawler.crawl.return_value = async_iter

        total = await service.ingest(source_names=["test"])
        assert total > 0


class TestAsyncIngestionServiceWorkerPool:
    def test_isolated_executors_created(self, service):
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        assert hasattr(service, "_parse_executor")
        assert hasattr(service, "_chunk_executor")
        assert hasattr(service, "_embed_executor")
        assert hasattr(service, "_store_executor")
        assert isinstance(service._parse_executor, ProcessPoolExecutor)
        assert isinstance(service._chunk_executor, ProcessPoolExecutor)
        assert isinstance(service._embed_executor, ThreadPoolExecutor)
        assert isinstance(service._store_executor, ThreadPoolExecutor)

    def test_processing_concurrency_from_settings(self, service):
        assert service._processing_concurrency == 2

    @pytest.mark.asyncio
    async def test_multi_page_batch_flush(self, mock_settings, mock_crawler, mock_embeddings, mock_vector_store):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        # Use picklable functions for ProcessPoolExecutor
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock()
        chunker_mock.chunk = _picklable_chunk

        service = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
        )

        urls = [f"https://example.com/{i}" for i in range(3)]
        raws = [_make_raw(url=url) for url in urls]
        async_iter = MagicMock()
        async_iter.__aiter__.return_value = iter(raws)
        mock_crawler.crawl.return_value = async_iter

        total = await service.ingest()

        assert total == 3
        assert mock_embeddings.embed_texts.call_count >= 2
        assert mock_vector_store.upsert_chunks.call_count >= 2
