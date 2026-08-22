from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.domain.models import DocumentChunk, IngestionEvent, ParsedDocument, RawDocument
from data_engineering_copilot.services.chunker import DocumentChunker


@pytest.fixture
def mock_settings():
    from tests.conftest import make_settings

    return make_settings(
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
    frontier = AsyncMock()
    frontier.stats = AsyncMock(return_value={"DISCOVERED": 1})
    frontier.all_urls = AsyncMock(return_value=[])
    frontier.reactivate_missing = AsyncMock(return_value=0)
    frontier.mark_processed = AsyncMock()
    frontier.mark_failed = AsyncMock()
    frontier.mark_skipped = AsyncMock()
    frontier.close = AsyncMock()
    c.frontier = frontier
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
    e.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    return e


@pytest.fixture
def mock_vector_store():
    v = MagicMock()
    v.upsert_chunks = AsyncMock()
    v.get_content_hash_for_url = AsyncMock(return_value=None)
    v.delete_by_url = AsyncMock()
    v.count_urls = AsyncMock(return_value=0)
    return v


@pytest.fixture
def _thread_executors():
    """Provide ThreadPoolExecutor instances for parse/chunk to avoid fork deadlocks in tests."""
    pe = ThreadPoolExecutor(max_workers=2)
    ce = ThreadPoolExecutor(max_workers=2)
    yield pe, ce
    pe.shutdown(wait=False)
    ce.shutdown(wait=False)


@pytest.fixture
def service(
    mock_settings, mock_crawler, mock_parser, mock_chunker, mock_embeddings, mock_vector_store, _thread_executors
):
    from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

    parse_exec, chunk_exec = _thread_executors
    svc = AsyncIngestionService(
        settings=mock_settings,
        crawler=mock_crawler,
        parser=mock_parser,
        chunker=mock_chunker,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        parse_executor=parse_exec,
        chunk_executor=chunk_exec,
    )
    yield svc
    svc.stop()


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


async def _picklable_chunk(parsed):
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


def _make_svc(mock_settings, mock_crawler, **kwargs):
    """Helper to create AsyncIngestionService with ThreadPoolExecutor defaults."""
    from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

    parse_exec = ThreadPoolExecutor(max_workers=2)
    chunk_exec = ThreadPoolExecutor(max_workers=2)
    svc = AsyncIngestionService(
        settings=mock_settings,
        crawler=mock_crawler,
        parse_executor=parse_exec,
        chunk_executor=chunk_exec,
        **kwargs,
    )
    return svc


class TestAsyncIngestionServiceInit:
    def test_init_accepts_components(
        self,
        mock_settings,
        mock_crawler,
        mock_parser,
        mock_chunker,
        mock_embeddings,
        mock_vector_store,
        _thread_executors,
    ):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        parse_exec, chunk_exec = _thread_executors
        s = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=mock_parser,
            chunker=mock_chunker,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
            parse_executor=parse_exec,
            chunk_executor=chunk_exec,
        )
        assert s.settings is mock_settings
        assert s.crawler is mock_crawler
        s.stop()


class _AsyncListIterator:
    """Async iterator wrapper for a plain list, usable with ``async for``."""

    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class TestAsyncIngestionServiceIngest:
    @pytest.mark.asyncio
    async def test_single_page_indexed(self, mock_settings, mock_crawler):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        # Mirrors the real DocumentChunker contract: extract_sentences exists but
        # returns None ("not supported") -> _process_raw falls through to chunk().
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        total = await service.ingest()

        assert total == 1
        mock_crawler.crawl.assert_called()

    @pytest.mark.asyncio
    async def test_worker_error_is_isolated(self, mock_settings, mock_crawler):
        """A single page's processing error must not fail the whole run."""
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = MagicMock(side_effect=RuntimeError("boom"))
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        # The run completes without raising; the page is marked FAILED.
        total = await asyncio.wait_for(service.ingest(), timeout=10)
        assert total == 0
        mock_crawler.frontier.mark_failed.assert_awaited()

    @pytest.mark.asyncio
    async def test_on_event_callback(self, mock_settings, mock_crawler):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        events = []
        total = await service.ingest(on_event=events.append)

        assert total == 1
        assert len(events) > 0
        assert events[0].event_type == "source_start"
        assert any(e.event_type == "page_indexed" for e in events)

    @pytest.mark.asyncio
    async def test_skips_none_parsed(self, mock_settings, mock_crawler):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse_skip
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock()
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock()
        vector_store_mock.delete_by_url = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw, raw])

        total = await service.ingest()
        assert total == 0

    @pytest.mark.asyncio
    async def test_content_hash_dedup(self, mock_settings, mock_crawler, mock_vector_store):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=mock_vector_store,
        )

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        mock_vector_store.get_content_hash_for_url.return_value = "sha256:somehash"

        original_hash = AsyncIngestionService._compute_content_hash
        AsyncIngestionService._compute_content_hash = cast(Any, staticmethod(lambda text: "sha256:somehash"))

        try:
            total = await service.ingest()
            assert total == 0  # skipped by dedup
        finally:
            AsyncIngestionService._compute_content_hash = original_hash

    @pytest.mark.asyncio
    async def test_respects_source_names(self, mock_settings, mock_crawler):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.upsert_chunks = AsyncMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.delete_by_url = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        total = await service.ingest(source_names=["test"])
        assert total > 0


class TestAsyncIngestionServiceWorkerPool:
    def test_isolated_executors_created(self, service):
        from concurrent.futures import ThreadPoolExecutor

        assert hasattr(service, "_parse_executor")
        assert hasattr(service, "_chunk_executor")
        assert isinstance(service._parse_executor, ThreadPoolExecutor)
        assert isinstance(service._chunk_executor, ThreadPoolExecutor)

    def test_processing_concurrency_from_settings(self, service):
        assert service._processing_concurrency == 2

    @pytest.mark.asyncio
    async def test_multi_page_batch_flush(self, mock_settings, mock_crawler, mock_embeddings, mock_vector_store):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
        )

        urls = [f"https://example.com/{i}" for i in range(3)]
        raws = [_make_raw(url=url) for url in urls]
        mock_crawler.crawl.return_value = _AsyncListIterator(raws)

        total = await service.ingest()

        assert total == 3
        assert mock_embeddings.embed_texts.call_count >= 2
        assert mock_vector_store.upsert_chunks.call_count >= 2


class TestDedupAuthority:
    """Qdrant is the source of truth for content-hash dedup."""

    @pytest.mark.asyncio
    async def test_stored_hash_consults_vector_store_only(self, service, mock_vector_store):
        """A stale/divergent Redis hash must never trigger a dedup skip."""
        fake_redis = MagicMock()
        registry = MagicMock()
        registry.get_html_hash = AsyncMock(return_value="sha256:stale")
        service._redis_client = fake_redis
        service._url_registries["test"] = registry
        mock_vector_store.get_content_hash_for_url = AsyncMock(return_value=None)

        result = await service._get_stored_content_hash("https://example.com/doc", "test")
        assert result is None
        registry.get_html_hash.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_page_indexed_when_redis_divergent_and_qdrant_empty(
        self, mock_settings, mock_crawler, mock_vector_store
    ):
        """Regression: after `dec reset-qdrant`, pages previously recorded in
        Redis must still be re-indexed because Qdrant no longer has them."""
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=mock_vector_store,
        )
        fake_redis = MagicMock()
        registry = MagicMock()
        registry.get_html_hash = AsyncMock(return_value="sha256:stale")
        registry.set_html_hash = AsyncMock()
        service._redis_client = fake_redis
        service._url_registries["test"] = registry

        mock_vector_store.get_content_hash_for_url = AsyncMock(return_value=None)
        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        total = await service.ingest()
        assert total == 1
        mock_vector_store.upsert_chunks.assert_awaited()


class TestFrontierStateTransitions:
    """The ingestion worker owns PROCESSED/SKIPPED/FAILED transitions."""

    @pytest.mark.asyncio
    async def test_marks_processed_after_successful_flush(self, mock_settings, mock_crawler):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )
        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        await service.ingest()
        mock_crawler.frontier.mark_processed.assert_awaited()

    @pytest.mark.asyncio
    async def test_marks_failed_when_flush_errors(self, mock_settings, mock_crawler):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock(side_effect=RuntimeError("store down"))

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )
        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        total = await service.ingest()
        assert total == 0
        mock_crawler.frontier.mark_failed.assert_awaited()
        mock_crawler.frontier.mark_processed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_marks_skipped_for_no_content(self, mock_settings, mock_crawler):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse_skip
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock()
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock()
        vector_store_mock.delete_by_url = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )
        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        total = await service.ingest()
        assert total == 0
        mock_crawler.frontier.mark_skipped.assert_awaited()
        mock_crawler.frontier.mark_processed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_marks_processed_for_duplicate(self, mock_settings, mock_crawler):
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock()
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value="sha256:somehash")

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )
        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        original_hash = AsyncIngestionService._compute_content_hash
        AsyncIngestionService._compute_content_hash = cast(Any, staticmethod(lambda text: "sha256:somehash"))
        try:
            total = await service.ingest()
            assert total == 0
            mock_crawler.frontier.mark_processed.assert_awaited()
            vector_store_mock.upsert_chunks.assert_not_called()
        finally:
            AsyncIngestionService._compute_content_hash = original_hash


_REAL_CONTENT_HTML = """
<html><body><main>
<h1>Real content page</h1>
<p>This page has plenty of real readable content for the parser to extract.
It is long enough to pass the minimum word threshold used by the markdown
converter, so the parser should return a valid document instead of skipping.
Adding more words here ensures we stay comfortably above the cutoff and the
chunking step has something meaningful to work with.</p>
</main></body></html>
"""


class TestRealChunkerThroughProcessRaw:
    """Regression tests: real chunkers must index content through _process_raw.

    Guards the regression from commit d7e595d where DocumentChunker gained an
    ``extract_sentences`` stub returning ``None`` (meaning "not supported") but
    ``_process_raw`` treated ``None`` as "no content", silently skipping every
    page under the default ``sentence_preserving`` strategy.
    """

    @staticmethod
    def _real_parser():
        from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser

        return MarkdownParser()

    async def _process(self, service, raw):
        loop = asyncio.get_running_loop()
        return await service._process_raw(
            loop,
            raw,
            None,
            lambda *a, **kw: IngestionEvent(event_type="test", source_name="", message=""),
        )

    @pytest.mark.asyncio
    async def test_document_chunker_indexes_content(self, mock_settings, mock_crawler):
        """Default DocumentChunker (extract_sentences -> None) must fall through
        to chunk() and index content, not skip it."""
        from data_engineering_copilot.services.chunker import DocumentChunker

        chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1] * 2048])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=self._real_parser(),
            chunker=chunker,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )
        raw = RawDocument(
            source_name="test",
            url="https://example.com/real",
            html=_REAL_CONTENT_HTML,
        )
        result = await self._process(service, raw)
        assert result.disposition == "indexed"
        assert result.parsed is not None
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_header_aware_chunker_indexes_content(self, mock_settings, mock_crawler):
        """HeaderAwareChunker (extract_sentences -> None) must index content."""
        from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker

        chunker = HeaderAwareChunker(chunk_size_words=75, overlap_words=15)
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1] * 2048])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=self._real_parser(),
            chunker=chunker,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )
        raw = RawDocument(
            source_name="test",
            url="https://example.com/real",
            html=_REAL_CONTENT_HTML,
        )
        result = await self._process(service, raw)
        assert result.disposition == "indexed"
        assert result.parsed is not None
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_empty_sentences_skips_with_log_event(self, mock_settings, mock_crawler):
        """A chunker whose extract_sentences returns [] must skip AND emit a
        page_skipped event (observable, not silent)."""
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: []
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock()
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)

        events: list[IngestionEvent] = []

        def on_event(event: IngestionEvent) -> None:
            events.append(event)

        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        service = AsyncIngestionService(
            settings=mock_settings,
            crawler=mock_crawler,
            parser=MagicMock(parse=_picklable_parse),
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
            parse_executor=ThreadPoolExecutor(max_workers=2),
            chunk_executor=ThreadPoolExecutor(max_workers=2),
        )
        raw = _make_raw(html=_REAL_CONTENT_HTML)
        loop = asyncio.get_running_loop()
        result = await service._process_raw(
            loop,
            raw,
            on_event,
            lambda event_type, *a, **kw: IngestionEvent(event_type=event_type, **kw),
        )
        assert result.disposition == "no_content"
        assert any(e.event_type == "page_skipped" for e in events)
        service.stop()

    @pytest.mark.asyncio
    async def test_semantic_chunker_list_branch_indexes_content(self, mock_settings, mock_crawler):
        """Real SemanticChunker must exercise the list branch of _process_raw:
        extract_sentences() -> embed_texts(sentences) -> chunk(parsed, embeddings).

        This pins the else-branch that had zero coverage while the None/[] bug
        was live: SemanticChunker.extract_sentences returns a real list (never
        None for valid text), so the page must be indexed with chunks.
        """
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        class _Embedder:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                dim = 8
                return [[0.1] * dim for _ in texts]

        chunker = SemanticChunker(
            chunk_size_words=50,
            overlap_words=5,
            embedding_model=_Embedder(),
            min_chunk_words=5,
            min_semantic_similarity=0.1,
        )
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock()

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=self._real_parser(),
            chunker=chunker,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )
        raw = RawDocument(
            source_name="test",
            url="https://example.com/real",
            html=_REAL_CONTENT_HTML,
        )
        result = await self._process(service, raw)
        assert result.disposition == "indexed"
        assert result.parsed is not None
        assert len(result.chunks) >= 1
        # The list branch must have called embed_texts with the extracted sentences.
        embeddings_mock.embed_texts.assert_awaited_once()


class TestReactivation:
    @pytest.mark.asyncio
    async def test_reactivates_missing_urls_after_full_crawl(self, mock_settings, mock_crawler, mock_vector_store):
        """After a fully-drained crawl, URLs missing from Qdrant (reset) are
        re-discovered so the next run re-indexes them."""

        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=mock_vector_store,
        )
        # Simulate a fully-drained crawl: frontier has no DISCOVERED records.
        mock_crawler.frontier.stats = AsyncMock(return_value={"DISCOVERED": 0, "PROCESSED": 1})
        mock_crawler.frontier.reactivate_missing = AsyncMock(return_value=1)

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])
        await service.ingest()

        mock_crawler.frontier.reactivate_missing.assert_awaited()
        await_args = mock_crawler.frontier.reactivate_missing.await_args
        assert await_args is not None
        call_kwargs = await_args.kwargs
        assert call_kwargs.get("max_attempts") == mock_settings.frontier_max_attempts

    @pytest.mark.asyncio
    async def test_no_reactivation_on_partial_crawl(self, mock_settings, mock_crawler):
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)
        vector_store_mock.upsert_chunks = AsyncMock()
        # Non-empty index: the start-of-run reset check must not fire.
        vector_store_mock.count_urls = AsyncMock(return_value=1)

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
        )
        mock_crawler.frontier.stats = AsyncMock(return_value={"DISCOVERED": 10, "PROCESSED": 1})
        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])

        await service.ingest()
        mock_crawler.frontier.reactivate_missing.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reactivates_at_start_when_index_empty_despite_partial_crawl(
        self, mock_settings, mock_crawler, mock_vector_store
    ):
        """Regression: after `dec reset-qdrant` (empty index + PROCESSED history),
        a page-capped run still re-discovers missing URLs at the start, even when
        DISCOVERED records remain (which suppress the post-crawl reactivation)."""
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=mock_vector_store,
        )
        mock_vector_store.count_urls = AsyncMock(return_value=0)
        mock_crawler.frontier.stats = AsyncMock(return_value={"DISCOVERED": 10, "PROCESSED": 5})
        mock_crawler.frontier.reactivate_missing = AsyncMock(return_value=5)

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])
        await service.ingest()

        mock_crawler.frontier.reactivate_missing.assert_awaited()
        call_args = mock_crawler.frontier.reactivate_missing.await_args
        assert call_args is not None
        assert call_args.args[0] == "test"
        assert call_args.args[1] == set()
        assert call_args.kwargs.get("max_attempts") == mock_settings.frontier_max_attempts

    @pytest.mark.asyncio
    async def test_no_start_reactivation_when_index_nonempty(self, mock_settings, mock_crawler, mock_vector_store):
        """A non-empty index means content is present, so no start-of-run reset."""
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=mock_vector_store,
        )
        mock_vector_store.count_urls = AsyncMock(return_value=1)
        mock_crawler.frontier.stats = AsyncMock(return_value={"DISCOVERED": 5, "PROCESSED": 5})
        mock_crawler.frontier.reactivate_missing = AsyncMock(return_value=0)

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])
        await service.ingest()

        mock_crawler.frontier.reactivate_missing.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reactivation_failure_is_nonfatal(self, mock_settings, mock_crawler, mock_vector_store):
        """A transient error in the start-of-run reset check must not fail ingestion."""
        parser_mock = MagicMock()
        parser_mock.parse = _picklable_parse
        chunker_mock = MagicMock(spec=DocumentChunker)
        chunker_mock.extract_sentences = lambda text: None
        chunker_mock.chunk = _picklable_chunk
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=parser_mock,
            chunker=chunker_mock,
            embeddings=embeddings_mock,
            vector_store=mock_vector_store,
        )
        mock_vector_store.count_urls = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        raw = _make_raw()
        mock_crawler.crawl.return_value = _AsyncListIterator([raw])
        await service.ingest()

        mock_crawler.frontier.reactivate_missing.assert_not_awaited()


class _CountingChunker:
    """Delegating wrapper counting ``chunk()`` calls on a real strategy."""

    def __init__(self, inner):
        self.inner = inner
        self.chunk_calls = 0

    async def chunk(self, document, precomputed_embeddings=None):
        self.chunk_calls += 1
        self.last_embeddings = precomputed_embeddings
        return await self.inner.chunk(document, precomputed_embeddings)

    def extract_sentences(self, text):
        return self.inner.extract_sentences(text)


class _CountingRouter:
    """Delegating wrapper counting ``route()`` calls on a real router."""

    def __init__(self, inner):
        self.inner = inner
        self.route_calls = 0

    def route(self, document):
        self.route_calls += 1
        return self.inner.route(document)


class TestRouterThroughProcessRaw:
    """Behavioral tests: real router + real chunker objects through _process_raw."""

    @staticmethod
    def _parser_for(parsed):
        parser_mock = MagicMock()
        parser_mock.parse = MagicMock(return_value=parsed)
        return parser_mock

    async def _process(self, service, raw):
        loop = asyncio.get_running_loop()
        return await service._process_raw(
            loop,
            raw,
            None,
            lambda *a, **kw: IngestionEvent(event_type="test", source_name="", message=""),
        )

    @pytest.mark.asyncio
    async def test_json_doc_routes_to_structured_with_one_route_and_one_chunk_call(self, mock_settings, mock_crawler):
        from data_engineering_copilot.services.chunker import DocumentChunker
        from data_engineering_copilot.services.chunker_router import ChunkerRouter
        from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
        from data_engineering_copilot.services.structured_data_chunker import StructuredDataChunker

        parsed = ParsedDocument(
            source_name="test",
            title="Config",
            url="https://example.com/config.json",
            text='{"name": "Spark", "version": "4.0"}',
            doc_type="json",
        )
        structured = _CountingChunker(StructuredDataChunker())
        router = ChunkerRouter(
            generic_strategy=DocumentChunker(),
            structured_strategy=structured,
            code_strategy=DocumentChunker(),
            guide_strategy=HeaderAwareChunker(),
        )
        counting_router = _CountingRouter(router)

        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1] * 2048])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=self._parser_for(parsed),
            chunker=DocumentChunker(),
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
            chunker_router=counting_router,
        )
        raw = RawDocument(source_name="test", url="https://example.com/config.json", html="")
        result = await self._process(service, raw)

        assert counting_router.route_calls == 1
        assert structured.chunk_calls == 1
        assert structured.last_embeddings is None, "structured strategy must not trigger embedding"
        embeddings_mock.embed_texts.assert_not_awaited()
        assert result.disposition == "indexed"
        assert result.chunks
        assert all(c.chunk_type == "structured" for c in result.chunks)
        assert all(c.section_header == "$" for c in result.chunks)

    @pytest.mark.asyncio
    async def test_routed_chunks_preserve_document_metadata(self, mock_settings, mock_crawler):
        from data_engineering_copilot.services.chunker import DocumentChunker
        from data_engineering_copilot.services.chunker_router import ChunkerRouter
        from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
        from data_engineering_copilot.services.structured_data_chunker import StructuredDataChunker

        parsed = ParsedDocument(
            source_name="test",
            title="Config",
            url="https://example.com/config.json",
            text='{"name": "Spark"}',
            doc_type="json",
        )
        router = ChunkerRouter(
            generic_strategy=DocumentChunker(),
            structured_strategy=StructuredDataChunker(),
            code_strategy=DocumentChunker(),
            guide_strategy=HeaderAwareChunker(),
        )
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock()
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=self._parser_for(parsed),
            chunker=DocumentChunker(),
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
            chunker_router=router,
        )
        raw = RawDocument(source_name="test", url="https://example.com/config.json", html="")
        result = await self._process(service, raw)

        assert result.disposition == "indexed"
        for chunk in result.chunks:
            assert chunk.source_name == "test"
            assert chunk.title == "Config"
            assert chunk.url == "https://example.com/config.json"
            assert chunk.doc_type == "json"

    @pytest.mark.asyncio
    async def test_generic_route_embeds_sentences_once(self, mock_settings, mock_crawler):
        """A sentence-supporting strategy routed through the router embeds exactly once."""
        from data_engineering_copilot.services.chunker import DocumentChunker
        from data_engineering_copilot.services.chunker_router import ChunkerRouter

        class _SentenceChunker:
            def __init__(self, inner):
                self.inner = inner
                self.chunk_calls = 0

            async def chunk(self, document, precomputed_embeddings=None):
                self.chunk_calls += 1
                return await self.inner.chunk(document, precomputed_embeddings)

            def extract_sentences(self, text):
                return ["sentence one", "sentence two"]

        parsed = ParsedDocument(
            source_name="test",
            title="Doc",
            url="https://example.com/plain",
            text="sentence one. sentence two.",
        )
        generic = _SentenceChunker(DocumentChunker(chunk_size_chars=500))
        router = ChunkerRouter(generic_strategy=generic)
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock(return_value=[[0.1] * 2048, [0.2] * 2048])
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=self._parser_for(parsed),
            chunker=generic,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
            chunker_router=router,
        )
        raw = RawDocument(source_name="test", url="https://example.com/plain", html="")
        result = await self._process(service, raw)

        embeddings_mock.embed_texts.assert_awaited_once()
        embeddings_mock.embed_texts.assert_awaited_with(["sentence one", "sentence two"])
        assert generic.chunk_calls == 1
        assert result.disposition == "indexed"
        assert result.chunks

    @pytest.mark.asyncio
    async def test_semantic_route_empty_sentences_skips_as_no_content(self, mock_settings, mock_crawler):
        from data_engineering_copilot.services.chunker_router import ChunkerRouter

        class _EmptySentenceChunker:
            def __init__(self):
                self.chunk_calls = 0

            async def chunk(self, document, precomputed_embeddings=None):
                self.chunk_calls += 1
                return []

            def extract_sentences(self, text):
                return []

        parsed = ParsedDocument(
            source_name="test",
            title="Doc",
            url="https://example.com/plain",
            text="",
        )
        empty = _EmptySentenceChunker()
        router = ChunkerRouter(generic_strategy=empty)
        embeddings_mock = MagicMock()
        embeddings_mock.embed_texts = AsyncMock()
        vector_store_mock = MagicMock()
        vector_store_mock.get_content_hash_for_url = AsyncMock(return_value=None)

        service = _make_svc(
            mock_settings,
            mock_crawler,
            parser=self._parser_for(parsed),
            chunker=empty,
            embeddings=embeddings_mock,
            vector_store=vector_store_mock,
            chunker_router=router,
        )
        raw = RawDocument(source_name="test", url="https://example.com/plain", html="")
        result = await self._process(service, raw)

        assert result.disposition == "no_content"
        assert empty.chunk_calls == 0
        embeddings_mock.embed_texts.assert_not_awaited()
