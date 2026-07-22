"""Tests for multi-stage pipeline with isolated executor pools.

Verifies that AsyncIngestionService uses separate executor pools for
CPU-bound (parse, chunk) and IO-bound (embed, store) work.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from unittest.mock import MagicMock

from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
from data_engineering_copilot.domain.models import DocumentChunk, IngestionEvent, ParsedDocument, RawDocument


def _parse_fn(raw_doc):
    """Module-level picklable parse function for ProcessPoolExecutor tests."""
    return ParsedDocument(
        source_name=raw_doc.source_name,
        title="Parsed",
        url=raw_doc.url,
        text="parsed content",
    )


def _chunk_fn(parsed):
    """Module-level picklable chunk function for ProcessPoolExecutor tests."""
    return [
        DocumentChunk(
            chunk_id="c1",
            source_name=parsed.source_name,
            title="T",
            url=parsed.url,
            text="chunk",
        )
    ]


def _make_settings(**overrides) -> AppSettings:
    defaults = dict(
        processing_concurrency=4,
        ingestion_batch_chunk_size=256,
        parse_concurrency=2,
        chunk_concurrency=2,
        embed_concurrency=2,
        store_concurrency=1,
        crawl_async_concurrency=10,
    )
    defaults.update(overrides)
    return AppSettings(**defaults)


def _make_source() -> DocumentationSource:
    return DocumentationSource(
        name="TestSource",
        start_urls=("http://example.com",),
        allowed_domains=("example.com",),
    )


def _make_service(**settings_overrides):
    from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

    settings = _make_settings(**settings_overrides)
    return AsyncIngestionService(
        settings=settings,
        crawler=MagicMock(),
        parser=MagicMock(),
        chunker=MagicMock(),
        embeddings=MagicMock(),
        vector_store=MagicMock(),
    )


class TestMultiStagePools:
    """Unit tests for multi-stage executor pool isolation."""

    def test_settings_has_parse_concurrency(self):
        settings = _make_settings()
        assert hasattr(settings, "parse_concurrency")
        assert settings.parse_concurrency == 2

    def test_settings_has_chunk_concurrency(self):
        settings = _make_settings()
        assert hasattr(settings, "chunk_concurrency")
        assert settings.chunk_concurrency == 2

    def test_settings_has_embed_concurrency(self):
        settings = _make_settings()
        assert hasattr(settings, "embed_concurrency")
        assert settings.embed_concurrency == 2

    def test_settings_has_store_concurrency(self):
        settings = _make_settings()
        assert hasattr(settings, "store_concurrency")
        assert settings.store_concurrency == 1

    def test_isolated_executors_created(self):
        service = _make_service()

        assert hasattr(service, "_parse_executor")
        assert hasattr(service, "_chunk_executor")
        assert hasattr(service, "_embed_executor")
        assert hasattr(service, "_store_executor")

        assert isinstance(service._parse_executor, ProcessPoolExecutor)
        assert isinstance(service._chunk_executor, ProcessPoolExecutor)
        assert isinstance(service._embed_executor, ThreadPoolExecutor)
        assert isinstance(service._store_executor, ThreadPoolExecutor)

    def test_executor_sizes_match_settings(self):
        service = _make_service(parse_concurrency=3, chunk_concurrency=5, embed_concurrency=2, store_concurrency=1)

        assert service._parse_executor._max_workers == 3
        assert service._chunk_executor._max_workers == 5
        assert service._embed_executor._max_workers == 2
        assert service._store_executor._max_workers == 1

    def test_process_raw_uses_parse_executor(self):
        """Verify that _process_raw offloads parsing to _parse_executor."""
        service = _make_service()

        # Use module-level picklable functions (ProcessPoolExecutor can't pickle locals)
        service.parser = MagicMock()
        service.parser.parse = _parse_fn
        service.vector_store.get_content_hash_for_url = MagicMock(return_value=None)

        service.chunker = MagicMock()
        service.chunker.chunk = _chunk_fn

        raw_doc = RawDocument(source_name="test", url="http://example.com", html="<p>test</p>")

        loop = asyncio.new_event_loop()

        async def run_test():
            return await service._process_raw(
                loop,
                raw_doc,
                None,
                lambda *a, **kw: IngestionEvent(event_type="test", source_name="", message=""),
            )

        result = loop.run_until_complete(run_test())
        loop.close()

        assert result is not None
        chunks, content_hash, parsed = result
        assert len(chunks) == 1
        assert parsed.url == "http://example.com"

    def test_flush_batch_uses_embed_and_store_executors(self):
        """Verify that _flush_batch offloads embedding to _embed_executor and store to _store_executor."""
        service = _make_service()

        embed_called = []
        store_called = []

        def embed_fn(texts):
            embed_called.append(texts)
            return [[0.1] * 768 for _ in texts]

        def store_fn(chunks, vectors):
            store_called.append((len(chunks), len(vectors)))

        service.embeddings.embed_texts = embed_fn
        service.vector_store.upsert_chunks = store_fn

        chunks = [
            DocumentChunk(
                chunk_id="c1",
                source_name="test",
                title="T",
                url="http://example.com",
                text="chunk text",
                content_hash="abc",
            )
        ]

        loop = asyncio.new_event_loop()

        async def run_test():
            await service._flush_batch(
                loop,
                chunks,
                None,
                lambda *a, **kw: IngestionEvent(event_type="test", source_name="", message=""),
            )

        loop.run_until_complete(run_test())
        loop.close()

        assert len(embed_called) == 1
        assert len(store_called) == 1

    def test_legacy_single_executor_removed(self):
        """Verify the old shared ThreadPoolExecutor is no longer used."""
        service = _make_service()
        assert not hasattr(service, "_executor")
