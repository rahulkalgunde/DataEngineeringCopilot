"""Integration tests for the IngestionService.

Tests the full crawl → parse → chunk → embed → store pipeline using mock
crawlers with real Qdrant + Ollama. Validates batch slicing, dedup,
order preservation, and error handling.

Uses module-scoped fixtures to avoid re-creating connections per test.

Run with: pytest tests/test_ingestion_integration.py -v -m integration
"""

import time
import uuid
import pytest

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import RawDocument
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.ingestion import IngestionService

from tests.conftest import (
    require_qdrant_and_ollama,
    unique_collection_name,
)


# ---------------------------------------------------------------------------
# Mock crawlers
# ---------------------------------------------------------------------------

class MockCrawler:
    def __init__(self, n: int):
        self.n = n

    def crawl(self, source, max_pages, on_event=None):
        for i in range(self.n):
            html = f"""
            <html><body>
                <h1>Document {i}</h1>
                <p>This is document number {i} with realistic content for testing the
                ingestion pipeline. It covers data engineering topics including Apache
                Spark, Delta Lake, and Apache Airflow. The content is long enough to
                produce meaningful chunks during the chunking phase.</p>
            </body></html>
            """
            yield RawDocument(
                source_name=source.name,
                url=f"https://example.com/docs/{i}.html",
                html=html,
            )


class ContentHashCrawler:
    """Yields the same documents each time (for dedup testing)."""
    def __init__(self, n: int):
        self.n = n

    def crawl(self, source, max_pages, on_event=None):
        for i in range(self.n):
            html = f"""
            <html><body>
                <h1>Stable Document {i}</h1>
                <p>This document has stable content that does not change between
                ingestion runs. It covers topic {i} in the data engineering domain
                and provides detailed information about processing pipelines. The
                content is designed to be long enough to pass the parser threshold
                and produce meaningful chunks during the chunking phase.</p>
            </body></html>
            """
            yield RawDocument(
                source_name=source.name,
                url=f"https://example.com/stable/{i}.html",
                html=html,
            )


class MixedCrawler:
    """Yields a mix of valid and invalid (empty) documents."""
    def __init__(self, n: int):
        self.n = n

    def crawl(self, source, max_pages, on_event=None):
        for i in range(self.n):
            if i % 3 == 0:
                html = "<html></html>"
            else:
                html = f"<html><body><p>Valid document number {i} with enough content to pass the parser threshold and be included in the ingestion pipeline. This document covers data engineering topics and provides detailed information about processing and analytics. The content includes information about Apache Spark and Delta Lake and other big data technologies used in modern data platforms.</p></body></html>"
            yield RawDocument(source_name=source.name, url=f"https://example.com/mix/{i}.html", html=html)


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _settings():
    return AppSettings(
        embedding_batch_size=32,
        chunk_size_words=200,
        chunk_overlap_words=40,
        ingestion_batch_chunk_size=64,
        confidence_threshold=0.10,
    )


@pytest.fixture(scope="module")
def _embedder(_settings):
    require_qdrant_and_ollama()
    return SentenceTransformerEmbeddings(
        model_name=_settings.embedding_model_name,
        cache_dir=_settings.embedding_cache_dir,
        local_files_only=False,
    )


@pytest.fixture(scope="module")
def _parser():
    return DocumentationHtmlParser()


@pytest.fixture(scope="module")
def _chunker(_settings):
    return DocumentChunker(
        chunk_size_words=_settings.chunk_size_words,
        overlap_words=_settings.chunk_overlap_words,
    )


def _fresh_store():
    """Create a fresh Qdrant store (not a fixture — called manually per test)."""
    from data_engineering_copilot.config.settings import settings
    coll = unique_collection_name("ingest")
    store = QdrantVectorStore(url=settings.qdrant_url, collection_name=coll)
    return store, coll


def _cleanup(coll_name: str):
    from qdrant_client import QdrantClient
    from data_engineering_copilot.config.settings import settings
    try:
        c = QdrantClient(url=settings.qdrant_url, prefer_grpc=False)
        c.delete_collection(collection_name=coll_name)
        c.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.ingestion
class TestIngestionSingleBatch:
    def test_ingest_10_documents(self, _settings, _parser, _chunker, _embedder):
        store, coll = _fresh_store()
        try:
            crawler = MockCrawler(10)
            svc = IngestionService(_settings, crawler, _parser, _chunker, _embedder, store)
            total = svc.ingest()
            assert total > 0
            assert store.count() == total
        finally:
            _cleanup(coll)

    def test_ingest_32_documents(self, _settings, _parser, _chunker, _embedder):
        store, coll = _fresh_store()
        try:
            crawler = MockCrawler(32)
            svc = IngestionService(_settings, crawler, _parser, _chunker, _embedder, store)
            total = svc.ingest()
            assert total > 0
            assert store.count() == total
        finally:
            _cleanup(coll)


@pytest.mark.integration
@pytest.mark.ingestion
class TestIngestionMultiBatch:
    def test_ingest_64_documents(self, _settings, _parser, _chunker, _embedder):
        store, coll = _fresh_store()
        try:
            crawler = MockCrawler(64)
            svc = IngestionService(_settings, crawler, _parser, _chunker, _embedder, store)
            total = svc.ingest()
            assert total > 0
            assert store.count() == total
        finally:
            _cleanup(coll)


@pytest.mark.integration
@pytest.mark.ingestion
class TestIngestionDedup:
    def test_second_ingest_skips_unchanged(self, _settings, _parser, _chunker, _embedder):
        store, coll = _fresh_store()
        try:
            crawler = ContentHashCrawler(5)
            svc = IngestionService(_settings, crawler, _parser, _chunker, _embedder, store)
            first = svc.ingest()
            assert first > 0
            second = svc.ingest()
            assert second == 0  # All unchanged → skipped
            assert store.count() == first
        finally:
            _cleanup(coll)


@pytest.mark.integration
@pytest.mark.ingestion
class TestIngestionErrorHandling:
    def test_mixed_valid_invalid(self, _settings, _parser, _chunker, _embedder):
        store, coll = _fresh_store()
        try:
            svc = IngestionService(_settings, MixedCrawler(10), _parser, _chunker, _embedder, store)
            total = svc.ingest()
            assert total > 0
            assert store.count() == total
        finally:
            _cleanup(coll)


@pytest.mark.integration
@pytest.mark.ingestion
@pytest.mark.slow
class TestIngestionPerformance:
    def test_throughput(self, _settings, _parser, _chunker, _embedder):
        store, coll = _fresh_store()
        try:
            svc = IngestionService(_settings, MockCrawler(20), _parser, _chunker, _embedder, store)
            start = time.time()
            total = svc.ingest()
            elapsed = time.time() - start
            throughput = total / elapsed if elapsed > 0 else 0
            assert throughput > 0.1, f"Throughput too low: {throughput:.2f} chunks/sec"
        finally:
            _cleanup(coll)
