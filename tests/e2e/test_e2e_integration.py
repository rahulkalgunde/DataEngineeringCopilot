"""End-to-end integration tests.

Tests the complete pipeline from ingestion through RAG query, verifying that
data flows correctly through all layers: crawl → parse → chunk → embed → store →
retrieve → rerank → assemble → generate.

Run with: pytest tests/test_e2e_integration.py -v -m integration
"""

import time

import pytest

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import Answer, RawDocument
from data_engineering_copilot.infrastructure.embeddings import OllamaEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.ingestion import IngestionService
from data_engineering_copilot.services.rag import RagAnswerService
from tests.conftest import (
    require_qdrant_and_ollama,
    unique_collection_name,
)

# ---------------------------------------------------------------------------
# Mock crawler
# ---------------------------------------------------------------------------


class SparkDocsCrawler:
    def __init__(self, source_index: int = 0):
        """source_index varies the URL path to avoid content-hash dedup across tests."""
        self.source_index = source_index

    def crawl(self, source, max_pages, on_event=None):
        prefix = f"e2e{self.source_index}"
        pages = [
            (
                "Spark Quick Start",
                "Apache Spark is a unified analytics engine for large-scale data processing. Spark provides an interface for programming entire clusters with implicit data parallelism and fault tolerance. It supports multiple programming languages and provides a rich set of built-in tools for SQL, streaming, and machine learning workloads.",
                f"https://example.com/{prefix}/quick-start.html",
            ),
            (
                "Spark SQL Guide",
                "Spark SQL is a Spark module for structured data processing. It provides a programming abstraction called DataFrames and SQL, and can act as a distributed SQL query engine. Spark SQL supports reading and writing data in various structured formats including JSON, Parquet, ORC, and Hive tables.",
                f"https://example.com/{prefix}/sql.html",
            ),
            (
                "Spark Streaming",
                "Structured Streaming is a scalable stream processing engine built on the Spark SQL engine. You can express your streaming computation the same way you would express a batch computation on static data. The Spark SQL engine will run it incrementally and continuously to update the result table.",
                f"https://example.com/{prefix}/streaming.html",
            ),
            (
                "MLlib Overview",
                "MLlib is Spark's machine learning library. It provides practical machine learning tools including classification, regression, clustering, collaborative filtering, and feature extraction. MLlib also provides utilities for linear algebra, statistics, and data handling for large-scale applications.",
                f"https://example.com/{prefix}/mllib.html",
            ),
            (
                "Delta Lake Introduction",
                "Delta Lake is an open-source storage framework that enables building a lakehouse architecture on top of data lakes. It provides ACID transactions, scalable metadata handling, and unifies streaming and batch data processing for reliable data pipelines.",
                f"https://example.com/{prefix}/intro.html",
            ),
        ]
        for title, text, url in pages:
            html = f"<html><body><h1>{title}</h1><p>{text}</p></body></html>"
            yield RawDocument(source_name=source.name, url=url, html=html)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store():
    from data_engineering_copilot.config.settings import settings

    coll = unique_collection_name("e2e")
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


def _make_components():
    require_qdrant_and_ollama()
    settings = AppSettings(
        embedding_batch_size=32,
        chunk_size_words=200,
        chunk_overlap_words=40,
        retrieval_top_k=10,
        reranker_enabled=True,
        reranker_top_k=5,
        max_context_chars=2000,
        confidence_threshold=0.10,
        ingestion_batch_chunk_size=64,
    )
    embedder = OllamaEmbeddings(
        model_name=settings.embedding_model_name,
    )
    ollama = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
    )
    parser = DocumentationHtmlParser()
    chunker = DocumentChunker(
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    return settings, embedder, ollama, parser, chunker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    def test_ingest_then_query(self):
        settings, embedder, ollama, parser, chunker = _make_components()
        store, coll = _make_store()
        try:
            ingest_svc = IngestionService(settings, SparkDocsCrawler(source_index=1), parser, chunker, embedder, store)
            start = time.time()
            total_chunks = ingest_svc.ingest()
            time.time() - start
            assert total_chunks > 0
            assert store.count() == total_chunks

            rag_svc = RagAnswerService(vector_store=store, ollama_client=ollama, embedder=embedder)
            start = time.time()
            answer = rag_svc.answer("What is Apache Spark?")
            time.time() - start

            assert isinstance(answer, Answer)
            assert len(answer.text) > 20, f"RAG answer too short: {answer.text!r}"
            assert answer.confidence >= 0.0
        finally:
            _cleanup(coll)

    def test_ingest_then_multiple_queries(self):
        settings, embedder, ollama, parser, chunker = _make_components()
        store, coll = _make_store()
        try:
            ingest_svc = IngestionService(settings, SparkDocsCrawler(source_index=2), parser, chunker, embedder, store)
            total = ingest_svc.ingest()
            assert total > 0

            rag_svc = RagAnswerService(vector_store=store, ollama_client=ollama, embedder=embedder)
            for q in ["What is Apache Spark?", "What is Delta Lake?", "What is MLlib?"]:
                answer = rag_svc.answer(q)
                assert len(answer.text) > 10, f"Bad answer for: {q}"
        finally:
            _cleanup(coll)

    def test_ingestion_events_fired(self):
        settings, embedder, ollama, parser, chunker = _make_components()
        store, coll = _make_store()
        try:
            events = []
            ingest_svc = IngestionService(settings, SparkDocsCrawler(source_index=3), parser, chunker, embedder, store)
            total = ingest_svc.ingest(on_event=lambda e: events.append(e))

            event_types = [e.event_type for e in events]
            assert "source_start" in event_types
            assert "source_complete" in event_types
            assert "page_indexed" in event_types
            assert total > 0
        finally:
            _cleanup(coll)
