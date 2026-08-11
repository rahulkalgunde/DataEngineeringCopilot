"""E2E ingest → query tests: ingest sample HTML → retrieve → LLM answer.

Uses factory functions with custom AppSettings (isolated collection).
Real Redis, Qdrant, and Ollama. Cleans up collection after session.
"""

from __future__ import annotations

import dataclasses
import hashlib
import uuid

import pytest

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import RawDocument
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.services.chunker import DocumentChunker
from tests.conftest import require_qdrant_and_ollama

SAMPLE_HTML = """<html><head><title>Apache Spark Overview</title></head><body>
<nav>Navigation sidebar</nav>
<main>
<h1>Apache Spark Overview</h1>
<p>Apache Spark is a unified analytics engine for large-scale data processing.
It provides high-level APIs in Scala, Java, Python, and R, and an optimized
engine that supports general execution graphs.</p>
<p>Spark SQL is a Spark module for structured data processing. It provides a
programming abstraction called DataFrames and can also act as a distributed
SQL query engine.</p>
<pre><code class="language-python">from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("example").getOrCreate()</code></pre>
</main></body></html>"""


def _build_ingestion_service(settings: AppSettings):
    """Build AsyncIngestionService with custom settings (isolated collection)."""
    from data_engineering_copilot.factory import build_async_ingestion_service

    return build_async_ingestion_service(app_settings=settings)


def _build_rag_service(settings: AppSettings):
    """Build AsyncRagService with custom settings (isolated collection)."""
    from data_engineering_copilot.factory import build_rag_service

    return build_rag_service(app_settings=settings)


@pytest.mark.e2e
class TestIngestQueryPipeline:
    """Journey 1: Ingest sample HTML → query → verify answer."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_infra(self, e2e_settings):
        require_qdrant_and_ollama(e2e_settings.qdrant_url)

    async def test_ingest_sample_then_query(self, e2e_settings):
        """Full pipeline: parse → chunk → embed → upsert → query → answer."""
        service = _build_ingestion_service(e2e_settings)
        try:
            await service.vector_store.initialize()

            raw = RawDocument(
                source_name="Apache Spark Documentation",
                url="https://spark.apache.org/docs/latest/",
                html=SAMPLE_HTML,
            )
            parsed = MarkdownParser().parse(raw)
            assert parsed is not None, "Parser should return parsed doc"

            chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
            chunks = await chunker.chunk(parsed)
            assert len(chunks) >= 1, "Should produce at least one chunk"

            texts = [c.text for c in chunks]
            embeddings = await service.embeddings.embed_texts(texts)
            await service.vector_store.upsert_chunks(chunks, embeddings)

            count = await service.vector_store.count()
            assert count == len(chunks), "All chunks should be upserted"

            # Now build RAG and query
            rag = _build_rag_service(e2e_settings)
            answer = await rag.answer("What is Apache Spark?")
            assert answer is not None
            assert len(answer.text) > 20, "Answer should be substantial"
            assert answer.confidence >= 0.0
        finally:
            await service.embeddings.close()

    async def test_ingest_idempotent(self, e2e_settings):
        """Ingesting the same source twice should not create duplicate chunks."""
        service = _build_ingestion_service(e2e_settings)
        try:
            await service.vector_store.initialize()

            raw = RawDocument(
                source_name="Apache Spark Documentation",
                url="https://spark.apache.org/docs/latest/",
                html=SAMPLE_HTML,
            )
            parsed = MarkdownParser().parse(raw)
            assert parsed is not None
            chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
            chunks = await chunker.chunk(parsed)
            texts = [c.text for c in chunks]

            embeddings = await service.embeddings.embed_texts(texts)
            await service.vector_store.upsert_chunks(chunks, embeddings)
            count1 = await service.vector_store.count()

            embeddings2 = await service.embeddings.embed_texts(texts)
            await service.vector_store.upsert_chunks(chunks, embeddings2)
            count2 = await service.vector_store.count()

            assert count2 == count1, "Idempotent upsert should not increase count"
        finally:
            await service.embeddings.close()

    @pytest.mark.serial
    async def test_query_before_ingest_returns_low_confidence(self, e2e_settings):
        """Querying an empty collection should return low confidence."""
        isolated = e2e_settings.model_copy(
            update={
                "collection_name": f"e2e_empty_{uuid.uuid4().hex[:8]}",
                "redis_url": e2e_settings.redis_url.rsplit("/", 1)[0] + "/15",
            }
        )
        rag = _build_rag_service(isolated)
        await rag.vector_store.initialize()
        answer = await rag.answer(f"What is the capital of Atlantida {uuid.uuid4().hex[:6]}?")
        assert answer.confidence == 0.0

    async def test_content_hash_persisted(self, e2e_settings):
        """After upsert, content hash should be retrievable via URL."""
        service = _build_ingestion_service(e2e_settings)
        try:
            await service.vector_store.initialize()

            url = f"https://spark.apache.org/docs/latest/content-hash/{uuid.uuid4().hex}"
            raw = RawDocument(source_name="test", url=url, html=SAMPLE_HTML)
            parsed = MarkdownParser().parse(raw)
            assert parsed is not None
            content_hash = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()

            chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
            chunks = await chunker.chunk(parsed)
            chunks = [dataclasses.replace(c, content_hash=content_hash) for c in chunks]
            texts = [c.text for c in chunks]
            embeddings = await service.embeddings.embed_texts(texts)
            await service.vector_store.upsert_chunks(chunks, embeddings)

            stored = await service.vector_store.get_content_hash_for_url(url)
            assert stored == content_hash
        finally:
            await service.embeddings.close()

    async def test_delete_by_url_removes_chunks(self, e2e_settings):
        """Deleting by URL should remove all chunks for that URL."""
        isolated = e2e_settings.model_copy(
            update={"collection_name": f"e2e_delete_test_{uuid.uuid4().hex[:8]}"}
        )
        service = _build_ingestion_service(isolated)
        try:
            await service.vector_store.initialize()

            url = f"https://spark.apache.org/docs/latest/delete-test/{uuid.uuid4().hex}"
            raw = RawDocument(source_name="test", url=url, html=SAMPLE_HTML)
            parsed = MarkdownParser().parse(raw)
            assert parsed is not None
            chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
            chunks = await chunker.chunk(parsed)
            texts = [c.text for c in chunks]
            embeddings = await service.embeddings.embed_texts(texts)
            await service.vector_store.upsert_chunks(chunks, embeddings)

            count_before = await service.vector_store.count()
            assert count_before > 0

            await service.vector_store.delete_by_url(url)
            count_after = await service.vector_store.count()
            assert count_after == 0
        finally:
            await service.embeddings.close()
