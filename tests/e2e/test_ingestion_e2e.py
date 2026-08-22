"""End-to-end tests for the async ingestion pipeline.

Tests the complete flow: parse HTML → chunk → embed → upsert → query.
Uses sample HTML to avoid depending on external site availability.

Requires Qdrant (testcontainer) and Ollama (Docker Compose).

Run with: ``dec_venv/bin/python -m pytest tests/e2e/ -v -m ingestion``
"""

import dataclasses
import hashlib
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.domain.models import RawDocument
from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import LocalSentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.services.chunker import DocumentChunker
from tests.conftest import require_qdrant_and_ollama


class _AsyncListIterator:
    """Async iterator wrapper for a plain list, usable with ``async for``."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


SAMPLE_HTML = """<html><head><title>Apache Spark Overview</title></head><body>
<nav>Navigation sidebar</nav>
<main>
<h1>Apache Spark Overview</h1>
<p>Apache Spark is a unified analytics engine for large-scale data processing.
It provides high-level APIs in Scala, Java, Python, and R, and an optimized
engine that supports general execution graphs. It also supports a rich set of
higher-level tools including Spark SQL for SQL and structured data processing,
pandas API on Spark for pandas workloads, MLlib for machine learning, GraphX
for graph processing, and Structured Streaming for stream processing.</p>
<p>Spark offers over 80 high-level operators that make it easy to build parallel
apps. You can use it interactively from the Scala, Python, R, and SQL shells.
Spark is designed to cover a wide range of workloads such as batch applications,
iterative algorithms, interactive queries, and streaming. Apart from the
interactive shells, Spark also supports running applications in Java, Scala,
Python, and R.</p>
<h2>Spark SQL</h2>
<p>Spark SQL is a Spark module for structured data processing. It provides a
programming abstraction called DataFrames and can also act as a distributed
SQL query engine. It enables unmodified Hadoop Hive queries to run up to
100x faster on existing deployments and data.</p>
<h2>Structured Streaming</h2>
<p>Structured Streaming is a scalable and fault-tolerant stream processing
engine built on the Spark SQL engine. You can express your streaming
computation the same way you would express a batch computation on static
data. The Spark SQL engine will take care of running it incrementally and
continuously and updating the final result as streaming data continues.</p>
<pre><code class="language-python">from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("example").getOrCreate()
df = spark.read.csv("data.csv")
df.show()</code></pre>
</main>
<footer>Footer links</footer>
</body></html>"""


@pytest.fixture
async def embedder(e2e_settings):
    require_qdrant_and_ollama(e2e_settings.qdrant_url)
    emb = LocalSentenceTransformerEmbeddings(
        model_name=e2e_settings.local_hf_embedding_model,
        embedding_dimension=e2e_settings.get_embedding_dimension(),
    )
    yield emb
    await emb.close()


@pytest.fixture
async def vector_store(e2e_settings):
    require_qdrant_and_ollama(e2e_settings.qdrant_url)
    from qdrant_client import QdrantClient

    coll = f"e2e_ingest_{__import__('uuid').uuid4().hex[:8]}"
    store = AsyncQdrantVectorStore(
        url=e2e_settings.qdrant_url,
        collection_name=coll,
        embedding_dimension=2048,
    )
    await store.initialize()
    yield store
    await store.close()
    try:
        c = QdrantClient(url=e2e_settings.qdrant_url, prefer_grpc=False)
        c.delete_collection(collection_name=coll)
        c.close()
    except Exception:
        pass


@pytest.mark.ingestion
class TestIngestionPipelineE2E:
    """End-to-end test: parse HTML → chunk → embed → upsert → query."""

    async def _ingest_sample(self, vector_store, embedder) -> int:
        raw = RawDocument(
            source_name="Apache Spark Documentation",
            url="https://spark.apache.org/docs/latest/",
            html=SAMPLE_HTML,
        )
        parsed = MarkdownParser().parse(raw)
        assert parsed is not None

        chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
        chunks = await chunker.chunk(parsed)
        assert len(chunks) >= 2

        texts = [c.text for c in chunks]
        embeddings = await embedder.embed_texts(texts)
        await vector_store.upsert_chunks(chunks, embeddings)
        return len(chunks)

    async def test_parse_chunk_embed_upsert(self, vector_store, embedder):
        n = await self._ingest_sample(vector_store, embedder)
        assert await vector_store.count() == n

    async def test_service_ingest_real_chunker_indexes(self, e2e_settings, vector_store, embedder):
        """Run the real AsyncIngestionService.ingest() with a real DocumentChunker.

        This crosses the exact service↔real-chunker seam in _process_raw that the
        hand-wired tests bypass. Guards the d7e595d regression where every page
        was silently marked SKIPPED (0 chunks indexed) under the default
        sentence_preserving strategy. Asserts chunks_indexed > 0 (CI invariant).
        """
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        crawler = MagicMock()
        crawler.crawl = MagicMock(
            return_value=_AsyncListIterator(
                [
                    RawDocument(
                        source_name="Apache Spark Documentation",
                        url="https://spark.apache.org/docs/latest/",
                        html=SAMPLE_HTML,
                    )
                ]
            )
        )
        frontier = MagicMock()
        frontier.stats = AsyncMock(return_value={"DISCOVERED": 1})
        frontier.all_urls = AsyncMock(return_value=[])
        frontier.reactivate_missing = AsyncMock(return_value=0)
        frontier.mark_processed = AsyncMock()
        frontier.mark_failed = AsyncMock()
        frontier.mark_skipped = AsyncMock()
        frontier.close = AsyncMock()
        crawler.frontier = frontier

        from data_engineering_copilot.config.settings import DocumentationSource

        source = DocumentationSource(
            name="Apache Spark Documentation",
            start_urls=("https://spark.apache.org/docs/latest/",),
            allowed_domains=("spark.apache.org",),
            url_prefixes=("https://spark.apache.org/docs/latest/",),
        )
        crawler.crawl.return_value = _AsyncListIterator(
            [
                RawDocument(
                    source_name=source.name,
                    url="https://spark.apache.org/docs/latest/",
                    html=SAMPLE_HTML,
                )
            ]
        )

        service = AsyncIngestionService(
            settings=e2e_settings,
            crawler=crawler,
            parser=MarkdownParser(),
            chunker=DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100),
            embeddings=embedder,
            vector_store=vector_store,
        )
        try:
            total = await service.ingest(
                max_pages_per_source=1,
                source_names=["Apache Spark Documentation"],
            )
        finally:
            service.stop()

        assert total > 0, "Service must index at least one chunk (was silently skipping every page)"
        assert frontier.mark_skipped.await_count == 0, "No page should have been silently skipped"
        assert frontier.mark_processed.await_count >= 1

        # ingest() closes the shared vector_store client internally; use a fresh
        # client over the same collection to assert points were persisted.
        verify_store = AsyncQdrantVectorStore(
            url=e2e_settings.qdrant_url,
            collection_name=vector_store._collection_name,
            embedding_dimension=2048,
        )
        try:
            assert await verify_store.count() > 0, "Qdrant must contain indexed chunks"
        finally:
            await verify_store.close()

    async def test_service_ingest_real_semantic_chunker_indexes(self, e2e_settings, vector_store, embedder):
        """Real SemanticChunker through AsyncIngestionService.ingest().

        Crosses the service↔real-SemanticChunker seam including the list branch
        of _process_raw (extract_sentences -> embed_texts -> chunk(parsed,
        embeddings)), which had zero coverage at any level while the None/[] bug
        was live. Asserts chunks were indexed and no page was silently skipped.
        """
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        crawler = MagicMock()
        frontier = MagicMock()
        frontier.stats = AsyncMock(return_value={"DISCOVERED": 1})
        frontier.all_urls = AsyncMock(return_value=[])
        frontier.reactivate_missing = AsyncMock(return_value=0)
        frontier.mark_processed = AsyncMock()
        frontier.mark_failed = AsyncMock()
        frontier.mark_skipped = AsyncMock()
        frontier.close = AsyncMock()
        crawler.frontier = frontier

        crawler.crawl.return_value = _AsyncListIterator(
            [
                RawDocument(
                    source_name="Apache Spark Documentation",
                    url="https://spark.apache.org/docs/latest/",
                    html=SAMPLE_HTML,
                )
            ]
        )

        service = AsyncIngestionService(
            settings=e2e_settings,
            crawler=crawler,
            parser=MarkdownParser(),
            chunker=SemanticChunker(
                chunk_size_words=80,
                overlap_words=10,
                embedding_model=embedder,
                min_chunk_words=15,
            ),
            embeddings=embedder,
            vector_store=vector_store,
        )
        try:
            total = await service.ingest(
                max_pages_per_source=1,
                source_names=["Apache Spark Documentation"],
            )
        finally:
            service.stop()

        assert total > 0, "SemanticChunker must index at least one chunk"
        assert frontier.mark_skipped.await_count == 0, "No page should have been silently skipped"
        assert frontier.mark_processed.await_count >= 1

        verify_store = AsyncQdrantVectorStore(
            url=e2e_settings.qdrant_url,
            collection_name=vector_store._collection_name,
            embedding_dimension=2048,
        )
        try:
            assert await verify_store.count() > 0, "Qdrant must contain semantically chunked points"
        finally:
            await verify_store.close()

    async def test_ingested_data_queryable(self, vector_store, embedder):
        await self._ingest_sample(vector_store, embedder)

        query = "What is Apache Spark used for?"
        query_emb = await embedder.embed_query(query)
        results = await vector_store.query(query_emb, top_k=5)

        assert len(results) > 0
        assert any("unified analytics engine" in r.chunk.text.lower() for r in results), "Should find Spark content"

    async def test_ingestion_idempotent(self, vector_store, embedder):
        n = await self._ingest_sample(vector_store, embedder)
        await self._ingest_sample(vector_store, embedder)
        assert await vector_store.count() == n

    async def test_content_hash_persisted(self, vector_store, embedder):
        url = f"https://spark.apache.org/docs/latest/content-hash/{uuid.uuid4().hex}"
        raw = RawDocument(
            source_name="Apache Spark Documentation",
            url=url,
            html=SAMPLE_HTML,
        )
        parsed = MarkdownParser().parse(raw)
        assert parsed is not None
        content_hash = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
        chunks = await chunker.chunk(parsed)
        chunks = [dataclasses.replace(c, content_hash=content_hash) for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = await embedder.embed_texts(texts)
        await vector_store.upsert_chunks(chunks, embeddings)

        stored = await vector_store.get_content_hash_for_url(url)
        assert stored == content_hash

    async def test_delete_by_url_removes_chunks(self, vector_store, embedder):
        url = f"https://spark.apache.org/docs/latest/delete-test/{uuid.uuid4().hex}"
        raw = RawDocument(
            source_name="Apache Spark Documentation",
            url=url,
            html=SAMPLE_HTML,
        )
        parsed = MarkdownParser().parse(raw)
        assert parsed is not None
        chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
        chunks = await chunker.chunk(parsed)
        texts = [c.text for c in chunks]
        embeddings = await embedder.embed_texts(texts)
        await vector_store.upsert_chunks(chunks, embeddings)

        count_before = await vector_store.count()
        assert count_before > 0

        await vector_store.delete_by_url(url)
        time.sleep(0.3)
        assert await vector_store.count() == 0


class TestParserAndChunker:
    """Parser and chunker edge cases with sample HTML (pure unit tests, no infra)."""

    def test_parser_strips_nav_and_footer(self):
        raw = RawDocument(
            source_name="Test",
            url="https://spark.apache.org/docs/latest/",
            html=SAMPLE_HTML,
        )
        parsed = MarkdownParser().parse(raw)
        assert parsed is not None
        assert "Navigation sidebar" not in parsed.text
        assert "Footer links" not in parsed.text
        assert "# Apache Spark Overview" in parsed.text
        assert parsed.title == "Apache Spark Overview"

    def test_parser_returns_none_for_short_page(self):
        raw = RawDocument(
            source_name="Test",
            url="https://example.com/short",
            html="<html><body><main><p>Too short.</p></main></body></html>",
        )
        parsed = MarkdownParser().parse(raw)
        assert parsed is None

    def test_parser_preserves_code_blocks(self):
        raw = RawDocument(
            source_name="Test",
            url="https://spark.apache.org/docs/latest/",
            html=SAMPLE_HTML,
        )
        parsed = MarkdownParser().parse(raw)
        assert parsed is not None
        assert "from pyspark.sql import SparkSession" in parsed.text
