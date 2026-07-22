"""End-to-end tests for the async ingestion pipeline.

Tests the complete flow: parse HTML → chunk → embed → upsert → query.
Uses sample HTML to avoid depending on external site availability.

Requires Qdrant and Ollama to be running.

Run with: ``dec_venv/bin/python -m pytest tests/e2e/ -v -m ingestion``
"""

import dataclasses
import hashlib
import time

import pytest

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import RawDocument
from data_engineering_copilot.infrastructure.embeddings import OllamaEmbeddings
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from tests.conftest import require_qdrant_and_ollama, unique_collection_name

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
def embedder():
    require_qdrant_and_ollama()
    return OllamaEmbeddings(model_name=AppSettings().embedding_model_name)


@pytest.fixture
def vector_store():
    require_qdrant_and_ollama()
    from qdrant_client import QdrantClient

    coll = unique_collection_name("e2e_ingest")
    store = QdrantVectorStore(url=AppSettings().qdrant_url, collection_name=coll)
    yield store
    try:
        c = QdrantClient(url=AppSettings().qdrant_url, prefer_grpc=False)
        c.delete_collection(collection_name=coll)
        c.close()
    except Exception:
        pass


@pytest.mark.ingestion
class TestIngestionPipelineE2E:
    """End-to-end test: parse HTML → chunk → embed → upsert → query."""

    def _ingest_sample(self, vector_store, embedder) -> int:
        raw = RawDocument(
            source_name="Apache Spark Documentation",
            url="https://spark.apache.org/docs/latest/",
            html=SAMPLE_HTML,
        )
        parsed = MarkdownParser().parse(raw)
        assert parsed is not None

        chunker = DocumentChunker(chunk_size_words=100, overlap_words=20)
        chunks = chunker.chunk(parsed)
        assert len(chunks) >= 2

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.upsert_chunks(chunks, embeddings)
        return len(chunks)

    def test_parse_chunk_embed_upsert(self, vector_store, embedder):
        n = self._ingest_sample(vector_store, embedder)
        assert vector_store.count() == n

    def test_ingested_data_queryable(self, vector_store, embedder):
        self._ingest_sample(vector_store, embedder)

        query = "What is Apache Spark used for?"
        query_emb = embedder.embed_query(query)
        results = vector_store.query(query_emb, top_k=5)

        assert len(results) > 0
        assert any(
            "unified analytics engine" in r.chunk.text.lower()
            for r in results
        ), "Should find Spark content"

    def test_ingestion_idempotent(self, vector_store, embedder):
        n = self._ingest_sample(vector_store, embedder)
        self._ingest_sample(vector_store, embedder)
        assert vector_store.count() == n

    def test_content_hash_persisted(self, vector_store, embedder):
        url = "https://spark.apache.org/docs/latest/"
        raw = RawDocument(
            source_name="Apache Spark Documentation",
            url=url,
            html=SAMPLE_HTML,
        )
        parsed = MarkdownParser().parse(raw)
        content_hash = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        chunker = DocumentChunker(chunk_size_words=100, overlap_words=20)
        chunks = chunker.chunk(parsed)
        chunks = [dataclasses.replace(c, content_hash=content_hash) for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.upsert_chunks(chunks, embeddings)

        stored = vector_store.get_content_hash_for_url(url)
        assert stored == content_hash

    def test_delete_by_url_removes_chunks(self, vector_store, embedder):
        url = "https://spark.apache.org/docs/latest/"
        raw = RawDocument(
            source_name="Apache Spark Documentation",
            url=url,
            html=SAMPLE_HTML,
        )
        parsed = MarkdownParser().parse(raw)
        chunker = DocumentChunker(chunk_size_words=100, overlap_words=20)
        chunks = chunker.chunk(parsed)
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.upsert_chunks(chunks, embeddings)

        count_before = vector_store.count()
        assert count_before > 0

        vector_store.delete_by_url(url)
        time.sleep(0.3)
        assert vector_store.count() == 0


@pytest.mark.ingestion
class TestParserAndChunker:
    """Parser and chunker edge cases with sample HTML."""

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
