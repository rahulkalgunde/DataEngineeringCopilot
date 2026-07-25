"""Tests for the langchain-based DocumentChunker with language detection."""

import pytest

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.chunker import DocumentChunker


class TestDocumentChunkerInitialization:
    def test_init_with_valid_parameters(self):
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100

    def test_init_defaults(self):
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100

    def test_init_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentChunker(chunk_size=0, chunk_overlap=10)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentChunker(chunk_size=-100, chunk_overlap=10)

    def test_init_overlap_must_be_valid(self):
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            DocumentChunker(chunk_size=100, chunk_overlap=-1)
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            DocumentChunker(chunk_size=100, chunk_overlap=100)


class TestLanguageDetection:
    def setup_method(self):
        self.chunker = DocumentChunker()

    def test_detects_python_from_api_python_path(self):
        lang = self.chunker._detect_language("https://spark.apache.org/docs/latest/api/python/index.html")
        assert lang is not None
        assert lang.name == "PYTHON"

    def test_detects_python_from_pyspark_path(self):
        lang = self.chunker._detect_language(
            "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql.html"
        )
        assert lang is not None
        assert lang.name == "PYTHON"

    def test_detects_scala_from_api_scala_path(self):
        lang = self.chunker._detect_language(
            "https://spark.apache.org/docs/latest/api/scala/org/apache/spark/index.html"
        )
        assert lang is not None
        assert lang.name == "SCALA"

    def test_detects_java_from_api_java_path(self):
        lang = self.chunker._detect_language("https://spark.apache.org/docs/latest/api/java/index.html")
        assert lang is not None
        assert lang.name == "JAVA"

    def test_detects_r_from_api_r_path(self):
        lang = self.chunker._detect_language("https://spark.apache.org/docs/latest/api/R/index.html")
        assert lang is not None
        assert lang.name == "R"

    def test_returns_none_for_sql_urls(self):
        lang = self.chunker._detect_language("https://spark.apache.org/docs/latest/api/sql/agg-functions/")
        assert lang is None

        lang = self.chunker._detect_language("https://spark.apache.org/docs/latest/sql-ref-annotated-types.html")
        assert lang is None

    def test_returns_none_for_non_code_urls(self):
        lang = self.chunker._detect_language("https://spark.apache.org/docs/latest/quick-start.html")
        assert lang is None

        lang = self.chunker._detect_language("https://spark.apache.org/docs/latest/")
        assert lang is None

        lang = self.chunker._detect_language("https://example.com/general/doc")
        assert lang is None


class TestLangchainChunker:
    def test_splits_text_into_chunks(self):
        text = "word " * 5000
        document = ParsedDocument(
            source_name="Test Source",
            title="Test Document",
            url="https://example.com/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(document)
        assert len(chunks) > 1
        assert all(c.source_name == "Test Source" for c in chunks)
        assert all(c.title == "Test Document" for c in chunks)
        assert all(c.url == document.url for c in chunks)

    def test_small_document_produces_single_chunk(self):
        text = "Short document with minimal content."
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(document)
        assert len(chunks) == 1

    def test_python_splitter_splits_on_class_boundaries(self):
        code = """
class SparkSession:
    def __init__(self, spark_context):
        self._sc = spark_context

    def sql(self, query):
        return self._sc.sql(query)

class DataFrame:
    def __init__(self, jdf):
        self._jdf = jdf

    def show(self, n=20):
        print(self._jdf.show(n))
"""
        document = ParsedDocument(
            source_name="PySpark",
            title="API",
            url="https://spark.apache.org/docs/latest/api/python/",
            text=code,
        )
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(document)
        assert len(chunks) >= 2

    def test_empty_text_returns_empty_list(self):
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text="",
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk(document)
        assert chunks == []

    def test_preserves_metadata_across_chunks(self):
        text = "word " * 5000
        document = ParsedDocument(
            source_name="Apache Spark Documentation",
            title="Spark SQL Guide",
            url="https://spark.apache.org/docs/latest/sql-programming-guide.html",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(document)
        for chunk in chunks:
            assert chunk.source_name == "Apache Spark Documentation"
            assert chunk.title == "Spark SQL Guide"
            assert chunk.url == document.url
            assert chunk.chunk_id.startswith("apache-spark-documentation:")


class TestChunkQualityValidation:
    def test_is_valid_chunk_with_valid_text(self):
        chunker = DocumentChunker()
        assert chunker._is_valid_chunk("hello world") is True

    def test_is_valid_chunk_rejects_empty_text(self):
        chunker = DocumentChunker()
        assert chunker._is_valid_chunk("") is False
        assert chunker._is_valid_chunk("   ") is False

    def test_is_valid_chunk_rejects_punctuation_only(self):
        chunker = DocumentChunker()
        assert chunker._is_valid_chunk("!@#$%") is False

    def test_is_valid_chunk_accepts_mixed_content(self):
        chunker = DocumentChunker()
        assert chunker._is_valid_chunk("hello 123 world!") is True


class TestChunkIDGeneration:
    def test_chunk_id_format(self):
        text = "word " * 500
        document = ParsedDocument(
            source_name="Apache Spark",
            title="Test",
            url="https://spark.apache.org/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(document)
        for i, chunk in enumerate(chunks):
            parts = chunk.chunk_id.split(":")
            assert len(parts) == 3
            assert parts[0] == "apache-spark"
            assert len(parts[1]) == 10
            assert parts[2] == f"{i:04d}"

    def test_chunk_id_deterministic(self):
        text = "word " * 500
        document = ParsedDocument(
            source_name="Test Source",
            title="Test",
            url="https://example.com/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks1 = chunker.chunk(document)
        chunks2 = chunker.chunk(document)
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=False):
            assert c1.chunk_id == c2.chunk_id

    def test_chunk_id_different_for_different_urls(self):
        text = "word " * 500
        doc1 = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/page1",
            text=text,
        )
        doc2 = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/page2",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks1 = chunker.chunk(doc1)
        chunks2 = chunker.chunk(doc2)
        assert chunks1[0].chunk_id != chunks2[0].chunk_id


class TestEdgeCases:
    def test_zero_overlap(self):
        text = "word " * 500
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(document)
        assert len(chunks) > 0

    def test_max_overlap(self):
        text = "word " * 500
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=499)
        chunks = chunker.chunk(document)
        assert len(chunks) > 0

    def test_very_small_chunk_size(self):
        text = "word " * 100
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk(document)
        assert len(chunks) > 0

    def test_tiny_chunk_size_with_large_overlap(self):
        text = "word " * 100
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=9)
        chunks = chunker.chunk(document)
        assert len(chunks) > 0

    def test_special_characters_in_text(self):
        text = "Hello! @world# $test% ^code& *examples. (More) [content] {here}."
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(document)
        assert len(chunks) > 0


class TestIntegration:
    def test_complex_document_chunking(self):
        text = (
            "Apache Spark is a unified computing engine for big data processing. "
            "It provides high-level APIs in Scala, Java, Python, and R. "
            "Spark is 100x faster than Hadoop. "
            "It provides comprehensive tools and libraries for data processing. "
            "This includes support for SQL, machine learning, and streaming. "
            "Spark runs on Hadoop, Mesos, Kubernetes, or standalone clusters. "
            "It can access HDFS, HBase, and other storage systems. "
            "Spark provides an interactive shell for ad-hoc querying. "
            "It supports multiple programming languages for flexibility. "
            "The RDD abstraction enables efficient parallel processing."
            "Apache Spark is a unified computing engine for big data processing. "
        ) * 20
        document = ParsedDocument(
            source_name="Apache Spark Documentation",
            title="Spark Overview",
            url="https://spark.apache.org/docs/latest/",
            text=text,
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(document)
        assert len(chunks) > 0
        assert all(isinstance(c.chunk_id, str) for c in chunks)
        assert all(len(c.text) > 0 for c in chunks)
        assert all(c.source_name == "Apache Spark Documentation" for c in chunks)

    def test_multiple_documents_independent_ids(self):
        docs = [
            ParsedDocument(
                source_name="Source 1",
                title="Document 1",
                url="https://example.com/doc1",
                text="word " * 1000,
            ),
            ParsedDocument(
                source_name="Source 2",
                title="Document 2",
                url="https://example.com/doc2",
                text="word " * 1000,
            ),
        ]
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunker.chunk(doc))
        chunk_ids = [c.chunk_id for c in all_chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunker_reusability(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        for i in range(5):
            document = ParsedDocument(
                source_name=f"Source {i}",
                title=f"Document {i}",
                url=f"https://example.com/doc{i}",
                text="word " * 1000,
            )
            chunks = chunker.chunk(document)
            assert len(chunks) > 0
