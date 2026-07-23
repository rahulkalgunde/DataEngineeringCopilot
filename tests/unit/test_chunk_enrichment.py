"""Tests for DocumentChunk enrichment (Phase 1C)."""

from __future__ import annotations

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.services.chunk_enrichment import enrich_chunks


def _make_chunk(text: str = "Test text", url: str = "http://example.com/1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id="src:abc1234:0000",
        source_name="spark-docs",
        title="Spark Guide",
        url=url,
        text=text,
    )


class TestEnrichChunks:
    def test_empty_input_returns_empty(self):
        assert enrich_chunks([]) == []

    def test_adds_content_quality_score(self):
        chunk = _make_chunk(text="Spark SQL is a powerful module for structured data processing.")
        result = enrich_chunks([chunk])
        assert len(result) == 1
        assert 0.0 <= result[0].content_quality_score <= 1.0

    def test_quality_score_low_for_gibberish(self):
        chunk = _make_chunk(text="aaa bbb ccc ddd")
        result = enrich_chunks([chunk])
        assert result[0].content_quality_score < 0.5

    def test_quality_score_high_for_code_and_text(self):
        chunk = _make_chunk(
            text="Use spark.sql('SELECT * FROM table') to query data. "
            "The DataFrame API provides transformations like filter and select."
        )
        result = enrich_chunks([chunk])
        assert result[0].content_quality_score > 0.5

    def test_source_type_detected_for_api_reference(self):
        chunk = _make_chunk(
            text="def my_function(param): return param\nclass MyClass:\n    pass",
            url="http://example.com/api/reference",
        )
        result = enrich_chunks([chunk])
        assert result[0].source_type in ("api_reference", "tutorial", "concept")

    def test_source_type_detected_for_tutorial(self):
        chunk = _make_chunk(
            text="Step 1: Install the package. Step 2: Run the following command to initialize the project.",
        )
        result = enrich_chunks([chunk])
        assert result[0].source_type in ("tutorial", "concept")

    def test_entities_extracted_from_text(self):
        chunk = _make_chunk(
            text="Apache Spark and Delta Lake integrate with Databricks for ETL workflows.",
        )
        result = enrich_chunks([chunk])
        assert isinstance(result[0].extracted_entities, tuple)

    def test_all_chunks_enriched(self):
        chunks = [_make_chunk(text=f"Chunk {i} with meaningful content for testing.") for i in range(5)]
        result = enrich_chunks(chunks)
        assert len(result) == 5
        assert all(c.content_quality_score > 0 for c in result)

    def test_enrichment_does_not_modify_original(self):
        chunk = _make_chunk(text="Original text for testing enrichment pipeline.")
        original_id = chunk.chunk_id
        enrich_chunks([chunk])
        assert chunk.chunk_id == original_id
        assert chunk.content_quality_score == 0.0

    def test_url_based_source_type(self):
        chunk = _make_chunk(
            text="Configure the application using the properties file.",
            url="http://example.com/configuration-guide",
        )
        result = enrich_chunks([chunk])
        assert result[0].source_type in ("configuration", "tutorial", "concept")

    def test_very_short_text_low_quality(self):
        chunk = _make_chunk(text="Hi.")
        result = enrich_chunks([chunk])
        assert result[0].content_quality_score < 0.3
