"""Tests for ContextCompressor — redundancy elimination."""

from __future__ import annotations

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.services.context_compression import ContextCompressor


def _chunk(cid: str, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=cid, source_name="s", title="t", url="http://x", text=text)


class TestContextCompressor:
    def test_empty_input_returns_empty(self):
        cc = ContextCompressor(enabled=True, similarity_threshold=0.9)
        assert cc.compress([], "query") == []

    def test_disabled_returns_original(self):
        cc = ContextCompressor(enabled=False, similarity_threshold=0.9)
        chunks = [_chunk("a", "Alpha text"), _chunk("b", "Beta text")]
        assert cc.compress(chunks, "query") == chunks

    def test_deduplicates_identical_text(self):
        cc = ContextCompressor(enabled=True, similarity_threshold=0.9)
        chunks = [
            _chunk("a", "Spark SQL is great for data processing."),
            _chunk("b", "Spark SQL is great for data processing."),
        ]
        result = cc.compress(chunks, "query")
        assert len(result) == 1

    def test_keeps_distinct_chunks(self):
        cc = ContextCompressor(enabled=True, similarity_threshold=0.9)
        chunks = [
            _chunk("a", "Spark SQL module for structured data."),
            _chunk("b", "Kafka Streams processes real-time events."),
        ]
        result = cc.compress(chunks, "query")
        assert len(result) == 2

    def test_relevance_boosts_ranking(self):
        cc = ContextCompressor(enabled=True, similarity_threshold=0.9)
        chunks = [
            _chunk("a", "Random unrelated text about something else entirely."),
            _chunk("b", "Spark SQL provides DataFrame API for queries."),
        ]
        result = cc.compress(chunks, "Spark SQL DataFrame")
        assert result[0].chunk_id == "b"

    def test_max_chunks_respected(self):
        cc = ContextCompressor(enabled=True, similarity_threshold=0.9, max_chunks=2)
        chunks = [_chunk(str(i), f"Distinct text {i} about topic {i}.") for i in range(10)]
        result = cc.compress(chunks, "query")
        assert len(result) <= 2

    def test_returns_list_of_document_chunks(self):
        cc = ContextCompressor(enabled=True, similarity_threshold=0.9)
        chunks = [_chunk("x", "Hello world test text.")]
        result = cc.compress(chunks, "hello")
        assert all(isinstance(c, DocumentChunk) for c in result)
