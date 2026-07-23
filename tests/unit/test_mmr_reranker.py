"""Tests for MMR diversity reranking."""

from __future__ import annotations

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.mmr_reranker import mmr_rerank


def _chunk(cid: str, text: str, score: float = 0.8) -> RetrievedChunk:
    chunk = DocumentChunk(chunk_id=cid, source_name="s", title="t", url="http://x", text=text)
    return RetrievedChunk(chunk=chunk, distance=1.0 - score, confidence=score)


class TestMMRRerank:
    def test_empty_returns_empty(self):
        assert mmr_rerank([], top_k=5) == []

    def test_single_chunk_returned(self):
        chunks = [_chunk("a", "Spark SQL")]
        result = mmr_rerank(chunks, top_k=5)
        assert len(result) == 1

    def test_top_k_limits_output(self):
        chunks = [_chunk(str(i), f"Chunk {i} text") for i in range(10)]
        result = mmr_rerank(chunks, top_k=3)
        assert len(result) == 3

    def test_most_confident_first(self):
        chunks = [
            _chunk("a", "Unrelated text", score=0.3),
            _chunk("b", "Spark SQL module", score=0.9),
        ]
        result = mmr_rerank(chunks, top_k=2)
        assert result[0].chunk.chunk_id == "b"

    def test_diversity_penalty(self):
        chunks = [
            _chunk("a", "Spark SQL provides DataFrame API for queries.", score=0.9),
            _chunk("b", "Spark SQL provides DataFrame API for queries.", score=0.89),
            _chunk("c", "Kafka Streams processes real-time events.", score=0.7),
        ]
        result = mmr_rerank(chunks, top_k=3, lambda_param=0.5)
        assert result[0].chunk.chunk_id == "a"
        # Second chunk should be the diverse one (Kafka), not the duplicate
        assert result[1].chunk.chunk_id == "c"
        assert result[2].chunk.chunk_id == "b"

    def test_lambda_1_is_pure_relevance(self):
        chunks = [
            _chunk("a", "Spark SQL", score=0.9),
            _chunk("b", "Kafka Streams", score=0.7),
        ]
        result = mmr_rerank(chunks, top_k=2, lambda_param=1.0)
        assert result[0].chunk.chunk_id == "a"
        assert result[1].chunk.chunk_id == "b"
