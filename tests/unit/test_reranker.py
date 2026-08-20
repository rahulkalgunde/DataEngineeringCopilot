"""Tests for the CrossEncoderReranker service."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.async_rag import _rerank_pool_size
from data_engineering_copilot.services.reranker import CrossEncoderReranker, _truncate_doc_for_rerank


def create_test_chunks(num_chunks=5):
    query = "What is machine learning?"
    chunks = []
    for i in range(num_chunks):
        text = f"This is chunk {i} containing relevant information about machine learning concepts."
        relevance_score = 1.0 - (i * 0.2)
        chunks.append(
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=f"chunk_{i}",
                    source_name="test_source",
                    title=f"Chunk {i}",
                    url=f"http://example.com/chunk{i}",
                    text=text,
                    content_hash=f"hash_{i}",
                ),
                distance=1.0 - relevance_score,
                confidence=relevance_score,
            )
        )
    return chunks, query


def create_test_scores(num_scores):
    """Return realistic cross-encoder logits (raw, unnormalized)."""
    return [2.0 - (i * 0.8) for i in range(num_scores)]


class TestMMR:
    """Lexical MMR diversity selection (no model required)."""

    def test_mmr_keeps_lexically_similar_chunks_with_distinct_facts(self):
        from data_engineering_copilot.services.reranker import mmr_rerank

        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="fact-a",
                    source_name="spark",
                    title="Fact A",
                    url="http://example.com/a",
                    text="filter a DataFrame with isNotNull to keep rows where a column has a value",
                    content_hash="hash_a",
                ),
                distance=0.1,
                confidence=0.93,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="fact-b",
                    source_name="sql-guide",
                    title="Fact B",
                    url="http://example.com/b",
                    text="filter a DataFrame with isNotNull to drop null rows in the column",
                    content_hash="hash_b",
                ),
                distance=0.12,
                confidence=0.91,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="off-topic",
                    source_name="other",
                    title="Off topic",
                    url="http://example.com/o",
                    text="completely unrelated content about kubernetes schedulers",
                    content_hash="hash_o",
                ),
                distance=0.8,
                confidence=0.2,
            ),
        ]

        selected = mmr_rerank(chunks, top_k=2, lambda_param=0.5)

        assert len(selected) == 2
        assert {c.chunk.chunk_id for c in selected} == {"fact-a", "fact-b"}
        assert selected[0].chunk.chunk_id == "fact-a"

    def test_mmr_removes_redundant_duplicate(self):
        from data_engineering_copilot.services.reranker import mmr_rerank

        text = "spark.sql.shuffle.partitions controls the number of partitions"
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=f"dup-{i}",
                    source_name=f"source-{i}",
                    title=f"Dup {i}",
                    url=f"http://example.com/dup{i}",
                    text=text,
                    content_hash=f"hash_dup{i}",
                ),
                distance=1.0 - (0.9 - i * 0.02),
                confidence=0.9 - i * 0.02,
            )
            for i in range(3)
        ] + [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="unique",
                    source_name="spark",
                    title="Unique",
                    url="http://example.com/u",
                    text="adaptive query execution rewrites the physical plan at runtime",
                    content_hash="hash_u",
                ),
                distance=0.2,
                confidence=0.8,
            )
        ]

        selected = mmr_rerank(chunks, top_k=2, lambda_param=0.5)

        assert len(selected) == 2
        assert selected[-1].chunk.chunk_id == "unique"


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker initialization and behavior."""

    def test_initialization_with_valid_model(self):
        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            reranker._init_sync()

            assert reranker.model is mock_model
            assert reranker.model_name == "test_model"

    def test_initialization_with_import_error(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            reranker = CrossEncoderReranker(model_name="test_model")

            assert reranker.model is None
            assert reranker.model_name == "test_model"

    def test_initialization_default_model(self):
        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker()

            assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
            assert reranker.model is None

            asyncio.run(reranker.initialize())

            assert reranker.model is mock_model

    async def test_rerank_happy_path(self):
        test_chunks, query = create_test_chunks(num_chunks=5)
        mock_scores = create_test_scores(5)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_scores
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            reranked_chunks = await reranker.rerank(query, test_chunks, top_k=3)

            assert len(reranked_chunks) == 3
            expected_pairs = [[query, chunk.chunk.text] for chunk in test_chunks]
            mock_model.predict.assert_called_once_with(expected_pairs)

    async def test_rerank_empty_chunks(self):
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            reranker = CrossEncoderReranker(model_name="test_model")
        result = await reranker.rerank("query", [], top_k=10)
        assert result == []

    async def test_rerank_model_not_available(self):
        test_chunks, query = create_test_chunks(num_chunks=3)
        reranker = CrossEncoderReranker(model_name="test_model")
        reranker.model = None

        with patch.object(CrossEncoderReranker, "_init_sync", return_value=None):
            result = await reranker.rerank(query, test_chunks, top_k=2)

        assert len(result) == 2
        assert result[0].chunk.text == test_chunks[0].chunk.text
        assert result[1].chunk.text == test_chunks[1].chunk.text

    async def test_rerank_fewer_chunks_than_top_k(self):
        test_chunks, query = create_test_chunks(num_chunks=3)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.7, 0.5]
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")

            result = await reranker.rerank(query, test_chunks, top_k=10)
            assert len(result) == 3

            result = await reranker.rerank(query, test_chunks, top_k=3)
            assert len(result) == 3

    async def test_rerank_top_k_limit(self):
        test_chunks, query = create_test_chunks(num_chunks=10)
        mock_scores = create_test_scores(10)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_scores
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            result = await reranker.rerank(query, test_chunks, top_k=3)

            assert len(result) == 3
            for i in range(len(result) - 1):
                assert result[i].confidence >= result[i + 1].confidence
            for chunk in result:
                assert 0.0 <= chunk.confidence <= 1.0

    async def test_rerank_negative_logits_normalized(self):
        """Verify raw negative logits are sigmoid-then-min-max normalized to [0, 1]."""
        test_chunks, query = create_test_chunks(num_chunks=5)
        # Realistic ms-marco-MiniLM-L-6-v2 logits (can be negative)
        negative_logits = [-3.0, -1.5, 0.2, 1.8, 4.0]

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = negative_logits
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            # top_k < num_chunks to force actual reranking
            result = await reranker.rerank(query, test_chunks, top_k=3)

            assert len(result) == 3
            # All confidence values must be in [0, 1]
            for chunk in result:
                assert 0.0 <= chunk.confidence <= 1.0, f"confidence={chunk.confidence} out of range"
            # Min-max normalized within the pool: the highest logit (4.0) → 1.0.
            assert result[0].confidence == pytest.approx(1.0)
            # Verify descending order
            for i in range(len(result) - 1):
                assert result[i].confidence >= result[i + 1].confidence

    async def test_rerank_model_prediction_error(self):
        test_chunks, query = create_test_chunks(num_chunks=3)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception("Model prediction failed")
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            result = await reranker.rerank(query, test_chunks, top_k=2)

            assert len(result) == 2
            assert result[0].chunk.text == test_chunks[0].chunk.text

    def test_is_available_true(self):
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            reranker = CrossEncoderReranker(model_name="test_model")
        reranker.model = MagicMock()
        assert reranker.is_available() is True

    def test_is_available_false(self):
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            reranker = CrossEncoderReranker(model_name="test_model")
        reranker.model = None
        assert reranker.is_available() is False

    async def test_empty_chunks_no_model_prediction(self):
        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            result = await reranker.rerank("query", [], top_k=5)

            assert result == []
            mock_model.predict.assert_not_called()

    async def test_large_batch_performance(self):
        test_chunks, query = create_test_chunks(num_chunks=100)
        mock_scores = create_test_scores(num_scores=100)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_scores
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            result = await reranker.rerank(query, test_chunks, top_k=10)

            assert len(result) == 10
            assert result[0].confidence >= result[9].confidence
            for chunk in result:
                assert 0.0 <= chunk.confidence <= 1.0


def test_min_max_normalize():
    """Uniform scaling: different raw score distributions map to [0, 1]."""
    from data_engineering_copilot.services.reranker import _min_max_normalize

    assert _min_max_normalize([0.95, 0.8, 0.2]) == pytest.approx([1.0, 0.8, 0.0])
    assert _min_max_normalize([3, 1, 2]) == pytest.approx([1.0, 0.0, 0.5])
    # Degenerate pools pass through unchanged.
    assert _min_max_normalize([0.5, 0.5]) == [0.5, 0.5]
    assert _min_max_normalize([0.0]) == [0.0]
    assert _min_max_normalize([]) == []


class TestDocumentTruncation:
    def test_short_doc_unchanged(self):
        assert _truncate_doc_for_rerank("Short text", 2000) == "Short text"

    def test_truncate_at_paragraph(self):
        para1 = "A" * 1200
        text = para1 + "\n\n" + "B" * 800
        assert _truncate_doc_for_rerank(text, 1500) == para1

    def test_truncate_at_newline(self):
        line1 = "X" * 800
        text = line1 + "\n" + "Y" * 800
        assert _truncate_doc_for_rerank(text, 1200) == line1

    def test_hard_truncation(self):
        assert len(_truncate_doc_for_rerank("A" * 3000, 2000)) <= 2000

    def test_code_fence_integrity(self):
        code = "```python\nprint('hello')\n```\n"
        text = code + "D" * 3000
        result = _truncate_doc_for_rerank(text, 100)
        assert result.count("```") % 2 == 0

    def test_exact_limit(self):
        text = "X" * 2000
        assert _truncate_doc_for_rerank(text, 2000) == text


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


class TestRerankPoolSize:
    def test_default_formula(self):
        assert _rerank_pool_size(50, 30) == 240

    def test_configured_overrides(self):
        assert _rerank_pool_size(50, 30, configured_pool=100) == 100

    def test_zero_uses_formula(self):
        assert _rerank_pool_size(50, 30, configured_pool=0) == 240

    def test_small_top_k(self):
        assert _rerank_pool_size(10, 5) == 40
