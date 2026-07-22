"""Tests for the CrossEncoderReranker service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.reranker import CrossEncoderReranker


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
    return [0.95 - (i * 0.1) for i in range(num_scores)]


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker initialization and behavior."""

    def test_initialization_with_valid_model(self):
        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")

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

            assert reranker.model_name == "cross-encoder/qnli-distilroberta-base"
            assert reranker.model is mock_model

    def test_rerank_happy_path(self):
        test_chunks, query = create_test_chunks(num_chunks=5)
        mock_scores = create_test_scores(5)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_scores
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            reranked_chunks = reranker.rerank(query, test_chunks, top_k=3)

            assert len(reranked_chunks) == 3
            expected_pairs = [[query, chunk.chunk.text] for chunk in test_chunks]
            mock_model.predict.assert_called_once_with(expected_pairs)

    def test_rerank_empty_chunks(self):
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            reranker = CrossEncoderReranker(model_name="test_model")
        result = reranker.rerank("query", [], top_k=10)
        assert result == []

    def test_rerank_model_not_available(self):
        test_chunks, query = create_test_chunks(num_chunks=3)
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            reranker = CrossEncoderReranker(model_name="test_model")
        reranker.model = None

        result = reranker.rerank(query, test_chunks, top_k=2)

        assert len(result) == 2
        assert result[0].chunk.text == test_chunks[0].chunk.text
        assert result[1].chunk.text == test_chunks[1].chunk.text

    def test_rerank_fewer_chunks_than_top_k(self):
        test_chunks, query = create_test_chunks(num_chunks=3)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.7, 0.5]
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")

            result = reranker.rerank(query, test_chunks, top_k=10)
            assert len(result) == 3

            result = reranker.rerank(query, test_chunks, top_k=3)
            assert len(result) == 3

    def test_rerank_top_k_limit(self):
        test_chunks, query = create_test_chunks(num_chunks=10)
        mock_scores = create_test_scores(10)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_scores
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            result = reranker.rerank(query, test_chunks, top_k=3)

            assert len(result) == 3
            for i in range(len(result) - 1):
                assert result[i].confidence >= result[i + 1].confidence

    def test_rerank_model_prediction_error(self):
        test_chunks, query = create_test_chunks(num_chunks=3)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception("Model prediction failed")
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            result = reranker.rerank(query, test_chunks, top_k=2)

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

    def test_empty_chunks_no_model_prediction(self):
        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            result = reranker.rerank("query", [], top_k=5)

            assert result == []
            mock_model.predict.assert_not_called()

    def test_large_batch_performance(self):
        test_chunks, query = create_test_chunks(num_chunks=100)
        mock_scores = create_test_scores(num_scores=100)

        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_scores
            mock_cross_encoder.return_value = mock_model

            reranker = CrossEncoderReranker(model_name="test_model")
            result = reranker.rerank(query, test_chunks, top_k=10)

            assert len(result) == 10
            assert result[0].confidence >= result[9].confidence


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
