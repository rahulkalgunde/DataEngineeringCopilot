"""Evaluation suite with mocked embedder — no infra dependency."""

from __future__ import annotations

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.rag_evaluation import (
    AnswerEvaluator,
    EvaluationResult,
    RAGEvaluator,
    RetrievalEvaluator,
)


def _retrieved_chunk(chunk_id: str, text: str, score: float = 0.8) -> RetrievedChunk:
    chunk = DocumentChunk(chunk_id=chunk_id, source_name="test", title="t", url="http://x", text=text)
    return RetrievedChunk(chunk=chunk, distance=1.0 - score, confidence=score)


class TestRetrievalEvaluator:
    def test_perfect_retrieval(self):
        ev = RetrievalEvaluator()
        retrieved = [
            _retrieved_chunk("a", "Spark SQL is a module for data processing."),
            _retrieved_chunk("b", "Delta Lake provides ACID transactions."),
        ]
        relevant_ids = {"a", "b"}
        result = ev.evaluate(retrieved, relevant_ids)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_partial_retrieval(self):
        ev = RetrievalEvaluator()
        retrieved = [
            _retrieved_chunk("a", "Spark SQL module."),
            _retrieved_chunk("c", "Irrelevant text."),
        ]
        relevant_ids = {"a", "b"}
        result = ev.evaluate(retrieved, relevant_ids)
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5

    def test_empty_retrieval(self):
        ev = RetrievalEvaluator()
        result = ev.evaluate([], {"a"})
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_mrr(self):
        ev = RetrievalEvaluator()
        retrieved = [
            _retrieved_chunk("b", "Not first."),
            _retrieved_chunk("a", "First relevant."),
        ]
        result = ev.evaluate(retrieved, {"a"})
        assert result["mrr"] == 0.5  # relevant at rank 2


class TestAnswerEvaluator:
    def test_relevance_score(self):
        ev = AnswerEvaluator()
        score = ev.score_relevance(
            "Spark SQL provides DataFrame API.",
            "Spark SQL is a module for structured data processing.",
        )
        assert 0.0 <= score <= 1.0

    def test_empty_answer_zero_score(self):
        ev = AnswerEvaluator()
        score = ev.score_relevance("", "some context")
        assert score == 0.0


class TestRAGEvaluator:
    def test_full_evaluation(self):
        ev = RAGEvaluator()
        retrieved = [
            _retrieved_chunk("a", "Spark SQL module."),
            _retrieved_chunk("b", "Delta Lake ACID."),
        ]
        result = ev.evaluate(
            query="What is Spark SQL?",
            answer="Spark SQL provides DataFrame API.",
            retrieved_chunks=retrieved,
            relevant_chunk_ids={"a"},
        )
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.retrieval_precision <= 1.0
        assert 0.0 <= result.retrieval_recall <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0
        assert result.overall_score >= 0.0
