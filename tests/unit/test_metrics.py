"""Tests for the MetricsCollector service."""

from __future__ import annotations

from data_engineering_copilot.domain.models import Answer, DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.metrics import (
    AnswerMetrics,
    MetricsCollector,
    QueryMetrics,
    RetrievalMetrics,
)


def create_test_retrieved_chunks(count=5, base_confidence=0.8):
    chunks = []
    for i in range(count):
        confidence = base_confidence + (i * 0.02)
        chunk = DocumentChunk(
            chunk_id=f"chunk_{i}",
            source_name=f"source_{i}",
            title=f"Title {i}",
            url=f"http://example.com/chunk{i}",
            text=f"This is chunk {i} with some content that is meaningful for testing.",
            content_hash=f"hash_{i}",
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            distance=1.0 - confidence,
            confidence=confidence,
        )
        chunks.append(retrieved)
    return chunks


def create_test_answer(text="Sample answer content here.", confidence=0.8, source_count=2):
    sources = []
    for i in range(source_count):
        sources.append(
            DocumentChunk(
                chunk_id=f"source_chunk_{i}",
                source_name=f"relevant_source_{i}",
                title=f"Relevant Source {i}",
                url=f"http://example.com/source{i}",
                text=f"This is source {i} with relevant information.",
                content_hash=f"source_hash_{i}",
            )
        )
    return Answer(text=text, sources=tuple(sources), confidence=confidence)


class TestMetricsCollector:
    def test_initialization(self):
        collector = MetricsCollector()
        assert collector.queries == []

    def test_classify_query_difficulty_easy(self):
        collector = MetricsCollector()
        easy_queries = [
            "What is Python?",
            "Explain SQL.",
        ]
        for query in easy_queries:
            assert collector.classify_query_difficulty(query) == "easy"

    def test_classify_query_difficulty_medium(self):
        collector = MetricsCollector()
        medium_queries = [
            "Explain SQL joins. What is the difference between inner and left join?",
            "What are the main components of a data pipeline?",
            "How does neural networks work in deep learning?",
        ]
        for query in medium_queries:
            assert collector.classify_query_difficulty(query) == "medium"

    def test_classify_query_difficulty_hard(self):
        collector = MetricsCollector()
        hard_queries = [
            "How does neural networks work in deep learning? Compare with traditional ML. What are the advantages?",
        ]
        for query in hard_queries:
            assert collector.classify_query_difficulty(query) == "hard"

    def test_retrieval_metrics_empty_chunks(self):
        collector = MetricsCollector()
        metrics = collector.compute_retrieval_metrics("test query", [])
        assert metrics.retrieved_count == 0
        assert metrics.top_confidence == 0.0

    def test_retrieval_metrics_with_data(self):
        collector = MetricsCollector()
        chunks = create_test_retrieved_chunks(count=5, base_confidence=0.7)
        metrics = collector.compute_retrieval_metrics("test query", chunks)
        assert metrics.retrieved_count == 5
        assert metrics.top_confidence == 0.7
        assert metrics.mrr == 1.0

    def test_answer_metrics_structure(self):
        collector = MetricsCollector()
        answer = create_test_answer(
            text="Answer: Machine learning is a field of AI. Key Points:\n- Supervised learning",
            confidence=0.85,
        )
        metrics = collector.compute_answer_metrics("What is ML?", answer)
        assert metrics.has_key_sections is True
        assert metrics.has_uncertainty_markers is False
        assert metrics.source_count == 2

    def test_answer_metrics_uncertainty(self):
        collector = MetricsCollector()
        answer = create_test_answer(
            text="Answer: Machine learning is a field. Not covered in this context.",
            confidence=0.6,
            source_count=1,
        )
        metrics = collector.compute_answer_metrics("What is ML?", answer)
        assert metrics.has_uncertainty_markers is True
        assert metrics.source_count == 1

    def test_answer_metrics_no_sources(self):
        collector = MetricsCollector()
        answer = create_test_answer(text="Machine learning is not covered.", confidence=0.0, source_count=0)
        metrics = collector.compute_answer_metrics("What is ML?", answer)
        assert metrics.source_count == 0

    def test_record_query_complete(self):
        collector = MetricsCollector()
        chunks = create_test_retrieved_chunks(count=3)
        answer = create_test_answer()
        qm = collector.record_query("test query", chunks, answer, was_answered=True)
        assert len(collector.queries) == 1
        assert qm.was_answered is True
        assert qm.answer_metrics is not None
        assert qm.retrieval_metrics is not None

    def test_record_query_no_answer(self):
        collector = MetricsCollector()
        chunks = create_test_retrieved_chunks(count=2)
        qm = collector.record_query("test query", chunks, was_answered=False)
        assert qm.was_answered is False
        assert qm.answer_metrics is None

    def test_record_query_none_answer(self):
        collector = MetricsCollector()
        chunks = create_test_retrieved_chunks(count=1)
        qm = collector.record_query("test query", chunks, answer=None, was_answered=False)
        assert qm.answer_metrics is None
        assert qm.confidence_score == 0.0

    def test_session_summary_empty(self):
        collector = MetricsCollector()
        summary = collector.get_session_summary()
        assert summary["total_queries"] == 0
        assert summary["answer_rate"] == 0.0

    def test_session_summary_with_data(self):
        collector = MetricsCollector()
        chunks = create_test_retrieved_chunks(count=3)
        for i in range(3):
            answer = create_test_answer() if i < 2 else None
            collector.record_query(f"query {i}", chunks, answer, was_answered=answer is not None)
        summary = collector.get_session_summary()
        assert summary["total_queries"] == 3
        assert summary["answered_queries"] == 2
        assert "avg_mrr" in summary

    def test_session_summary_by_difficulty(self):
        collector = MetricsCollector()
        queries = [
            "What is Python?",
            "Explain SQL joins. What is the difference between inner and left join?",
            "How does neural networks work in deep learning? Compare with traditional ML. What are the advantages?",
        ]
        for query_text in queries:
            collector.record_query(query_text, create_test_retrieved_chunks(), was_answered=True)
        summary = collector.get_session_summary()
        bd = summary["by_difficulty"]
        assert bd["easy"]["count"] == 1
        assert bd["medium"]["count"] == 1
        assert bd["hard"]["count"] == 1

    def test_str_retrieval_metrics(self):
        rm = RetrievalMetrics(
            query="What is machine learning? This is a longer query to test truncation.",
            retrieved_count=10,
            top_confidence=0.95,
            mrr=0.8,
            precision_at_3=0.7,
            precision_at_5=0.6,
        )
        s = str(rm)
        assert "RetrievalMetrics" in s
        assert "retrieved=10" in s
        assert "mrr=0.800" in s

    def test_str_answer_metrics(self):
        am = AnswerMetrics(
            question="What is ML?",
            answer_length=25,
            answer_chars=180,
            has_key_sections=True,
            has_uncertainty_markers=False,
            source_count=3,
        )
        s = str(am)
        assert "AnswerMetrics" in s
        assert "len=25" in s
        assert "sections=True" in s

    def test_str_query_metrics(self):
        qm = QueryMetrics(
            query="What is the meaning of life?",
            query_length=7,
            query_difficulty="hard",
            retrieval_metrics=RetrievalMetrics(query="sub", retrieved_count=5, top_confidence=0.8),
            answer_metrics=AnswerMetrics(
                question="test",
                answer_length=10,
                answer_chars=100,
                has_key_sections=False,
                has_uncertainty_markers=True,
                source_count=1,
            ),
            confidence_score=0.75,
            was_answered=True,
        )
        s = str(qm)
        assert "QueryMetrics" in s
        assert "difficulty=hard" in s
        assert "confidence=0.750" in s

    def test_multiple_recordings(self):
        collector = MetricsCollector()
        for i in range(5):
            chunks = create_test_retrieved_chunks(count=i + 1)
            answer = create_test_answer() if i % 2 == 0 else None
            qm = collector.record_query(f"query {i}", chunks, answer, was_answered=answer is not None)
            assert qm.retrieval_metrics.retrieved_count == i + 1
        assert len(collector.queries) == 5

    def test_data_structure_integrity(self):
        collector = MetricsCollector()
        chunks = create_test_retrieved_chunks(count=2)
        answer = create_test_answer()
        qm = collector.record_query("test", chunks, answer, True)
        assert isinstance(qm, QueryMetrics)
        assert isinstance(qm.retrieval_metrics, RetrievalMetrics)
        assert isinstance(qm.answer_metrics, AnswerMetrics)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
