"""Quality metrics and evaluation infrastructure for RAG system.

This module tracks and computes metrics for:
- Retrieval quality (MRR, precision@k, recall@k)
- Answer quality (length, structure validation)
- Confidence calibration (threshold analysis)
- Query difficulty classification
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from data_engineering_copilot.domain.models import Answer, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""

    query: str
    retrieved_count: int
    top_confidence: float
    mrr: float = 0.0  # Mean Reciprocal Rank
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0

    def __str__(self) -> str:
        return (
            f"RetrievalMetrics(query={self.query[:30]}..., "
            f"retrieved={self.retrieved_count}, "
            f"top_conf={self.top_confidence:.3f}, "
            f"mrr={self.mrr:.3f}, "
            f"p@3={self.precision_at_3:.3f}, "
            f"p@5={self.precision_at_5:.3f})"
        )


@dataclass
class AnswerMetrics:
    """Metrics for answer quality."""

    question: str
    answer_length: int
    answer_chars: int
    has_key_sections: bool  # Checks for "Answer:", "Key Points:", etc.
    has_uncertainty_markers: bool  # Checks for "not covered", "not clearly", etc.
    source_count: int

    def __str__(self) -> str:
        return (
            f"AnswerMetrics(question={self.question[:30]}..., "
            f"len={self.answer_length} words, "
            f"sections={self.has_key_sections}, "
            f"uncertainty={self.has_uncertainty_markers}, "
            f"sources={self.source_count})"
        )


@dataclass
class QueryMetrics:
    """Comprehensive metrics for a single query."""

    query: str
    timestamp: datetime = field(default_factory=datetime.now)
    query_length: int = 0
    query_difficulty: str = "medium"  # easy, medium, hard
    retrieval_metrics: RetrievalMetrics | None = None
    answer_metrics: AnswerMetrics | None = None
    confidence_score: float = 0.0
    was_answered: bool = False

    def __str__(self) -> str:
        return (
            f"QueryMetrics(query={self.query[:30]}..., "
            f"difficulty={self.query_difficulty}, "
            f"answered={self.was_answered}, "
            f"confidence={self.confidence_score:.3f})"
        )


class MetricsCollector:
    """Collects and analyzes RAG system metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.queries: list[QueryMetrics] = []

    def classify_query_difficulty(self, query: str) -> str:
        """Classify query difficulty based on characteristics.

        Args:
            query: User question

        Returns:
            Difficulty level: "easy", "medium", or "hard"
        """
        # Simple heuristic-based classification
        word_count = len(query.split())
        has_comparison = any(word in query.lower() for word in ["vs", "compare", "difference", "versus"])
        has_multiple_concepts = query.count(" and ") > 0 or query.count(",") > 1

        # Hard: long queries with comparisons or multiple concepts
        if word_count > 15 and (has_comparison or has_multiple_concepts):
            return "hard"
        # Easy: short, simple factual questions
        elif word_count < 8 and not (has_comparison or has_multiple_concepts):
            return "easy"
        # Medium: everything else
        else:
            return "medium"

    def compute_retrieval_metrics(
        self, query: str, retrieved_chunks: list[RetrievedChunk], top_k: int = 5
    ) -> RetrievalMetrics:
        """Compute retrieval quality metrics.

        Args:
            query: User question
            retrieved_chunks: Retrieved and ranked chunks
            top_k: k value for precision@k

        Returns:
            RetrievalMetrics object with computed metrics
        """
        if not retrieved_chunks:
            return RetrievalMetrics(query=query, retrieved_count=0, top_confidence=0.0)

        # MRR: Mean Reciprocal Rank (average of 1/rank for relevant results)
        # Simplified: assume first chunk is most relevant
        mrr = 1.0 / 1  # First chunk rank is 1

        # Precision@k: fraction of top-k results with confidence > threshold
        confidence_threshold = 0.45
        top_k_chunks = retrieved_chunks[:top_k]
        relevant_count = sum(1 for c in top_k_chunks if c.confidence >= confidence_threshold)

        precision_at_3 = relevant_count / 3 if len(retrieved_chunks) >= 3 else relevant_count / len(retrieved_chunks)
        precision_at_5 = (
            relevant_count / top_k if len(retrieved_chunks) >= top_k else relevant_count / len(retrieved_chunks)
        )

        return RetrievalMetrics(
            query=query,
            retrieved_count=len(retrieved_chunks),
            top_confidence=retrieved_chunks[0].confidence,
            mrr=mrr,
            precision_at_3=precision_at_3,
            precision_at_5=precision_at_5,
        )

    def compute_answer_metrics(self, question: str, answer: Answer) -> AnswerMetrics:
        """Compute answer quality metrics.

        Args:
            question: User question
            answer: Generated answer

        Returns:
            AnswerMetrics object with computed metrics
        """
        answer_text = answer.text

        # Length metrics
        word_count = len(answer_text.split())
        char_count = len(answer_text)

        # Section detection: look for structured sections
        has_sections = any(
            marker in answer_text for marker in ["Answer:", "Key Points:", "Important Notes:", "Not Covered:", "##"]
        )

        # Uncertainty markers: indicators of transparent limitations
        has_uncertainty = any(
            marker in answer_text.lower()
            for marker in [
                "not covered",
                "not clearly",
                "not addressed",
                "unclear",
                "limited information",
                "not specified",
                "documentation does not",
            ]
        )

        # Source count
        source_count = len(answer.sources) if answer.sources else 0

        return AnswerMetrics(
            question=question,
            answer_length=word_count,
            answer_chars=char_count,
            has_key_sections=has_sections,
            has_uncertainty_markers=has_uncertainty,
            source_count=source_count,
        )

    def record_query(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        answer: Answer | None = None,
        was_answered: bool = False,
    ) -> QueryMetrics:
        """Record metrics for a complete query cycle.

        Args:
            query: User question
            retrieved_chunks: Retrieved chunks
            answer: Generated answer (if any)
            was_answered: Whether an answer was successfully generated

        Returns:
            QueryMetrics object with all computed metrics
        """
        # Query metrics
        difficulty = self.classify_query_difficulty(query)

        # Retrieval metrics
        retrieval_metrics = self.compute_retrieval_metrics(query, retrieved_chunks)

        # Answer metrics (if available)
        answer_metrics = None
        if answer and was_answered:
            answer_metrics = self.compute_answer_metrics(query, answer)

        # Create comprehensive query metrics
        query_metrics = QueryMetrics(
            query=query,
            query_length=len(query.split()),
            query_difficulty=difficulty,
            retrieval_metrics=retrieval_metrics,
            answer_metrics=answer_metrics,
            confidence_score=answer.confidence if answer else 0.0,
            was_answered=was_answered,
        )

        self.queries.append(query_metrics)

        logger.info("Recorded query metrics: %s", query_metrics)

        return query_metrics

    def get_session_summary(self) -> dict:
        """Get summary statistics for current session.

        Returns:
            Dictionary with session-level metrics
        """
        if not self.queries:
            return {
                "total_queries": 0,
                "answered_queries": 0,
                "answer_rate": 0.0,
                "avg_retrieval_quality": 0.0,
                "avg_answer_length": 0,
            }

        answered = sum(1 for q in self.queries if q.was_answered)
        avg_mrr = sum(q.retrieval_metrics.mrr for q in self.queries if q.retrieval_metrics) / len(self.queries)
        avg_answer_len = sum(q.answer_metrics.answer_length for q in self.queries if q.answer_metrics) / max(
            1, answered
        )

        return {
            "total_queries": len(self.queries),
            "answered_queries": answered,
            "answer_rate": answered / len(self.queries) if self.queries else 0.0,
            "avg_mrr": avg_mrr,
            "avg_answer_length": int(avg_answer_len),
            "by_difficulty": self._summary_by_difficulty(),
        }

    def _summary_by_difficulty(self) -> dict:
        """Breakdown of metrics by query difficulty."""
        by_diff = {"easy": [], "medium": [], "hard": []}
        for q in self.queries:
            by_diff[q.query_difficulty].append(q)

        return {
            "easy": {
                "count": len(by_diff["easy"]),
                "answer_rate": sum(1 for q in by_diff["easy"] if q.was_answered) / max(1, len(by_diff["easy"])),
            },
            "medium": {
                "count": len(by_diff["medium"]),
                "answer_rate": sum(1 for q in by_diff["medium"] if q.was_answered) / max(1, len(by_diff["medium"])),
            },
            "hard": {
                "count": len(by_diff["hard"]),
                "answer_rate": sum(1 for q in by_diff["hard"] if q.was_answered) / max(1, len(by_diff["hard"])),
            },
        }
