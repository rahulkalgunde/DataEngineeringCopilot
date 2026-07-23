"""RAG evaluation suite — no infra dependency, mocked embedder compatible."""

from __future__ import annotations

import re
from dataclasses import dataclass

from data_engineering_copilot.domain.models import RetrievedChunk


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


@dataclass
class EvaluationResult:
    retrieval_precision: float
    retrieval_recall: float
    retrieval_mrr: float
    answer_relevance: float
    overall_score: float


class RetrievalEvaluator:
    """Evaluate retrieval quality against a set of relevant chunk IDs."""

    def evaluate(
        self,
        retrieved: list[RetrievedChunk],
        relevant_ids: set[str],
    ) -> dict[str, float]:
        retrieved_ids = [r.chunk.chunk_id for r in retrieved]
        if not retrieved_ids:
            return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}

        hits = sum(1 for cid in retrieved_ids if cid in relevant_ids)
        precision = hits / len(retrieved_ids)
        recall = hits / len(relevant_ids) if relevant_ids else 0.0

        # MRR: 1/rank of first relevant result
        mrr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant_ids:
                mrr = 1.0 / rank
                break

        return {"precision": precision, "recall": recall, "mrr": mrr}


class AnswerEvaluator:
    """Evaluate answer relevance to context using token overlap."""

    def score_relevance(self, answer: str, context: str) -> float:
        if not answer.strip():
            return 0.0
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)
        if not answer_tokens:
            return 0.0
        overlap = len(answer_tokens & context_tokens)
        return min(1.0, overlap / len(answer_tokens))


class RAGEvaluator:
    """Full RAG pipeline evaluation combining retrieval and answer metrics."""

    def __init__(
        self,
        retrieval_weight: float = 0.6,
        answer_weight: float = 0.4,
    ) -> None:
        self._retrieval_ev = RetrievalEvaluator()
        self._answer_ev = AnswerEvaluator()
        self._retrieval_weight = retrieval_weight
        self._answer_weight = answer_weight

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        relevant_chunk_ids: set[str],
    ) -> EvaluationResult:
        retrieval = self._retrieval_ev.evaluate(retrieved_chunks, relevant_chunk_ids)

        # Answer relevance: score against all retrieved context
        all_context = " ".join(r.chunk.text for r in retrieved_chunks)
        answer_rel = self._answer_ev.score_relevance(answer, all_context)

        overall = retrieval["precision"] * self._retrieval_weight + answer_rel * self._answer_weight

        return EvaluationResult(
            retrieval_precision=retrieval["precision"],
            retrieval_recall=retrieval["recall"],
            retrieval_mrr=retrieval["mrr"],
            answer_relevance=answer_rel,
            overall_score=round(overall, 4),
        )
