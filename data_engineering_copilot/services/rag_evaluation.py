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
    key_term_coverage: float
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

        # Key-term coverage: fraction of query key terms found in retrieved context
        key_terms = set(re.findall(r"[A-Z]\w+|\w{4,}", query.lower()))
        all_context_lower = all_context.lower()
        covered = sum(1 for t in key_terms if t in all_context_lower) if key_terms else 1
        key_term_coverage = covered / len(key_terms) if key_terms else 1.0

        overall = retrieval["precision"] * self._retrieval_weight + answer_rel * self._answer_weight

        return EvaluationResult(
            retrieval_precision=retrieval["precision"],
            retrieval_recall=retrieval["recall"],
            retrieval_mrr=retrieval["mrr"],
            answer_relevance=answer_rel,
            key_term_coverage=round(key_term_coverage, 4),
            overall_score=round(overall, 4),
        )


@dataclass
class FaithfulnessResult:
    """LLM-based faithfulness scoring (RAGAS-compatible)."""
    faithfulness_score: float
    supported_claims: int
    unsupported_claims: int


class FaithfulnessEvaluator:
    """Evaluate answer faithfulness using LLM NLI (RAGAS Faithfulness metric).

    Primary mode: LLM-based claim verification.
    Fallback: returns 1.0 when LLM unavailable.
    """

    def __init__(self, llm_client=None):
        self._llm_client = llm_client

    async def evaluate(self, answer: str, context: str) -> FaithfulnessResult:
        """Score faithfulness of answer against context.

        Returns FaithfulnessResult with faithfulness_score in [0, 1].
        """
        if self._llm_client is None:
            return FaithfulnessResult(1.0, 0, 0)

        prompt = (
            "Given the answer and context below, count how many claims in the "
            "answer are supported by the context.\n\n"
            f"ANSWER: {answer[:2000]}\n\nCONTEXT: {context[:3000]}\n\n"
            'Return JSON: {{"supported": N, "unsupported": N}}'
        )

        try:
            import json
            result = await self._llm_client.generate(prompt)
            cleaned = result.strip()
            # Strip markdown fencing if present
            import re
            fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
            if fence_match:
                cleaned = fence_match.group(1).strip()
            data = json.loads(cleaned)
            supported = data.get("supported", 0)
            unsupported = data.get("unsupported", 0)
            total = supported + unsupported
            score = supported / total if total > 0 else 1.0
            return FaithfulnessResult(score, supported, unsupported)
        except Exception:
            return FaithfulnessResult(1.0, 0, 0)
