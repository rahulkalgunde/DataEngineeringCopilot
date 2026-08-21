"""Lexical char-trigram MaxSim reranker.

A deterministic proxy for late-interaction scoring: per-query-token best
char-3gram overlap against each document token, averaged and min-max
normalized. NOT neural late-interaction — no token embeddings, no PLAID-class
optimizations. Kept behind ``reranker_type="colbert"`` for backward
compatibility; for true late interaction see the deferred experiment in
docs/research/rag_best_practices_comparison_2026-08-21.md.
"""

from __future__ import annotations

import logging
import re

from data_engineering_copilot.domain.models import RetrievedChunk
from data_engineering_copilot.services.reranker import _min_max_normalize, mmr_rerank

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r"\w+", text.lower())


def _char_ngram_overlap(query_tokens: list[str], doc_tokens: list[str], n: int = 3) -> float:
    """Compute average MaxSim-like overlap between query and document tokens.

    For each query token, find the maximum char-ngram overlap score with any
    document token. Average across query tokens.

    This is a lightweight proxy for ColBERT's late-interaction scoring that
    avoids loading neural models.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    def _ngrams(token: str, n: int) -> set[str]:
        if len(token) < n:
            return {token}
        return {token[i : i + n] for i in range(len(token) - n + 1)}

    query_ngrams = [_ngrams(t, n) for t in query_tokens]
    doc_ngrams = [_ngrams(t, n) for t in doc_tokens]

    total = 0.0
    for q_ng in query_ngrams:
        if not q_ng:
            continue
        best = 0.0
        for d_ng in doc_ngrams:
            if not d_ng:
                continue
            overlap = len(q_ng & d_ng)
            score = overlap / max(len(q_ng), len(d_ng))
            if score > best:
                best = score
        total += best

    return total / len(query_tokens)


class LexicalNgramReranker:
    """Char-trigram MaxSim proxy reranker.

    A lightweight lexical approximation of late-interaction scoring — this is
    a proxy, not neural late-interaction: no token embeddings are produced and
    ``model_name`` is accepted only for interface compatibility.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        max_query_tokens: int = 32,
        max_doc_tokens: int = 256,
    ) -> None:
        """Initialize the ColBERT late-interaction reranker.

        Args:
            model_name: Reserved for future model selection (currently unused
                by the char-ngram proxy implementation but accepted by the
                factory for interface consistency with neural rerankers).
            max_query_tokens: Maximum query tokens to consider.
            max_doc_tokens: Maximum document tokens to consider.
        """
        self.model_name = model_name
        self._max_query_tokens = max_query_tokens
        self._max_doc_tokens = max_doc_tokens

    def is_available(self) -> bool:
        return True  # Always available — no model loading needed

    async def initialize(self) -> None:
        pass  # No initialization needed

    async def close(self) -> None:
        pass  # No cleanup needed

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using ColBERT-style MaxSim scoring.

        Args:
            query: The user question.
            chunks: Retrieved chunks to rerank.
            top_k: Number of top results to return.

        Returns:
            List of top_k reranked chunks sorted by score.
        """
        if not chunks:
            return []

        if len(chunks) <= top_k:
            return chunks

        query_tokens = _tokenize(query)[: self._max_query_tokens]

        scores = []
        for chunk in chunks:
            doc_tokens = _tokenize(chunk.chunk.text)[: self._max_doc_tokens]
            score = _char_ngram_overlap(query_tokens, doc_tokens)
            scores.append(score)

        scores = _min_max_normalize(scores)

        scored = sorted(zip(chunks, scores, strict=False), key=lambda x: x[1], reverse=True)

        reranked = [
            RetrievedChunk(
                chunk=chunk.chunk,
                distance=1.0 - score,
                confidence=score,
            )
            for chunk, score in scored[:top_k]
        ]

        logger.info("colbert_rerank chunks_in=%d chunks_out=%d", len(chunks), len(reranked))
        return reranked

    def diversify_by_lexical_content(
        self,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Delegate diversity reranking to the shared MMR implementation."""
        return mmr_rerank(chunks, top_k)
