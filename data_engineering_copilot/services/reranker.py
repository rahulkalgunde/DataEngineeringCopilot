"""Reranking module for improving retrieval result quality.

This module implements cross-encoder reranking and MMR diversity reranking
to improve answer relevance by re-scoring chunks based on semantic similarity
to the query.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from data_engineering_copilot.domain.models import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Module-level singleton cache: model_name → CrossEncoderReranker
_reranker_cache: dict[str, CrossEncoderReranker] = {}
_cache_lock = threading.Lock()


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """Get or create a singleton CrossEncoderReranker for the given model."""
    if model_name not in _reranker_cache:
        with _cache_lock:
            if model_name not in _reranker_cache:
                _reranker_cache[model_name] = CrossEncoderReranker(model_name=model_name)
    return _reranker_cache[model_name]


def clear_reranker_cache() -> None:
    """Clear the reranker singleton cache (for testing)."""
    with _cache_lock:
        _reranker_cache.clear()


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model.

    Cross-encoders jointly encode the query and chunk, producing a relevance
    score that is more accurate than embedding similarity for ranking.

    This implementation uses the lightweight 'all-MiniLM-L6-v2' model
    via sentence-transformers for local inference.
    """

    def __init__(self, model_name: str = "cross-encoder/qnli-distilroberta-base"):
        """Initialize the cross-encoder reranker.

        Model loading is deferred — call ``await initialize()`` before first use
        to avoid blocking the event loop during the ~450MB download.
        """
        self.model_name = model_name
        self.model: CrossEncoder | None = None
        self._executor: ThreadPoolExecutor | None = None

    async def initialize(self) -> None:
        """Load the cross-encoder model off the event loop.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self.model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            loop = asyncio.get_running_loop()
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")
            self.model = await loop.run_in_executor(self._executor, lambda: CrossEncoder(self.model_name))
            logger.info("Initialized CrossEncoder reranker: %s", self.model_name)
        except ImportError:
            logger.warning(
                "sentence_transformers not available; reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as exc:
            logger.warning("Failed to initialize CrossEncoder reranker: %s", exc)

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Rerank chunks based on query relevance using cross-encoder.

        PyTorch inference is CPU-bound, so ``model.predict()`` runs in a
        dedicated thread pool to avoid blocking the asyncio event loop.

        Args:
            query: The user question
            chunks: Retrieved chunks to rerank
            top_k: Number of top results to return

        Returns:
            List of top_k reranked chunks sorted by cross-encoder score
        """
        if not chunks:
            return []

        if not self.model:
            logger.info("Reranker model not loaded; attempting synchronous init")
            await asyncio.to_thread(self._init_sync)
        if not self.model:
            logger.warning("Reranker model not available; returning chunks unchanged")
            return chunks[:top_k]

        if len(chunks) <= top_k:
            # Already have fewer chunks than requested; no need to rerank
            return chunks

        try:
            # Prepare texts for cross-encoder: (query, chunk_text) pairs
            chunk_texts = [chunk.chunk.text for chunk in chunks]
            pairs = [[query, text] for text in chunk_texts]

            # Score each (query, chunk) pair — raw logits from cross-encoder.
            # Run CPU-bound PyTorch inference OFF the event loop.
            executor = self._executor
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")
                self._executor = executor
            loop = asyncio.get_running_loop()
            raw_scores = await loop.run_in_executor(
                executor,
                lambda: self.model.predict(pairs),  # type: ignore[union-attr]
            )

            # Normalize logits to [0, 1] via sigmoid
            scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

            # Sort chunks by normalized score (highest first)
            scored_chunks = list(zip(chunks, scores, strict=False))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Build new RetrievedChunk instances with reranker score as confidence
            reranked = [
                RetrievedChunk(
                    chunk=chunk.chunk,
                    distance=1.0 - score,
                    confidence=score,
                )
                for chunk, score in scored_chunks[:top_k]
            ]

            logger.info(
                "Reranked %d chunks → %d chunks; top score=%.4f",
                len(chunks),
                len(reranked),
                scored_chunks[0][1] if scored_chunks else 0.0,
            )

            # Log score comparison (before vs after)
            original_top_score = chunks[0].confidence if chunks else 0.0
            new_top_score = scored_chunks[0][1] if scored_chunks else 0.0
            if abs(new_top_score - original_top_score) > 0.05:
                logger.info("Score improvement: embedding=%.4f → reranker=%.4f", original_top_score, new_top_score)

            return reranked

        except Exception as exc:
            logger.exception("Reranking failed; returning original chunks: %s", exc)
            return chunks[:top_k]

    def _init_sync(self) -> None:
        """Fallback synchronous model init for CLI/test contexts."""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            logger.info("Initialized CrossEncoder reranker (sync): %s", self.model_name)
        except Exception as exc:
            logger.warning("Failed to initialize CrossEncoder reranker (sync): %s", exc)

    def is_available(self) -> bool:
        """Check if reranker model is available.

        Returns:
            True if model is loaded and ready, False otherwise
        """
        return self.model is not None

    async def close(self) -> None:
        """Shut down the inference thread pool, releasing its worker threads."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def max_marginal_relevance(
        self,
        query_emb: list[float],
        chunks: list[RetrievedChunk],
        lambda_param: float = 0.5,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """MMR diversity reranking — relevance (cross-encoder or embedding score) + diversity.

        Args:
            query_emb: Query embedding vector.
            chunks: Candidate chunks (already scored by cross-encoder or embedding).
            lambda_param: Tradeoff between relevance (1.0) and diversity (0.0).
            top_k: Max chunks to return.
        """
        if not chunks or top_k <= 0:
            return []
        sorted_chunks = sorted(chunks, key=lambda c: c.confidence, reverse=True)
        selected: list[RetrievedChunk] = []
        remaining = list(sorted_chunks)
        selected_tokens: list[set[str]] = []
        while remaining and len(selected) < top_k:
            best_score = -1.0
            best_idx = 0
            for idx, chunk in enumerate(remaining):
                relevance = chunk.confidence
                chunk_tokens = _mmr_tokenize(chunk.chunk.text)
                max_sim = max((_mmr_cosine(chunk_tokens, s) for s in selected_tokens), default=0.0)
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            selected_tokens.append(_mmr_tokenize(chosen.chunk.text))
        return selected


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) diversity reranking
# ---------------------------------------------------------------------------


def _mmr_tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _mmr_cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def mmr_rerank(
    chunks: list[RetrievedChunk],
    top_k: int,
    lambda_param: float = 0.5,
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance reranking for diversity.

    Balances relevance (chunk.confidence) with diversity (penalizes chunks
    similar to already-selected ones).

    Args:
        chunks: Candidate chunks sorted by relevance.
        top_k: Maximum number of chunks to return.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
    """
    if not chunks or top_k <= 0:
        return []

    sorted_chunks = sorted(chunks, key=lambda c: c.confidence, reverse=True)
    selected: list[RetrievedChunk] = []
    remaining = list(sorted_chunks)
    selected_tokens: list[set[str]] = []

    while remaining and len(selected) < top_k:
        best_score = -1.0
        best_idx = 0

        for idx, chunk in enumerate(remaining):
            relevance = chunk.confidence
            chunk_tokens = _mmr_tokenize(chunk.chunk.text)

            max_sim = max((_mmr_cosine(chunk_tokens, s) for s in selected_tokens), default=0.0)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

            if abs(mmr_score - best_score) < 1e-9 and relevance > remaining[best_idx].confidence:
                best_score = mmr_score
                best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_tokens.append(_mmr_tokenize(chosen.chunk.text))

    return selected
