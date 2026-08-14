"""LLM-based reranker facade with the local cross-encoder as last resort.

Wraps a ``ProviderFallbackChain[RerankRequest, RerankResult]`` (cloud rerank
providers in priority order) and keeps the local ``CrossEncoderReranker`` as the
final degraded fallback. Implements the same ``RerankerProtocol`` surface as the
local reranker so ``AsyncRagService`` uses it unchanged.
"""

from __future__ import annotations

import logging

from data_engineering_copilot.domain.models import RerankRequest, RetrievedChunk
from data_engineering_copilot.infrastructure.llm_client import LLMClientError
from data_engineering_copilot.services.reranker import mmr_rerank

logger = logging.getLogger(__name__)


class LLMReranker:
    """Re-rank chunks through the cloud fallback chain, local last.

    ``rerank()`` rebuilds each chunk's ``confidence`` from the provider's
    normalized ``[0, 1]`` score (and ``distance`` as ``1 - score``), mirroring
    ``CrossEncoderReranker`` so the downstream confidence gate is unchanged.

    Availability semantics:
        - When a cloud chain is configured, reranking is available even before
          the local model is loaded (the local model loads lazily on the
          degraded path only).
        - Without a cloud chain, availability follows the local model exactly
          (the pre-existing local-only behavior).
    """

    def __init__(self, chain=None, local=None) -> None:
        """Build the LLM reranker facade.

        Args:
            chain: ``ProviderFallbackChain[RerankRequest, RerankResult]`` of
                cloud providers (may be ``None`` when no cloud reranker is
                configured).
            local: Local ``CrossEncoderReranker`` used as the last-resort
                fallback (may be ``None`` to disable local reranking).
        """
        self._chain = chain
        self._local = local

    async def initialize(self) -> None:
        """Ensure reranking is ready.

        With a cloud chain present, this is a no-op — the local model loads
        lazily only when the degraded path actually needs it. Without a chain,
        the local model is loaded off the event loop (pre-existing behavior).
        """
        if self._chain is None and self._local is not None:
            await self._local.initialize()

    def is_available(self) -> bool:
        """Whether reranking can run.

        True when a cloud chain is configured, or when the local model is
        loaded (local-only mode). False only when neither path can rerank.
        """
        if self._chain is not None:
            return True
        return self._local is not None and self._local.is_available()

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Rerank ``chunks`` against ``query``, returning at most ``top_k``.

        Cloud providers run first; on total chain failure the local cross-encoder
        is the degraded fallback. When no path can rerank, chunks are returned
        unchanged (trimmed to ``top_k``).
        """
        if not chunks:
            return []

        if self._chain is None:
            if self._local is None:
                return chunks[:top_k]
            return await self._local.rerank(query, chunks, top_k)

        try:
            request = RerankRequest(
                query=query,
                documents=[chunk.chunk.text for chunk in chunks],
                top_n=top_k,
            )
            result = await self._chain.execute(request)
        except LLMClientError:
            # All providers (including the local degraded fallback) were
            # skipped or failed — degrade to "no reranking".
            logger.warning("All rerank providers failed; returning chunks unchanged")
            return chunks[:top_k]

        return self._apply(result.rankings, chunks, top_k)

    def _apply(self, rankings, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Map provider ``(index, score)`` rankings back onto the chunks.

        Unranked chunks (the provider did not cover them) sort after all ranked
        ones, preserving their original relative order. Scores become
        ``confidence`` and ``distance = 1 - confidence``.
        """
        if not rankings:
            return chunks[:top_k]

        score_by_index = {index: float(score) for index, score in rankings}
        ordered = sorted(range(len(chunks)), key=lambda i: score_by_index.get(i, -1.0), reverse=True)
        result = [
            RetrievedChunk(
                chunk=chunks[i].chunk,
                distance=max(0.0, 1.0 - score_by_index.get(i, 0.0)),
                confidence=score_by_index.get(i, 0.0),
            )
            for i in ordered[:top_k]
        ]
        return result

    def diversify_by_lexical_content(self, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        """Lexical diversity reranking (delegates to the local reranker)."""
        if self._local is not None:
            return self._local.diversify_by_lexical_content(chunks, top_k=top_k)
        return mmr_rerank(chunks, top_k=top_k)

    async def close(self) -> None:
        """Close the cloud chain clients and the local reranker."""
        if self._chain is not None and hasattr(self._chain, "close"):
            await self._chain.close()
        elif self._local is not None:
            await self._local.close()
