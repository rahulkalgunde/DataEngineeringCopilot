"""True ColBERT reranker via PyLate (neural late interaction).

Distinct from the lexical char-trigram proxy (``colbert_reranker.py`` /
``LexicalNgramReranker``): this class loads a real ColBERT checkpoint
(``colbert-ir/colbertv2.0`` by default) through PyLate and scores query-token
× document-token MaxSim — actual late-interaction semantics.

Rerank-only usage: documents are encoded on the fly per candidate pool; no
multivector index is built (see docs/research/
deferred_experiments_research_2026-08-21.md §3 for the storage path).

Failure posture: fail-open. Missing dependency or runtime errors degrade to
pass-through order with a warning, matching the other rerankers.

Gate (dark flag): ``reranker_type="pylate_colbert"`` only after
``dec eval-rerank`` shows nDCG@10 >= cross_encoder + 0.02 AND p95 pool
latency <= 2x cross_encoder.
"""

from __future__ import annotations

import asyncio
import logging

from data_engineering_copilot.domain.models import RetrievedChunk

logger = logging.getLogger(__name__)

_DOC_TRUNCATION_CHARS = 2000


class PyLateColBERTReranker:
    """Neural late-interaction reranker backed by PyLate (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        doc_truncation_chars: int = _DOC_TRUNCATION_CHARS,
    ) -> None:
        self.model_name = model_name
        self._doc_truncation_chars = doc_truncation_chars
        self._model = None
        self._rank = None

    def diversify_by_lexical_content(
        self,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Delegate diversity reranking to the shared MMR implementation."""
        from data_engineering_copilot.services.reranker import mmr_rerank

        return mmr_rerank(chunks, top_k)

    async def is_available_async(self) -> bool:
        try:
            import pylate  # type: ignore[import-not-found]  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def is_available(self) -> bool:
        try:
            import pylate  # type: ignore[import-not-found]  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    async def initialize(self) -> None:
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from pylate import models as pylate_models  # type: ignore[import-not-found]
        from pylate import rank as pylate_rank  # type: ignore[import-not-found]

        logger.info("Loading PyLate ColBERT model %s", self.model_name)
        self._model = pylate_models.ColBERT(model_name_or_path=self.model_name, device="cpu")
        self._rank = pylate_rank

    async def close(self) -> None:
        self._model = None
        self._rank = None

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank chunks by neural MaxSim; min-max normalized within pool."""
        if len(chunks) <= 1:
            return chunks[:top_k]
        try:
            await self.initialize()
            assert self._model is not None and self._rank is not None
            from data_engineering_copilot.services.reranker import (
                _min_max_normalize,
                _truncate_doc_for_rerank,
            )

            texts = [_truncate_doc_for_rerank(c.chunk.text, self._doc_truncation_chars) for c in chunks]

            q_emb = await asyncio.to_thread(self._model.encode, [query], True)
            d_emb = await asyncio.to_thread(self._model.encode, texts, False)
            doc_ids = [c.chunk.chunk_id for c in chunks]
            ranked = await asyncio.to_thread(
                self._rank.rerank,
                [doc_ids],
                q_emb,
                d_emb,
            )
            score_by_id = {entry["id"]: float(entry["score"]) for entry in ranked[0]}
            raw = [score_by_id.get(c.chunk.chunk_id, 0.0) for c in chunks]
            normed = _min_max_normalize(raw)
            rescored = [
                RetrievedChunk(
                    chunk=original.chunk,
                    distance=1 - confidence,
                    confidence=confidence,
                )
                for original, confidence in zip(chunks, normed, strict=True)
            ]
            rescored.sort(key=lambda rc: rc.confidence, reverse=True)
            return rescored[:top_k]
        except Exception:
            logger.warning("pylate colbert rerank failed; returning original order", exc_info=True)
            return chunks[:top_k]
