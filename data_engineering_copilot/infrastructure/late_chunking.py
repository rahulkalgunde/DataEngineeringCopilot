"""Late chunking: contextual chunk embeddings via whole-document token pooling.

Jina-style late chunking (arXiv:2409.04701): run the long-context embedding
transformer over the ENTIRE document once, then mean-pool token vectors per
precomputed chunk span (offsets from ``start_offset``/``end_offset``). Chunk
embeddings end up conditioned on surrounding context instead of i.i.d.

Hard constraint: the document must fit the model's context window; over-long
documents raise rather than silently truncating tail chunks.
"""

from __future__ import annotations

import asyncio
import logging
import math

from data_engineering_copilot.domain.exceptions import EmbeddingError
from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import _load_model

logger = logging.getLogger(__name__)


class LateChunkEmbedder:
    """Whole-document token encoding with per-span mean pooling.

    Wraps a local sentence-transformers model (see
    ``LocalSentenceTransformerEmbeddings`` for the loading/caching contract).
    Not part of the ``EmbedderProtocol`` fallback chain: callers use it only
    when ``late_chunking_enabled`` is set AND the active embedder is local.
    """

    def __init__(self, model_name: str, max_tokens: int = 8192) -> None:
        self._model_name = model_name
        self.max_tokens = max_tokens
        self._encoder = _load_model(model_name)

    async def embed_document_spans(self, text: str, spans: list[tuple[int, int]]) -> list[list[float]]:
        """Pool token embeddings of *text* for each ``(start, end)`` char span."""
        if not spans:
            return []
        return await asyncio.to_thread(self._embed_sync, text, spans)

    def _embed_sync(self, text: str, spans: list[tuple[int, int]]) -> list[list[float]]:
        tokenizer = getattr(self._encoder, "tokenizer", None)
        if tokenizer is None:
            raise EmbeddingError("late chunking requires a sentence-transformers model with a tokenizer")
        encoded = tokenizer(text, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping") or []

        def _token_count() -> int:
            ids = encoded.get("input_ids")
            if ids is not None:
                try:
                    return len(ids)
                except TypeError:
                    pass
            return len(offsets)

        if _token_count() > self.max_tokens:
            raise EmbeddingError(
                f"document exceeds {self.max_tokens}-token model context "
                f"({_token_count()} tokens); fall back to naive chunk embedding"
            )

        # One forward pass over the full document -> token-level vectors.
        token_vecs = self._encoder.encode([text], output_value="token_embeddings")[0]

        pooled: list[list[float]] = []
        dim = len(token_vecs[0]) if token_vecs else 0
        for start, end in spans:
            acc = [0.0] * dim
            count = 0
            for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_idx >= len(token_vecs):
                    break
                # (0, 0) marks special tokens (CLS/SEP/padding) — exclude.
                if tok_start == 0 and tok_end == 0:
                    continue
                if tok_end <= start or tok_start >= end:
                    continue
                vec = token_vecs[tok_idx]
                for j in range(dim):
                    acc[j] += vec[j]
                count += 1
            if count == 0:
                # Span matched no tokens (e.g. whitespace-only): zero vector.
                pooled.append([0.0] * dim)
                continue
            mean = [a / count for a in acc]
            norm = math.sqrt(sum(x * x for x in mean))
            pooled.append([x / norm for x in mean] if norm > 0 else mean)
        return pooled
