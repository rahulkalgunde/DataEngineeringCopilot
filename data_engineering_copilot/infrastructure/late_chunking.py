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
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_engineering_copilot.domain.models import DocumentChunk

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
                    acc[j] += float(vec[j])
                count += 1
            if count == 0:
                # Span matched no tokens (e.g. whitespace-only): zero vector.
                pooled.append([0.0] * dim)
                continue
            mean = [a / count for a in acc]
            norm = math.sqrt(sum(x * x for x in mean))
            pooled.append([x / norm for x in mean] if norm > 0 else mean)
        return pooled


async def embed_document_grouped(
    chunks: list[DocumentChunk],
    *,
    naive_embed: Callable[[list[str]], Awaitable[list[list[float]]]],
    late_embedder: Callable[[], LateChunkEmbedder] | None,
    max_group_tokens: int = 8192,
) -> list[list[float]]:
    """Embed *chunks* with parent-grouped late chunking and graceful fallback.

    Segments sharing a non-empty ``parent_content_hash`` are joined into a
    pseudo-document (sorted by ``segment_index``) and pooled per segment span;
    unparented chunks go through the naive path directly. Any late-chunking
    failure (missing local model, context overflow, runtime error) falls back
    to ``naive_embed`` for the ENTIRE batch — builds never fail because of
    this feature.
    """

    if late_embedder is None or not chunks:
        return await naive_embed([c.text for c in chunks])

    groups: dict[str, list[int]] = {}
    for idx, chunk in enumerate(chunks):
        key = chunk.parent_content_hash if chunk.parent_content_hash else f"__solo__:{chunk.chunk_id}"
        if not chunk.parent_content_hash:
            continue  # solos always take the naive path below
        groups.setdefault(key, []).append(idx)

    try:
        late = late_embedder()
        vectors: list[list[float]] = [[] for _ in chunks]
        for _key, idxs in groups.items():
            ordered = sorted(idxs, key=lambda i: chunks[i].segment_index)
            pseudo_text = "\n".join(chunks[i].text for i in ordered)
            spans: list[tuple[int, int]] = []
            cursor = 0
            for i in ordered:
                text_len = len(chunks[i].text)
                spans.append((cursor, cursor + text_len))
                cursor += text_len + len("\n")
            pooled = await late.embed_document_spans(pseudo_text, spans)
            for pos, i in enumerate(ordered):
                vectors[i] = pooled[pos]
        # Naive-embed every chunk that stayed outside late grouping.
        solo_idxs = [i for i in range(len(chunks)) if not vectors[i]]
        if solo_idxs:
            solo_vecs = await naive_embed([chunks[i].text for i in solo_idxs])
            for pos, i in enumerate(solo_idxs):
                vectors[i] = solo_vecs[pos]
        return vectors
    except Exception as exc:
        logger.warning("late_chunking_fallback_to_naive reason=%s chunks=%d", type(exc).__name__, len(chunks))
        return await naive_embed([c.text for c in chunks])
