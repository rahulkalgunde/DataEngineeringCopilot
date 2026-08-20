"""Context compressor: redundancy elimination and relevance re-ranking."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Protocol

from data_engineering_copilot.domain.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Minimal embedding interface for semantic deduplication."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _cosine_tokens(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _cosine_vectors(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ContextCompressor:
    """Removes near-duplicate chunks and re-ranks by query relevance.

    Deduplication defaults to token-level Jaccard similarity (no embeddings
    required). When *embedding_model* is supplied (and exposes a synchronous
    ``embed_texts(list[str]) -> list[list[float]]``), deduplication uses
    cosine similarity over chunk embeddings instead, which captures semantic
    near-duplicates that token overlap misses. Async embedders are not
    supported in the synchronous ``compress`` path and fall back to token
    Jaccard automatically.
    """

    def __init__(
        self,
        enabled: bool = True,
        similarity_threshold: float = 0.85,
        max_chunks: int = 10,
        compression_ratio: float = 0.8,
        embedding_model: Embedder | None = None,
    ) -> None:
        self._enabled = enabled
        self._similarity_threshold = similarity_threshold
        self._max_chunks = max_chunks
        self._compression_ratio = compression_ratio
        self._embedding_model = embedding_model

    def compress(
        self,
        chunks: list[RetrievedChunk],
        query: str,
    ) -> list[RetrievedChunk]:
        """Deduplicate and re-rank chunks for the given query."""
        if not self._enabled or not chunks:
            return chunks

        # Step 1: deduplicate near-identical chunks (keep first occurrence)
        deduped = self._deduplicate(chunks)

        # Step 2: score relevance to query
        query_tokens = _tokenize(query)
        if not query_tokens:
            return deduped[: self._max_chunks]

        scored: list[tuple[float, int, RetrievedChunk]] = []
        for idx, chunk in enumerate(deduped):
            chunk_tokens = _tokenize(chunk.chunk.text)
            # Combine cosine similarity with simple overlap ratio
            cos_sim = _cosine_tokens(query_tokens, chunk_tokens)
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            score = cos_sim * 0.6 + overlap * 0.4
            scored.append((score, idx, chunk))

        # Sort by score descending, then original order for ties
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Apply compression ratio: keep at most ratio * input size, capped by max_chunks
        # Only apply ratio limit when input is larger than max_chunks
        if len(chunks) > self._max_chunks:
            ratio_limit = max(1, int(len(chunks) * self._compression_ratio))
            effective_limit = min(ratio_limit, self._max_chunks)
        else:
            effective_limit = self._max_chunks

        return [chunk for _, _, chunk in scored[:effective_limit]]

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove near-duplicate chunks.

        Uses embedding cosine similarity when an ``embedding_model`` is
        configured, otherwise falls back to token-level Jaccard.
        """
        model = self._embedding_model
        if model is None:
            return self._deduplicate_token(chunks)
        try:
            return self._deduplicate_embedded(chunks, model)
        except Exception:  # noqa: BLE001 - never let dedup crash the pipeline
            logger.warning("embedding dedup failed; falling back to token Jaccard")
            return self._deduplicate_token(chunks)

    def _deduplicate_embedded(
        self,
        chunks: list[RetrievedChunk],
        model: Embedder,
    ) -> list[RetrievedChunk]:
        if asyncio.iscoroutinefunction(model.embed_texts):
            logger.warning("async embedder unsupported in sync compress; using token Jaccard")
            return self._deduplicate_token(chunks)
        texts = [chunk.chunk.text for chunk in chunks]
        embeddings = model.embed_texts(texts)
        result: list[RetrievedChunk] = []
        seen: list[list[float]] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            is_dup = False
            for prev in seen:
                if _cosine_vectors(emb, prev) >= self._similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                result.append(chunk)
                seen.append(emb)
        return result

    def _deduplicate_token(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove near-duplicate chunks using Jaccard similarity."""
        result: list[RetrievedChunk] = []
        seen_tokens: list[set[str]] = []

        for chunk in chunks:
            tokens = _tokenize(chunk.chunk.text)
            is_dup = False
            for seen in seen_tokens:
                if _jaccard(tokens, seen) >= self._similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                result.append(chunk)
                seen_tokens.append(tokens)

        return result
