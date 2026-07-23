"""Context compressor: redundancy elimination and relevance re-ranking."""
from __future__ import annotations

import logging
import math
import re

from data_engineering_copilot.domain.models import DocumentChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ContextCompressor:
    """Removes near-duplicate chunks and re-ranks by query relevance.

    No embeddings required — uses token-level similarity.
    """

    def __init__(
        self,
        enabled: bool = True,
        similarity_threshold: float = 0.85,
        max_chunks: int = 10,
    ) -> None:
        self._enabled = enabled
        self._similarity_threshold = similarity_threshold
        self._max_chunks = max_chunks

    def compress(
        self,
        chunks: list[DocumentChunk],
        query: str,
    ) -> list[DocumentChunk]:
        """Deduplicate and re-rank chunks for the given query."""
        if not self._enabled or not chunks:
            return chunks

        # Step 1: deduplicate near-identical chunks (keep first occurrence)
        deduped = self._deduplicate(chunks)

        # Step 2: score relevance to query
        query_tokens = _tokenize(query)
        scored: list[tuple[float, int, DocumentChunk]] = []
        for idx, chunk in enumerate(deduped):
            chunk_tokens = _tokenize(chunk.text)
            # Combine cosine similarity with simple overlap ratio
            cos_sim = _cosine(query_tokens, chunk_tokens) if query_tokens else 0.0
            # Boost chunks that share more tokens with the query
            overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
            score = cos_sim * 0.6 + overlap * 0.4
            scored.append((score, idx, chunk))

        # Sort by score descending, then original order for ties
        scored.sort(key=lambda x: (-x[0], x[1]))

        return [chunk for _, _, chunk in scored[: self._max_chunks]]

    def _deduplicate(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Remove near-duplicate chunks using Jaccard similarity."""
        result: list[DocumentChunk] = []
        seen_tokens: list[set[str]] = []

        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            is_dup = False
            for seen in seen_tokens:
                if _jaccard(tokens, seen) >= self._similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                result.append(chunk)
                seen_tokens.append(tokens)

        return result
