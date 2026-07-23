"""MMR (Maximal Marginal Relevance) diversity reranking."""

from __future__ import annotations

import math

from data_engineering_copilot.domain.models import RetrievedChunk


def _tokenize(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _cosine(a: set[str], b: set[str]) -> float:
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

    # Sort by confidence descending
    sorted_chunks = sorted(chunks, key=lambda c: c.confidence, reverse=True)
    selected: list[RetrievedChunk] = []
    remaining = list(sorted_chunks)
    selected_tokens: list[set[str]] = []

    while remaining and len(selected) < top_k:
        best_score = -1.0
        best_idx = 0

        for idx, chunk in enumerate(remaining):
            relevance = chunk.confidence
            chunk_tokens = _tokenize(chunk.chunk.text)

            # Max similarity to already selected chunks
            max_sim = max((_cosine(chunk_tokens, s) for s in selected_tokens), default=0.0)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

            # Tie-break: prefer higher confidence
            if abs(mmr_score - best_score) < 1e-9 and relevance > remaining[best_idx].confidence:
                best_score = mmr_score
                best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_tokens.append(_tokenize(chosen.chunk.text))

    return selected
