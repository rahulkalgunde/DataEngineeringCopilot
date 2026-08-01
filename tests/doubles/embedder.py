"""Deterministic embedding double for pipeline-logic tests (no Ollama).

Produces reproducible bag-of-ngram vectors so cosine retrieval ranks chunks
that share vocabulary the same way a real embedder would, but without any
network or model dependency.
"""

from __future__ import annotations

import hashlib
import math
import re

from data_engineering_copilot.domain.protocols import EmbedderProtocol


def _text_vector(text: str, dimension: int) -> list[float]:
    """Seeded bag-of-ngrams vector (word + char-bigram hashing), L2-normalized."""
    vec = [0.0] * dimension
    lowered = text.lower()
    tokens = re.findall(r"\w+", lowered)
    grams = list(tokens)
    grams.extend(t[:-1] + "#" + t[-1] for t in tokens if len(t) > 1)
    if len(lowered) > 1:
        grams.extend(lowered[i : i + 2] for i in range(len(lowered) - 1))
    for gram in grams:
        idx = int.from_bytes(hashlib.sha256(gram.encode()).digest()[:4], "big") % dimension
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class StubEmbedder(EmbedderProtocol):
    """Deterministic, offline embedder for hermetic RAG tests."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        self.embed_texts_calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(texts)
        return [_text_vector(t, self.dimension) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _text_vector(text, self.dimension)

    async def close(self) -> None:
        pass
