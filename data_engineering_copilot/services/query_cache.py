"""Two-tier query cache: exact-match (SHA-256) + semantic similarity."""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import OrderedDict

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for canonical key."""
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class QueryCache:
    """In-memory two-tier query cache.

    Tier 1: exact match via SHA-256 of normalized query.
    Tier 2: semantic similarity via cosine distance of query embeddings.
    """

    def __init__(
        self,
        exact_enabled: bool = True,
        semantic_enabled: bool = True,
        similarity_threshold: float = 0.92,
        exact_max_size: int = 1024,
        semantic_max_size: int = 512,
    ) -> None:
        self._exact_enabled = exact_enabled
        self._semantic_enabled = semantic_enabled
        self._similarity_threshold = similarity_threshold
        self._exact_cache: OrderedDict[str, str] = OrderedDict()
        self._semantic_cache: list[tuple[list[float], str, str]] = []  # (embedding, query, answer)
        self._exact_max_size = exact_max_size
        self._semantic_max_size = semantic_max_size

    def _exact_key(self, query: str) -> str:
        normalized = _normalize_query(query)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # --- exact tier ---

    def get_exact(self, query: str) -> str | None:
        if not self._exact_enabled:
            return None
        key = self._exact_key(query)
        return self._exact_cache.get(key)

    def set_exact(self, query: str, answer: str) -> None:
        if not self._exact_enabled:
            return
        key = self._exact_key(query)
        self._exact_cache[key] = answer
        if len(self._exact_cache) > self._exact_max_size:
            self._exact_cache.popitem(last=False)

    # --- semantic tier ---

    def get_semantic(self, query: str, query_embedding: list[float]) -> str | None:
        if not self._semantic_enabled or not self._semantic_cache:
            return None
        best_score = -1.0
        best_answer = None
        for embedding, _cached_q, answer in self._semantic_cache:
            score = _cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best_score = score
                best_answer = answer
        if best_score >= self._similarity_threshold:
            return best_answer
        return None

    def set_semantic(self, query: str, query_embedding: list[float], answer: str) -> None:
        if not self._semantic_enabled:
            return
        self._semantic_cache.append((list(query_embedding), query, answer))
        if len(self._semantic_cache) > self._semantic_max_size:
            self._semantic_cache.pop(0)

    # --- combined ---

    def get(self, query: str, query_embedding: list[float] | None = None) -> str | None:
        """Two-tier lookup: exact first, then semantic."""
        exact = self.get_exact(query)
        if exact is not None:
            return exact
        if query_embedding is not None:
            return self.get_semantic(query, query_embedding)
        return None
