"""Two-tier query cache: exact-match (SHA-256) + semantic similarity."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict, deque

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


class QueryCache:
    """In-memory two-tier query cache.

    Tier 1: exact match via SHA-256 of normalized query (LRU eviction).
    Tier 2: semantic similarity via NumPy SIMD batch dot product.
    Entries in the semantic tier expire after ``ttl_seconds``.
    """

    def __init__(
        self,
        exact_enabled: bool = True,
        semantic_enabled: bool = True,
        similarity_threshold: float = 0.92,
        exact_max_size: int = 1024,
        semantic_max_size: int = 512,
        ttl_seconds: int = 3600,
    ) -> None:
        self._exact_enabled = exact_enabled
        self._semantic_enabled = semantic_enabled
        self._similarity_threshold = similarity_threshold
        self._exact_cache: OrderedDict[str, str] = OrderedDict()
        self._semantic_cache: deque[tuple[np.ndarray, str, str, float]] = deque()
        self._exact_max_size = exact_max_size
        self._semantic_max_size = semantic_max_size
        self._ttl_seconds = ttl_seconds

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
        if not self._semantic_enabled or not self._semantic_cache or not query_embedding:
            return None

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None
        q_unit = q_vec / q_norm

        now = time.monotonic()
        cutoff = now - self._ttl_seconds
        valid: list[tuple[np.ndarray, str, str, float]] = []
        while self._semantic_cache and self._semantic_cache[0][3] < cutoff:
            self._semantic_cache.popleft()
        valid = list(self._semantic_cache)

        if not valid:
            return None

        matrix = np.vstack([entry[0] for entry in valid])
        similarities = np.dot(matrix, q_unit)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self._similarity_threshold:
            return valid[best_idx][2]
        return None

    def set_semantic(self, query: str, query_embedding: list[float], answer: str) -> None:
        if not self._semantic_enabled or not query_embedding:
            return

        vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return
        unit_vec = vec / norm

        self._semantic_cache.append((unit_vec, query, answer, time.monotonic()))
        if len(self._semantic_cache) > self._semantic_max_size:
            self._semantic_cache.popleft()

    # --- combined ---

    def get(self, query: str, query_embedding: list[float] | None = None) -> str | None:
        exact = self.get_exact(query)
        if exact is not None:
            return exact
        if query_embedding is not None:
            return self.get_semantic(query, query_embedding)
        return None
