"""In-memory semantic cache with TTL expiry and LRU eviction."""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SemanticCache:
    """Embedding-based semantic cache with TTL and LRU eviction.

    Stores ``(query_embedding → answer_text)``, retrieves by cosine
    similarity ≥ ``threshold``.  Entries expire after ``ttl_seconds``.
    Evicts least-recently-used when exceeding ``max_entries``.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_entries: int = 1000,
    ) -> None:
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._cache: OrderedDict[str, tuple[list[float], str, float]] = OrderedDict()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def get(self, query_embedding: list[float]) -> str | None:
        now = time.monotonic()
        best_score = -1.0
        best_answer: str | None = None
        expired_keys: list[str] = []
        for key, (emb, answer, ts) in self._cache.items():
            if now - ts > self._ttl:
                expired_keys.append(key)
                continue
            score = self._cosine(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_answer = answer
        for k in expired_keys:
            self._cache.pop(k, None)
        if best_score >= self._threshold:
            return best_answer
        return None

    def set(self, query_embedding: list[float], answer: str) -> None:
        if len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
        key = str(id(query_embedding))
        self._cache[key] = (query_embedding, answer, time.monotonic())
