"""In-memory query cache with TTL for RAG results."""

from __future__ import annotations

import time
from dataclasses import dataclass

from data_engineering_copilot.domain.models import Answer


@dataclass
class _CacheEntry:
    answer: Answer
    expires_at: float


class QueryCache:
    """Simple in-memory TTL cache for query answers.

    Parameters
    ----------
    ttl_seconds : int
        Time-to-live for cached entries.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, _CacheEntry] = {}

    def _normalize_key(self, query: str) -> str:
        return " ".join(query.lower().split())

    def get(self, query: str) -> Answer | None:
        key = self._normalize_key(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        return entry.answer

    def set(self, query: str, answer: Answer) -> None:
        key = self._normalize_key(query)
        self._store[key] = _CacheEntry(
            answer=answer,
            expires_at=time.monotonic() + self._ttl,
        )

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)
