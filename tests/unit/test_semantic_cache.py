"""Tests for SemanticCache — cosine similarity, TTL expiry, LRU eviction."""

from __future__ import annotations

import time

from data_engineering_copilot.infrastructure.semantic_cache import SemanticCache


def _vec(*values: float) -> list[float]:
    return list(values)


class TestSemanticCacheGet:
    def test_miss_on_empty(self):
        cache = SemanticCache(threshold=0.9)
        assert cache.get([1.0, 0.0]) is None

    def test_hit_for_identical_vector(self):
        cache = SemanticCache(threshold=0.9)
        cache.set([1.0, 0.0], "answer1")
        assert cache.get([1.0, 0.0]) == "answer1"

    def test_miss_for_orthogonal_vector(self):
        cache = SemanticCache(threshold=0.9)
        cache.set([1.0, 0.0], "answer1")
        assert cache.get([0.0, 1.0]) is None

    def test_hit_for_similar_vector_above_threshold(self):
        cache = SemanticCache(threshold=0.8)
        cache.set([1.0, 0.0], "answer1")
        assert cache.get([0.95, 0.05]) == "answer1"


class TestSemanticCacheSet:
    def test_latest_answer_for_same_embedding(self):
        """Setting the same vector twice stores the latest answer."""
        cache = SemanticCache(threshold=0.9)
        vec = [1.0, 0.0]
        cache.set(vec, "answer1")
        cache.set(vec, "answer2")
        assert cache.get(vec) == "answer2"


class TestSemanticCacheTtl:
    def test_expired_entry_returns_none(self):
        cache = SemanticCache(threshold=0.9, ttl_seconds=0)
        cache.set([1.0, 0.0], "answer1")
        time.sleep(0.01)
        assert cache.get([1.0, 0.0]) is None

    def test_fresh_entry_within_ttl(self):
        cache = SemanticCache(threshold=0.9, ttl_seconds=60)
        cache.set([1.0, 0.0], "answer1")
        assert cache.get([1.0, 0.0]) == "answer1"


class TestSemanticCacheLru:
    def test_evicts_oldest_when_full(self):
        cache = SemanticCache(threshold=0.9, max_entries=2)
        cache.set([1.0, 0.0], "first")
        cache.set([0.0, 1.0], "second")
        cache.set([0.5, 0.5], "third")
        assert cache.get([1.0, 0.0]) is None
        assert cache.get([0.5, 0.5]) == "third"

    def test_fifo_eviction(self):
        cache = SemanticCache(threshold=0.9, max_entries=2)
        cache.set([1.0, 0.0], "first")
        cache.set([0.0, 1.0], "second")
        cache.set([0.5, 0.5], "third")
        assert cache.get([1.0, 0.0]) is None
        assert cache.get([0.5, 0.5]) == "third"
