"""Tests for two-tier query cache: exact match + semantic similarity."""

from __future__ import annotations

from data_engineering_copilot.services.query_cache import QueryCache


class TestExactCache:
    def test_miss_on_empty(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        result = cache.get_exact("What is Spark?")
        assert result is None

    def test_set_then_get_exact(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", "Spark is a data processing engine.")
        result = cache.get_exact("What is Spark?")
        assert result == "Spark is a data processing engine."

    def test_exact_case_insensitive(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", "answer")
        assert cache.get_exact("what is spark?") == "answer"

    def test_exact_disabled_returns_none(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", "answer")
        assert cache.get_exact("What is Spark?") is None


class TestSemanticCache:
    def test_miss_when_empty(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        result = cache.get_semantic("What is Spark?", [0.1] * 768)
        assert result is None

    def test_hit_for_identical_vector(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        vec = [0.5] * 768
        cache.set_semantic("What is Spark?", vec, "answer1")
        result = cache.get_semantic("What is Spark?", vec)
        assert result == "answer1"

    def test_miss_for_orthogonal_vector(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        vec_a = [1.0] + [0.0] * 767
        vec_b = [0.0] * 768
        cache.set_semantic("What is Spark?", vec_a, "answer1")
        result = cache.get_semantic("Something else", vec_b)
        assert result is None

    def test_semantic_disabled_returns_none(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_semantic("q", [1.0] * 768, "a")
        assert cache.get_semantic("q", [1.0] * 768) is None


class TestCombined:
    def test_exact_hit_skips_semantic(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=True, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", "exact_answer")
        result = cache.get("What is Spark?", query_embedding=[0.1] * 768)
        assert result == "exact_answer"

    def test_fallback_to_semantic(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=True, similarity_threshold=0.9)
        vec = [0.5] * 768
        cache.set_semantic("What is Spark?", vec, "semantic_answer")
        result = cache.get("What is Spark?", query_embedding=vec)
        assert result == "semantic_answer"

    def test_miss_returns_none(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=True, similarity_threshold=0.9)
        assert cache.get("What is Spark?", query_embedding=[0.1] * 768) is None
