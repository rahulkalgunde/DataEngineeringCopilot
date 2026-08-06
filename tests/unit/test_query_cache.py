"""Tests for two-tier query cache: exact match + semantic similarity.

Covers the scoped key contract: entries are isolated per ``CacheScope`` so
two tenants / source filters / embedding models never share cached answers.
Values are ``CachedAnswer`` envelopes, not bare strings.
"""

from __future__ import annotations

from data_engineering_copilot.domain.models import CachedAnswer, CacheScope, DocumentChunk
from data_engineering_copilot.services.query_cache import QueryCache, scope_fingerprint

_SPARK = CacheScope(tenant_id="tenant-a", role="reader", source_filter=("Spark",))
_DELTA = CacheScope(tenant_id="tenant-b", role="reader", source_filter=("Delta",))


def _source() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="c1",
        source_name="Apache Spark",
        title="Spark",
        url="https://example.com/spark",
        text="Apache Spark is a unified analytics engine.",
        doc_type="guide",
    )


def _answer(text: str = "Spark is a data processing engine.") -> CachedAnswer:
    return CachedAnswer(text=text, sources=(_source(),), confidence=0.91, groundedness_score=0.88)


class TestExactCache:
    def test_miss_on_empty(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        result = cache.get_exact("What is Spark?")
        assert result is None

    def test_set_then_get_exact(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", _answer())
        result = cache.get_exact("What is Spark?")
        assert result is not None
        assert result.text == "Spark is a data processing engine."

    def test_exact_case_insensitive(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", _answer())
        result = cache.get_exact("what is spark?")
        assert result is not None
        assert result.text == "Spark is a data processing engine."

    def test_exact_disabled_returns_none(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", _answer())
        assert cache.get_exact("What is Spark?") is None

    def test_envelope_preserves_confidence(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("q", _answer())
        result = cache.get_exact("q")
        assert result is not None
        assert result.confidence == 0.91
        assert result.groundedness_score == 0.88


class TestSemanticCache:
    def test_miss_when_empty(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        result = cache.get_semantic("What is Spark?", [0.1] * 768)
        assert result is None

    def test_hit_for_identical_vector(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        vec = [0.5] * 768
        cache.set_semantic("What is Spark?", vec, _answer("answer1"))
        result = cache.get_semantic("What is Spark?", vec)
        assert result is not None
        assert result.text == "answer1"

    def test_miss_for_orthogonal_vector(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        vec_a = [1.0] + [0.0] * 767
        vec_b = [0.0] * 768
        cache.set_semantic("What is Spark?", vec_a, _answer("answer1"))
        result = cache.get_semantic("Something else", vec_b)
        assert result is None

    def test_semantic_disabled_returns_none(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_semantic("q", [1.0] * 768, _answer("a"))
        assert cache.get_semantic("q", [1.0] * 768) is None


class TestSemanticDimensionGuard:
    """A vector cached under a different embedding dimension (e.g. from a
    previous embedding model) must be skipped, never crash the lookup."""

    def test_wrong_dim_entry_skipped_not_crashed(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        cache.set_semantic("stale", [0.5] * 768, _answer("stale 768"))
        result = cache.get_semantic("What is Spark?", [0.25] * 2048)
        assert result is None

    def test_wrong_dim_entry_does_not_hide_same_dim_hits(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        cache.set_semantic("stale", [0.5] * 768, _answer("stale 768"))
        vec = [0.5] + [0.0] * 2047
        cache.set_semantic("fresh", vec, _answer("fresh 2048"))
        result = cache.get_semantic("fresh", vec)
        assert result is not None
        assert result.text == "fresh 2048"


class TestCombined:
    def test_exact_hit_skips_semantic(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=True, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", _answer("exact_answer"))
        result = cache.get("What is Spark?", query_embedding=[0.1] * 768)
        assert result is not None
        assert result.text == "exact_answer"

    def test_fallback_to_semantic(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=True, similarity_threshold=0.9)
        vec = [0.5] * 768
        cache.set_semantic("What is Spark?", vec, _answer("semantic_answer"))
        result = cache.get("What is Spark?", query_embedding=vec)
        assert result is not None
        assert result.text == "semantic_answer"

    def test_miss_returns_none(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=True, similarity_threshold=0.9)
        assert cache.get("What is Spark?", query_embedding=[0.1] * 768) is None


class TestScopeIsolation:
    def test_scope_fingerprint_differs_by_tenant(self):
        a = CacheScope(tenant_id="tenant-a", role="reader", source_filter=("Spark",))
        b = CacheScope(tenant_id="tenant-b", role="reader", source_filter=("Spark",))
        assert scope_fingerprint(a) != scope_fingerprint(b)

    def test_scope_fingerprint_differs_by_source_filter(self):
        a = CacheScope(tenant_id="t", role="reader", source_filter=("Spark",))
        b = CacheScope(tenant_id="t", role="reader", source_filter=("Delta",))
        assert scope_fingerprint(a) != scope_fingerprint(b)

    def test_scope_fingerprint_differs_by_embedding_model(self):
        a = CacheScope(embedding_model="nomic-embed-text", collection_name="docs")
        b = CacheScope(embedding_model="bge-m3", collection_name="docs")
        assert scope_fingerprint(a) != scope_fingerprint(b)

    def test_scope_fingerprint_differs_by_index_generation(self):
        a = CacheScope(collection_name="docs", index_generation="spark-4.0.0-fa33ea00")
        b = CacheScope(collection_name="docs", index_generation="spark-4.0.1-abcdef12")
        assert scope_fingerprint(a) != scope_fingerprint(b)

    def test_different_generation_does_not_share_entries(self):
        gen_a = CacheScope(collection_name="docs", index_generation="spark-4.0.0")
        gen_b = CacheScope(collection_name="docs", index_generation="spark-4.0.1")
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("q", _answer("generation A answer"), scope=gen_a)
        assert cache.get_exact("q", scope=gen_a) is not None
        assert cache.get_exact("q", scope=gen_b) is None, "Generation B must not read Generation A's cached answer"

    def test_different_tenants_do_not_share_entries(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("What is Spark?", _answer("tenant-a answer"), scope=_SPARK)
        result_a = cache.get_exact("What is Spark?", scope=_SPARK)
        result_b = cache.get_exact("What is Spark?", scope=_DELTA)
        assert result_a is not None and result_a.text == "tenant-a answer"
        assert result_b is None, "Tenant B must not read Tenant A's cached answer"

    def test_different_source_filter_produces_different_key(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("q", _answer("spark answer"), scope=_SPARK)
        assert cache.get_exact("q", scope=_SPARK) is not None
        assert cache.get_exact("q", scope=_DELTA) is None

    def test_embedding_model_change_invalidates_key(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        scope_nomic = CacheScope(embedding_model="nomic-embed-text", collection_name="docs")
        scope_bge = CacheScope(embedding_model="bge-m3", collection_name="docs")
        cache.set_exact("q", _answer(), scope=scope_nomic)
        assert cache.get_exact("q", scope=scope_nomic) is not None
        assert cache.get_exact("q", scope=scope_bge) is None

    def test_semantic_entries_scoped_by_tenant(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        vec = [0.5] * 768
        cache.set_semantic("q", vec, _answer("tenant-a semantic"), scope=_SPARK)
        assert cache.get_semantic("q", vec, scope=_SPARK) is not None
        assert cache.get_semantic("q", vec, scope=_DELTA) is None


# ------------------------------------------------------------------
# Task 12: cache-poisoning prevention
# ------------------------------------------------------------------


class TestIsCacheable:
    def test_good_answer_is_cacheable(self):
        assert QueryCache.is_cacheable(_answer()) is True

    def test_empty_text_not_cacheable(self):
        assert QueryCache.is_cacheable(CachedAnswer(text="", sources=(_source(),), confidence=0.95)) is False
        assert QueryCache.is_cacheable(CachedAnswer(text="   ", sources=(_source(),), confidence=0.95)) is False

    def test_malformed_not_cacheable(self):
        assert QueryCache.is_cacheable("not an envelope") is False  # type: ignore[arg-type]

    def test_low_confidence_not_cacheable(self):
        assert QueryCache.is_cacheable(CachedAnswer(text="ok", sources=(_source(),), confidence=0.3)) is False

    def test_no_sources_not_cacheable(self):
        assert QueryCache.is_cacheable(CachedAnswer(text="ok", confidence=0.95)) is False

    def test_insufficient_context_not_cacheable(self):
        bad = CachedAnswer(
            text="I cannot answer this question.\n\nMissing information: x", sources=(_source(),), confidence=0.8
        )
        assert QueryCache.is_cacheable(bad) is False

    def test_safety_only_not_cacheable(self):
        bad = CachedAnswer(
            text="I'm sorry, but I cannot answer this question.",
            sources=(_source(),),
            confidence=0.8,
        )
        assert QueryCache.is_cacheable(bad) is False


class TestPoisoningPrevention:
    def test_bad_answer_never_enters_exact_cache(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        bad = CachedAnswer(text="", sources=(_source(),), confidence=0.95)
        cache.set_exact("What is Spark?", bad)
        assert cache.get_exact("What is Spark?") is None

    def test_bad_answer_never_enters_semantic_cache(self):
        cache = QueryCache(exact_enabled=False, semantic_enabled=True, similarity_threshold=0.9)
        vec = [0.5] * 768
        bad = CachedAnswer(text="I cannot answer this question.", sources=(_source(),), confidence=0.8)
        cache.set_semantic("What is Spark?", vec, bad)
        assert cache.get_semantic("What is Spark?", vec) is None

    def test_insufficient_answer_never_enters_exact_cache(self):
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        bad = CachedAnswer(
            text="I cannot answer because it is outside my knowledge repository.",
            sources=(_source(),),
            confidence=0.7,
        )
        cache.set_exact("What is Delta?", bad)
        assert cache.get_exact("What is Delta?") is None

    def test_generation_scope_remains_in_cache_identity(self):
        gen_a = CacheScope(collection_name="docs", index_generation="spark-4.0.0-fa33ea00")
        gen_b = CacheScope(collection_name="docs", index_generation="spark-4.0.1-abcdef12")
        cache = QueryCache(exact_enabled=True, semantic_enabled=False, similarity_threshold=0.9)
        cache.set_exact("q", _answer("gen A"), scope=gen_a)
        assert cache.get_exact("q", scope=gen_a) is not None
        assert cache.get_exact("q", scope=gen_b) is None, "Generation scope must stay part of cache identity"
