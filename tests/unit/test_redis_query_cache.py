"""Tests for RedisQueryCache — Redis-backed two-tier query cache.

Uses a fake async Redis client to avoid external Redis dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.services.redis_query_cache import RedisQueryCache


@pytest.fixture
def fake_redis():
    """Create a fake async Redis client that mimics basic get/set/scan/hset."""
    store: dict[str, str | dict] = {}
    counter = [0]

    class FakeRedis:
        async def get(self, key):
            val = store.get(key)
            if isinstance(val, str):
                return val
            return None

        async def setex(self, key, ttl, value):
            store[key] = value

        async def delete(self, *keys):
            for k in keys:
                store.pop(k, None)

        async def incr(self, key):
            counter[0] += 1
            return counter[0]

        async def hset(self, key, mapping):
            store[key] = mapping

        async def hgetall(self, key):
            val = store.get(key)
            if isinstance(val, dict):
                return val
            return {}

        async def scan(self, cursor=0, match=None, count=10):
            matching = [k for k in store if match is None or (match.endswith("*") and k.startswith(match[:-1]))]
            return (0, matching)

        async def expire(self, key, ttl):
            pass

        async def close(self):
            store.clear()

    return FakeRedis()


@pytest.fixture
def cache(fake_redis):
    with patch("redis.asyncio.from_url", return_value=fake_redis):
        c = RedisQueryCache(redis_url="redis://localhost:6379/0")
        yield c


class TestInit:
    def test_creates_redis_client(self):
        mock_redis = MagicMock()
        with patch("redis.asyncio.from_url", return_value=mock_redis) as mock_from_url:
            c = RedisQueryCache(redis_url="redis://test:6379/0")
            mock_from_url.assert_called_once_with("redis://test:6379/0", decode_responses=True)
            assert c._exact_ttl == 3600
            assert c._semantic_ttl == 7200
            assert c._similarity_threshold == 0.92


class TestExactCache:
    async def test_set_then_get(self, cache):
        await cache.set_exact("what is spark", "Spark is an engine")
        result = await cache.get_exact("what is spark")
        assert result == "Spark is an engine"

    async def test_miss_returns_none(self, cache):
        result = await cache.get_exact("unknown")
        assert result is None

    async def test_case_insensitive_normalization(self, cache):
        await cache.set_exact("What IS Spark?", "answer")
        result = await cache.get_exact("what is spark")
        assert result == "answer"

    async def test_punctuation_stripped(self, cache):
        await cache.set_exact("hello, world!", "answer")
        result = await cache.get_exact("hello world")
        assert result == "answer"


class TestSemanticCache:
    async def test_miss_when_empty(self, cache):
        result = await cache.get_semantic([0.1] * 768)
        assert result is None

    async def test_set_then_hit(self, cache):
        await cache.set_semantic("q", [1.0] * 768, "answer")
        result = await cache.get_semantic([0.99] * 768)
        assert result == "answer"

    async def test_orthogonal_vector_misses(self, cache):
        await cache.set_semantic("q", [1.0] * 768, "answer")
        result = await cache.get_semantic([-1.0] * 768)
        assert result is None

    async def test_zero_norm_returns_none(self, cache):
        result = await cache.get_semantic([0.0] * 768)
        assert result is None

    async def test_wrong_dim_stored_vector_skipped(self, cache):
        await cache.set_semantic("q", [1.0] * 768, "answer")
        result = await cache.get_semantic([0.99] * 2048)
        assert result is None


class TestCombined:
    async def test_exact_hit_skips_semantic(self, cache):
        await cache.set_exact("q", "exact_answer")
        result = await cache.get("q")
        assert result == "exact_answer"

    async def test_exact_miss_falls_back_to_semantic(self, cache):
        await cache.set_semantic("q", [1.0] * 768, "semantic_answer")
        result = await cache.get("q", query_embedding=[0.99] * 768)
        assert result == "semantic_answer"

    async def test_both_miss_returns_none(self, cache):
        result = await cache.get("unknown")
        assert result is None

    async def test_set_stores_in_both_tiers(self, cache):
        await cache.set("q", "answer", query_embedding=[1.0] * 768)
        exact = await cache.get_exact("q")
        semantic = await cache.get_semantic([1.0] * 768)
        assert exact == "answer"
        assert semantic == "answer"

    async def test_set_without_embedding_skips_semantic(self, cache):
        await cache.set("q", "answer")
        exact = await cache.get_exact("q")
        assert exact == "answer"


class TestClose:
    async def test_close_cleans_up(self, fake_redis):
        with patch("redis.asyncio.from_url", return_value=fake_redis):
            cache = RedisQueryCache()
        await cache.close()
