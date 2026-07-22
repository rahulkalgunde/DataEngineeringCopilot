"""Tests for QueryCache TTL expiration and edge cases."""

from __future__ import annotations

from unittest.mock import patch

from data_engineering_copilot.domain.models import Answer
from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache


class TestQueryCacheTTL:
    def test_entry_expires_after_ttl(self):
        cache = QueryCache(ttl_seconds=30)
        with patch("time.monotonic", return_value=100.0):
            cache.set("q", Answer(text="a", sources=(), confidence=0.9))
        with patch("time.monotonic", return_value=131.0):
            assert cache.get("q") is None

    def test_entry_valid_within_ttl(self):
        cache = QueryCache(ttl_seconds=30)
        with patch("time.monotonic", return_value=100.0):
            cache.set("q", Answer(text="a", sources=(), confidence=0.9))
        with patch("time.monotonic", return_value=129.0):
            assert cache.get("q") is not None

    def test_entry_exactly_at_ttl_boundary(self):
        cache = QueryCache(ttl_seconds=30)
        with patch("time.monotonic", return_value=100.0):
            cache.set("q", Answer(text="a", sources=(), confidence=0.9))
        with patch("time.monotonic", return_value=130.001):
            assert cache.get("q") is None

    def test_expired_entry_removed_from_store(self):
        cache = QueryCache(ttl_seconds=0)
        with patch("time.monotonic", return_value=1.0):
            cache.set("q", Answer(text="a", sources=(), confidence=0.9))
        with patch("time.monotonic", return_value=2.0):
            cache.get("q")
        assert cache.size() == 0

    def test_zero_ttl_immediate_expiry(self):
        cache = QueryCache(ttl_seconds=0)
        with patch("time.monotonic", return_value=1.0):
            cache.set("q", Answer(text="a", sources=(), confidence=0.9))
        with patch("time.monotonic", return_value=1.001):
            assert cache.get("q") is None

    def test_large_ttl_stays_valid(self):
        cache = QueryCache(ttl_seconds=86400)
        with patch("time.monotonic", return_value=1.0):
            cache.set("q", Answer(text="a", sources=(), confidence=0.9))
        with patch("time.monotonic", return_value=86400.0):
            assert cache.get("q") is not None


class TestQueryCacheConcurrency:
    def test_concurrent_set_get(self):
        cache = QueryCache(ttl_seconds=60)
        for i in range(100):
            cache.set(f"q{i}", Answer(text=f"a{i}", sources=(), confidence=0.9))
        assert cache.size() == 100
        for i in range(100):
            result = cache.get(f"q{i}")
            assert result is not None
            assert result.text == f"a{i}"

    def test_concurrent_overwrites(self):
        cache = QueryCache(ttl_seconds=60)
        for _ in range(50):
            cache.set("same_key", Answer(text="updated", sources=(), confidence=0.9))
        assert cache.size() == 1
        assert cache.get("same_key").text == "updated"


class TestQueryCacheSizeLimit:
    def test_no_size_limit_by_default(self):
        cache = QueryCache(ttl_seconds=60)
        for i in range(1000):
            cache.set(f"q{i}", Answer(text=f"a{i}", sources=(), confidence=0.9))
        assert cache.size() == 1000
