"""Unit tests for UrlRegistry — Redis-backed URL state store.

Tests cover:
- Basic CRUD: set/get html_hash
- Cache miss: get returns None for unknown URL
- Overwrite: update existing hash
- get_all_urls: returns full url→hash dict
- clear: removes all entries
- count: returns correct count
- Edge cases: empty registry, special characters, determinism
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

from data_engineering_copilot.infrastructure.url_registry import UrlRegistry


def _make_record(html_hash: str, discovered_at: float | None = None) -> str:
    return json.dumps(
        {
            "html_hash": html_hash,
            "discovered_at": discovered_at or time.time(),
        }
    )


class TestUrlRegistryCRUD:
    """Basic get/set operations."""

    def test_set_and_get_html_hash(self) -> None:
        mock_redis = MagicMock()
        url = "https://example.com/page1"
        record = _make_record("abc123")
        mock_redis.hget.return_value = record.encode("utf-8")

        registry = UrlRegistry(mock_redis, "test_source")
        registry.set_html_hash(url, "abc123")
        mock_redis.hset.assert_called_once()

        result = registry.get_html_hash(url)
        assert result == "abc123"

    def test_get_returns_none_for_missing_url(self) -> None:
        mock_redis = MagicMock()
        mock_redis.hget.return_value = None
        registry = UrlRegistry(mock_redis, "test_source")

        result = registry.get_html_hash("https://example.com/missing")
        assert result is None

    def test_overwrite_html_hash(self) -> None:
        mock_redis = MagicMock()
        url = "https://example.com/page1"
        registry = UrlRegistry(mock_redis, "test_source")

        registry.set_html_hash(url, "old_hash")
        registry.set_html_hash(url, "new_hash")

        mock_redis.hget.return_value = _make_record("new_hash").encode("utf-8")
        result = registry.get_html_hash(url)
        assert result == "new_hash"


class TestUrlRegistryBulk:
    """Bulk operations."""

    def test_get_all_urls(self) -> None:
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {
            b"https://example.com/a": _make_record("hash_a").encode("utf-8"),
            b"https://example.com/b": _make_record("hash_b").encode("utf-8"),
        }
        registry = UrlRegistry(mock_redis, "test_source")

        result = registry.get_all_urls()
        assert result == {
            "https://example.com/a": "hash_a",
            "https://example.com/b": "hash_b",
        }

    def test_clear(self) -> None:
        mock_redis = MagicMock()
        registry = UrlRegistry(mock_redis, "test_source")
        registry.clear()
        mock_redis.delete.assert_called_once_with("crawl:url_registry:test_source")

    def test_count(self) -> None:
        mock_redis = MagicMock()
        mock_redis.hlen.return_value = 42
        registry = UrlRegistry(mock_redis, "test_source")
        assert registry.count() == 42


class TestUrlRegistryEdgeCases:
    """Edge cases and special scenarios."""

    def test_empty_registry(self) -> None:
        mock_redis = MagicMock()
        mock_redis.hget.return_value = None
        mock_redis.hgetall.return_value = {}
        mock_redis.hlen.return_value = 0

        registry = UrlRegistry(mock_redis, "test_source")
        assert registry.get_html_hash("https://example.com/anything") is None
        assert registry.get_all_urls() == {}
        assert registry.count() == 0

    def test_special_characters_in_url(self) -> None:
        mock_redis = MagicMock()
        url = "https://example.com/page?q=search&lang=en#section"
        registry = UrlRegistry(mock_redis, "test_source")

        registry.set_html_hash(url, "special_hash")
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == "crawl:url_registry:test_source"
        assert call_args[0][1] == url

    def test_hash_is_string(self) -> None:
        mock_redis = MagicMock()
        mock_redis.hget.return_value = _make_record("abcdef1234567890").encode("utf-8")
        registry = UrlRegistry(mock_redis, "test_source")

        result = registry.get_html_hash("https://example.com/page")
        assert isinstance(result, str)
        assert result == "abcdef1234567890"

    def test_hash_plain_string_returned(self) -> None:
        mock_redis = MagicMock()
        mock_redis.hget.return_value = _make_record("plain_string_hash").encode("utf-8")
        registry = UrlRegistry(mock_redis, "test_source")

        result = registry.get_html_hash("https://example.com/page")
        assert result == "plain_string_hash"
