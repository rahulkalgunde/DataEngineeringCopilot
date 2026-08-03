"""Tests for AsyncUrlRegistry."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.infrastructure.async_url_registry import AsyncUrlRegistry


@pytest.fixture
def mock_redis():
    m = MagicMock()
    m.hget = AsyncMock()
    m.hset = AsyncMock()
    m.delete = AsyncMock()
    m.expire = AsyncMock()
    return m


async def test_async_get_html_hash_found(mock_redis):
    record = json.dumps({"html_hash": "hash123", "discovered_at": 1000.0})
    mock_redis.hget.return_value = record.encode("utf-8")

    registry = AsyncUrlRegistry(mock_redis, "test_source")
    val = await registry.get_html_hash("http://example.com")
    assert val == "hash123"
    mock_redis.hget.assert_called_once_with("crawl:url_registry:test_source", "http://example.com")


async def test_async_get_html_hash_not_found(mock_redis):
    mock_redis.hget.return_value = None

    registry = AsyncUrlRegistry(mock_redis, "test_source")
    val = await registry.get_html_hash("http://example.com/notfound")
    assert val is None


async def test_async_set_html_hash(mock_redis):
    registry = AsyncUrlRegistry(mock_redis, "test_source")
    await registry.set_html_hash("http://example.com", "hash123")
    mock_redis.hset.assert_called_once()
