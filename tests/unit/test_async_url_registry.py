"""Tests for AsyncUrlRegistry."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from data_engineering_copilot.infrastructure.async_url_registry import AsyncUrlRegistry


@pytest.fixture
def mock_async_redis():
    mock = AsyncMock()
    return mock


async def test_async_get_html_hash_found(mock_async_redis):
    record = json.dumps({"html_hash": "hash123", "discovered_at": 1000.0})
    mock_async_redis.hget = AsyncMock(return_value=record.encode("utf-8"))

    registry = AsyncUrlRegistry(mock_async_redis, "test_source")
    val = await registry.get_html_hash("http://example.com")
    assert val == "hash123"
    mock_async_redis.hget.assert_awaited_once_with("crawl:url_registry:test_source", "http://example.com")


async def test_async_get_html_hash_not_found(mock_async_redis):
    mock_async_redis.hget = AsyncMock(return_value=None)

    registry = AsyncUrlRegistry(mock_async_redis, "test_source")
    val = await registry.get_html_hash("http://example.com/notfound")
    assert val is None


async def test_async_set_html_hash(mock_async_redis):
    mock_async_redis.hset = AsyncMock()

    registry = AsyncUrlRegistry(mock_async_redis, "test_source")
    await registry.set_html_hash("http://example.com", "hash123")
    mock_async_redis.hset.assert_awaited_once()
