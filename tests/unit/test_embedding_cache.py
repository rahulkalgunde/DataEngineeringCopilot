"""Tests for the CachedEmbedder embedding cache (L1 memory + L2 Redis)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from data_engineering_copilot.domain.exceptions import EmbeddingError
from data_engineering_copilot.infrastructure.embedding_cache import CachedEmbedder


class _StubEmbedder:
    """EmbedderProtocol double: records calls, returns configured vectors."""

    def __init__(self, embed_texts_result, embed_query_result=None):
        self.embed_texts_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []
        self._embed_texts_result = embed_texts_result
        self._embed_query_result = embed_query_result or [0.1, 0.2]
        self.closed = False

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(list(texts))
        return self._embed_texts_result

    async def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return self._embed_query_result

    async def close(self) -> None:
        self.closed = True


async def test_embed_texts_miss_calls_inner_embedder():
    inner = _StubEmbedder([[0.1, 0.2], [0.3, 0.4]])
    cached = CachedEmbedder(inner)

    result = await cached.embed_texts(["hello world", "goodbye world"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert inner.embed_texts_calls == [["hello world", "goodbye world"]]


async def test_embed_texts_cache_hit_skips_inner_embedder():
    inner = _StubEmbedder([[0.1, 0.2]])
    cached = CachedEmbedder(inner)

    first = await cached.embed_texts(["hello world"])
    second = await cached.embed_texts(["hello world"])

    assert first == second == [[0.1, 0.2]]
    assert inner.embed_texts_calls == [["hello world"]]


async def test_embed_texts_partial_cache_hit_embeds_only_uncached():
    inner = _StubEmbedder([[0.9, 0.8]])
    cached = CachedEmbedder(inner)

    await cached.embed_texts(["cached text"])
    result = await cached.embed_texts(["cached text", "fresh text"])

    assert result == [[0.9, 0.8], [0.9, 0.8]]
    # Only the uncached text should reach the inner embedder on the second call.
    assert inner.embed_texts_calls == [["cached text"], ["fresh text"]]


async def test_embed_texts_short_embedder_result_raises():
    """A short inner result must raise EmbeddingError, not become [] vectors.

    Regression: previously trailing unfilled slots were coerced to empty []
    vectors, which flowed to Qdrant and surfaced as a confusing dimension
    error instead of a clear count-mismatch failure.
    """
    inner = _StubEmbedder([[0.1, 0.2]])  # 1 vector for 2 texts
    cached = CachedEmbedder(inner)

    with pytest.raises(EmbeddingError, match="1 vectors for 2 texts"):
        await cached.embed_texts(["alpha", "beta"])


async def test_embed_query_caches():
    inner = _StubEmbedder([])
    cached = CachedEmbedder(inner)

    first = await cached.embed_query("what is spark")
    second = await cached.embed_query("what is spark")

    assert first == second == [0.1, 0.2]
    assert inner.embed_query_calls == ["what is spark"]


async def test_close_closes_inner():
    inner = _StubEmbedder([])
    cached = CachedEmbedder(inner)

    await cached.close()

    assert inner.closed


async def test_redis_get_failure_falls_back_to_inner():
    redis_client = AsyncMock()
    redis_client.get.side_effect = RuntimeError("redis down")
    inner = _StubEmbedder([[0.5, 0.5]])
    cached = CachedEmbedder(inner, redis_client=redis_client)

    result = await cached.embed_texts(["text"])

    assert result == [[0.5, 0.5]]
    assert inner.embed_texts_calls == [["text"]]


async def test_redis_embed_cache_key_namespaced_by_dimension():
    redis_client = AsyncMock()
    redis_client.get.side_effect = [None]
    inner = _StubEmbedder([[0.5, 0.5]])
    cached = CachedEmbedder(inner, redis_client=redis_client, embedding_dimension=2048)

    await cached.embed_texts(["text"])

    key = redis_client.set.call_args.args[0]
    assert key.startswith("embed:cache:d2048:")


async def test_legacy_same_text_vector_not_served_after_dim_switch():
    """Regression: embedding_cache keyed by text hash only, so after a model
    switch a stale 2048-dim vector for the same text was served to a 2048-dim
    pipeline (crashing the semantic cache lookup). The dimension namespace
    must make legacy entries unreachable."""

    async def fake_get(key):
        if isinstance(key, bytes):
            key = key.decode()
        if key.startswith("embed:cache:d"):
            return None
        return json.dumps([0.1] * 2048)

    redis_client = AsyncMock()
    redis_client.get.side_effect = fake_get
    inner = _StubEmbedder([[0.5, 0.5]])
    cached = CachedEmbedder(inner, redis_client=redis_client, embedding_dimension=2048)

    result = await cached.embed_texts(["text"])

    assert result == [[0.5, 0.5]]
    assert inner.embed_texts_calls == [["text"]]
