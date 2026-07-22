"""Tests for async RAG service with query caching."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from data_engineering_copilot.domain.models import Answer


class TestQueryCache:
    def test_cache_hit_returns_cached_answer(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache

        cache = QueryCache(ttl_seconds=60)
        cache.set("what is spark", Answer(text="cached answer", sources=(), confidence=0.9))
        result = cache.get("what is spark")
        assert result is not None
        assert result.text == "cached answer"

    def test_cache_miss_returns_none(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache

        cache = QueryCache(ttl_seconds=60)
        result = cache.get("nonexistent query")
        assert result is None

    def test_cache_expires_after_ttl(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache

        cache = QueryCache(ttl_seconds=0)
        cache.set("expiring query", Answer(text="old", sources=(), confidence=0.8))
        time.sleep(0.01)
        result = cache.get("expiring query")
        assert result is None

    def test_cache_clear(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache

        cache = QueryCache(ttl_seconds=60)
        cache.set("q1", Answer(text="a1", sources=(), confidence=0.9))
        cache.clear()
        assert cache.get("q1") is None

    def test_cache_size(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache

        cache = QueryCache(ttl_seconds=60)
        cache.set("q1", Answer(text="a1", sources=(), confidence=0.9))
        cache.set("q2", Answer(text="a2", sources=(), confidence=0.8))
        assert cache.size() == 2

    def test_cache_strips_whitespace_in_key(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache

        cache = QueryCache(ttl_seconds=60)
        cache.set("  what is spark  ", Answer(text="answer", sources=(), confidence=0.9))
        result = cache.get("what is spark")
        assert result is not None


class TestAsyncRagService:
    @pytest.mark.asyncio
    async def test_answer_returns_cached_on_hit(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache
        from data_engineering_copilot.services.async_rag import AsyncRagService

        cache = QueryCache(ttl_seconds=60)
        cached = Answer(text="cached answer", sources=(), confidence=0.9)
        cache.set("what is spark", cached)

        mock_embedder = MagicMock()
        mock_vector_store = MagicMock()
        mock_ollama = MagicMock()

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=cache,
        )

        result = await service.answer("what is spark")
        assert result.text == "cached answer"
        mock_embedder.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_answer_caches_result_on_miss(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache
        from data_engineering_copilot.services.async_rag import AsyncRagService

        cache = QueryCache(ttl_seconds=60)
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]

        mock_vector_store = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.chunk.source_name = "test"
        mock_chunk.chunk.title = "Test"
        mock_chunk.chunk.url = "http://test.com"
        mock_chunk.chunk.text = "test content"
        mock_chunk.confidence = 0.9
        mock_chunk.distance = 0.1
        mock_vector_store.query.return_value = [mock_chunk]

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = "generated answer"

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=cache,
        )

        result = await service.answer("what is spark")
        assert result.text == "generated answer"
        cached = cache.get("what is spark")
        assert cached is not None
        assert cached.text == "generated answer"

    @pytest.mark.asyncio
    async def test_answer_without_cache_still_works(self):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]

        mock_vector_store = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.chunk.source_name = "test"
        mock_chunk.chunk.title = "Test"
        mock_chunk.chunk.url = "http://test.com"
        mock_chunk.chunk.text = "test content"
        mock_chunk.confidence = 0.9
        mock_chunk.distance = 0.1
        mock_vector_store.query.return_value = [mock_chunk]

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = "fresh answer"

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert result.text == "fresh answer"

    @pytest.mark.asyncio
    async def test_answer_embedding_failure_returns_error(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = RuntimeError("embed failed")

        mock_vector_store = MagicMock()
        mock_ollama = MagicMock()

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=QueryCache(ttl_seconds=60),
        )

        result = await service.answer("what is spark")
        assert "embedding failed" in result.text.lower() or "error" in result.text.lower()
        assert result.confidence == 0.0
