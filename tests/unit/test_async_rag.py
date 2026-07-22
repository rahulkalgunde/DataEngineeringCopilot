"""Tests for async RAG service with query caching."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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

        with patch("time.monotonic", return_value=1.0):
            cache.set("expiring query", Answer(text="old", sources=(), confidence=0.8))
        with patch("time.monotonic", return_value=2.0):
            result = cache.get("expiring query")
        assert result is None

    def test_cache_not_expired_within_ttl(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache

        cache = QueryCache(ttl_seconds=60)
        with patch("time.monotonic", return_value=1.0):
            cache.set("fresh query", Answer(text="fresh", sources=(), confidence=0.9))
        with patch("time.monotonic", return_value=60.0):
            result = cache.get("fresh query")
        assert result is not None
        assert result.text == "fresh"

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

    def test_cache_overwrites_same_key(self):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache

        cache = QueryCache(ttl_seconds=60)
        cache.set("q", Answer(text="first", sources=(), confidence=0.5))
        cache.set("q", Answer(text="second", sources=(), confidence=0.9))
        result = cache.get("q")
        assert result is not None
        assert result.text == "second"
        assert cache.size() == 1


class TestAsyncRagService:
    @pytest.fixture
    def mock_embedder(self):
        m = MagicMock()
        m.embed_query = AsyncMock(return_value=[0.1] * 768)
        return m

    @pytest.fixture
    def mock_vector_store(self):
        m = MagicMock()
        m.query = AsyncMock()
        m.upsert_chunks = AsyncMock()
        return m

    @pytest.fixture
    def mock_ollama(self):
        m = MagicMock()
        m.generate = AsyncMock()
        return m

    def _make_chunk(self, text="test content", confidence=0.9):
        chunk = MagicMock()
        chunk.chunk.source_name = "test"
        chunk.chunk.title = "Test"
        chunk.chunk.url = "http://test.com"
        chunk.chunk.text = text
        chunk.confidence = confidence
        chunk.distance = 1.0 - confidence
        return chunk

    @pytest.mark.asyncio
    async def test_answer_returns_cached_on_hit(self, mock_embedder, mock_vector_store, mock_ollama):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache
        from data_engineering_copilot.services.async_rag import AsyncRagService

        cache = QueryCache(ttl_seconds=60)
        cached = Answer(text="cached answer", sources=(), confidence=0.9)
        cache.set("what is spark", cached)

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=cache,
        )

        result = await service.answer("what is spark")
        assert result.text == "cached answer"
        mock_embedder.embed_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_answer_caches_result_on_miss(self, mock_embedder, mock_vector_store, mock_ollama):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache
        from data_engineering_copilot.services.async_rag import AsyncRagService

        cache = QueryCache(ttl_seconds=60)
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])
        mock_ollama.generate = AsyncMock(return_value="generated answer")

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
    async def test_answer_without_cache_still_works(self, mock_embedder, mock_vector_store, mock_ollama):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])
        mock_ollama.generate = AsyncMock(return_value="fresh answer")

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert result.text == "fresh answer"

    @pytest.mark.asyncio
    async def test_answer_embedding_failure_returns_error(self, mock_vector_store, mock_ollama):
        from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(side_effect=RuntimeError("embed failed"))

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=QueryCache(ttl_seconds=60),
        )

        result = await service.answer("what is spark")
        assert "embedding failed" in result.text.lower() or "error" in result.text.lower()
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_answer_empty_chunks_returns_outside_scope(self, mock_embedder, mock_vector_store, mock_ollama):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vector_store.query = AsyncMock(return_value=[])

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=None,
        )

        result = await service.answer("unknown question")
        assert "outside" in result.text.lower() or "knowledge repository" in result.text.lower()
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_answer_low_confidence_returns_outside_scope(self, mock_embedder, mock_vector_store, mock_ollama):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.01)])

        service = AsyncRagService(
            vector_store=mock_vector_store,
            ollama_client=mock_ollama,
            embedder=mock_embedder,
            cache=None,
        )

        result = await service.answer("low confidence question")
        assert "outside" in result.text.lower() or "knowledge repository" in result.text.lower()
