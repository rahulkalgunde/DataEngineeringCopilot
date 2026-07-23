"""Tests for async RAG service with query caching."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.domain.exceptions import LLMGenerationError, RetrievalError
from data_engineering_copilot.domain.models import Answer, RagConfig


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
    def config(self):
        return RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.3,
            reranker_enabled=False,
            max_context_chars=4000,
        )

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
    def mock_llm(self):
        m = MagicMock()
        m.generate = AsyncMock()
        return m

    @pytest.fixture
    def mock_reranker(self):
        m = MagicMock()
        m.rerank = MagicMock(return_value=[])
        m.is_available = MagicMock(return_value=False)
        return m

    @pytest.fixture
    def mock_telemetry(self):
        m = MagicMock()
        m.start_observation = MagicMock(return_value=MagicMock())
        m.flush = MagicMock()
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

    def _make_service(
        self, config=None, vector_store=None, llm=None, embedder=None, reranker=None, telemetry=None, cache=None
    ):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        return AsyncRagService(
            config=config or RagConfig(),
            vector_store=vector_store or MagicMock(query=AsyncMock(return_value=[])),
            llm_client=llm or MagicMock(generate=AsyncMock(return_value="answer")),
            embedder=embedder or MagicMock(embed_query=AsyncMock(return_value=[0.1] * 768)),
            reranker=reranker,
            telemetry=telemetry,
            cache=cache,
        )

    @pytest.mark.asyncio
    async def test_answer_returns_cached_on_hit(self, mock_embedder, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_cache import QueryCache

        cache = QueryCache(exact_enabled=True, semantic_enabled=False)
        cache.set_exact("what is spark", "cached answer")

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=cache,
        )

        result = await service.answer("what is spark")
        assert result.text == "cached answer"

    @pytest.mark.asyncio
    async def test_answer_caches_result_on_miss(self, mock_embedder, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_cache import QueryCache

        cache = QueryCache(exact_enabled=True, semantic_enabled=False)
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])
        mock_llm.generate = AsyncMock(return_value="generated answer")

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=cache,
        )

        result = await service.answer("what is spark")
        assert result.text == "generated answer"
        cached = cache.get_exact("what is spark")
        assert cached == "generated answer"

    @pytest.mark.asyncio
    async def test_answer_without_cache_still_works(self, mock_embedder, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])
        mock_llm.generate = AsyncMock(return_value="fresh answer")

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert result.text == "fresh answer"

    @pytest.mark.asyncio
    async def test_answer_embedding_failure_raises_retrieval_error(self, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(side_effect=RuntimeError("embed failed"))

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        with pytest.raises(RetrievalError, match="(?i)embedding"):
            await service.answer("what is spark")

    @pytest.mark.asyncio
    async def test_answer_empty_chunks_returns_outside_scope(self, mock_embedder, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vector_store.query = AsyncMock(return_value=[])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("unknown question")
        assert "outside" in result.text.lower() or "knowledge repository" in result.text.lower()
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_answer_low_confidence_returns_outside_scope(
        self, mock_embedder, mock_vector_store, mock_llm, config
    ):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.01)])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("low confidence question")
        assert "outside" in result.text.lower() or "knowledge repository" in result.text.lower()

    @pytest.mark.asyncio
    async def test_answer_vector_store_failure_raises_retrieval_error(self, mock_embedder, mock_llm, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(side_effect=RuntimeError("connection lost"))

        service = AsyncRagService(
            config=config,
            vector_store=mock_vs,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        with pytest.raises(RetrievalError, match="(?i)vector store"):
            await service.answer("what is spark")

    @pytest.mark.asyncio
    async def test_answer_llm_failure_raises_generation_error(self, mock_embedder, mock_vector_store, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("model overloaded"))
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        with pytest.raises(LLMGenerationError, match="(?i)generation"):
            await service.answer("what is spark")

    @pytest.mark.asyncio
    async def test_answer_uses_config_values_not_global_settings(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        custom_config = RagConfig(
            retrieval_top_k=10,
            confidence_threshold=0.5,
            max_context_chars=2000,
        )
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.6)])

        service = AsyncRagService(
            config=custom_config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        await service.answer("what is spark")
        mock_vector_store.query.assert_awaited_once()
        call_kwargs = mock_vector_store.query.call_args
        assert call_kwargs[1].get("top_k") == 10 or call_kwargs[0][1] == 10

    @pytest.mark.asyncio
    async def test_answer_uses_injected_reranker(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_reranker = MagicMock()
        mock_reranker.is_available = MagicMock(return_value=True)
        mock_reranker.rerank = MagicMock(return_value=[self._make_chunk()])

        config = RagConfig(reranker_enabled=True, reranker_top_k=3)
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=mock_reranker,
            telemetry=None,
            cache=None,
        )

        await service.answer("what is spark")
        mock_reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_skips_reranker_when_disabled(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_reranker = MagicMock()
        mock_reranker.is_available = MagicMock(return_value=True)
        mock_reranker.rerank = MagicMock(return_value=[])

        config = RagConfig(reranker_enabled=False)
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=mock_reranker,
            telemetry=None,
            cache=None,
        )

        await service.answer("what is spark")
        mock_reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_answer_uses_injected_telemetry(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_telemetry = MagicMock()
        mock_trace = MagicMock()
        mock_telemetry.start_observation = MagicMock(return_value=mock_trace)

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])
        mock_llm.generate = AsyncMock(return_value="answer")

        service = AsyncRagService(
            config=RagConfig(),
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=mock_telemetry,
            cache=None,
        )

        await service.answer("what is spark")
        mock_telemetry.start_observation.assert_called()
        mock_telemetry.flush.assert_called_once()
