"""Tests for async RAG service with query caching."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.domain.exceptions import LLMGenerationError, RetrievalError
from data_engineering_copilot.domain.models import Answer, RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.groundedness import GroundednessVerifier
from data_engineering_copilot.services.query_rewriting import QueryRewriter


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
        m.rerank = AsyncMock(return_value=[])
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
        self,
        config=None,
        vector_store=None,
        llm=None,
        embedder=None,
        reranker=None,
        telemetry=None,
        cache=None,
        code_llm=None,
    ):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        return AsyncRagService(
            config=config or RagConfig(),
            vector_store=vector_store or MagicMock(query=AsyncMock(return_value=[])),
            llm_client=llm or MagicMock(generate=AsyncMock(return_value="answer")),
            code_llm_client=code_llm,
            embedder=embedder or MagicMock(embed_query=AsyncMock(return_value=[0.1] * 768)),
            reranker=reranker,
            telemetry=telemetry,
            cache=cache,
        )

    @pytest.mark.asyncio
    async def test_answer_returns_cached_on_hit(self, mock_embedder, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.domain.models import CachedAnswer, DocumentChunk
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_cache import QueryCache

        cache = QueryCache(exact_enabled=True, semantic_enabled=False)
        source = DocumentChunk(
            chunk_id="c1",
            source_name="test",
            title="Test",
            url="http://test.com",
            text="test content",
            doc_type="guide",
        )
        cache.set_exact("what is spark", CachedAnswer(text="cached answer", sources=(source,), confidence=0.9))

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
        assert result.confidence == 0.9

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
        assert cached is not None
        assert cached.text == "generated answer"

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

        with pytest.raises(RetrievalError, match="(?i)vector store query failed"):
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
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, chunks, top_k: chunks)

        config = RagConfig(reranker_enabled=True, reranker_top_k=3)
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(), self._make_chunk()])

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
    async def test_answer_retrieval_only_skips_generation(self, mock_embedder, mock_vector_store, mock_llm):
        """``retrieval_only=True`` returns the assembled context and final
        chunk sources without invoking the LLM, groundedness verifier or scope
        verifier (used by ``dec evaluate --spark`` to score retrieval recall
        without paying for answer generation)."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_llm.generate = AsyncMock(return_value="should not be called")
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(), self._make_chunk()])
        groundedness = MagicMock()
        groundedness.async_verify_with_score = AsyncMock(return_value=(True, [], 1.0))
        scope = MagicMock()
        scope.verify_scope = AsyncMock(return_value=(True, None))

        service = AsyncRagService(
            config=RagConfig(),
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
            groundedness_verifier=groundedness,
            scope_verifier=scope,
        )

        result = await service.answer("what is spark", retrieval_only=True)
        assert result.text == ""
        assert len(result.sources) == 2
        assert result.sources[0].url == "http://test.com"
        assert result.stage_times["total"] > 0
        mock_llm.generate.assert_not_called()
        groundedness.async_verify_with_score.assert_not_called()
        scope.verify_scope.assert_not_called()

    @pytest.mark.asyncio
    async def test_answer_reranker_score_above_reranker_threshold_rescues_answer(
        self, mock_embedder, mock_vector_store, mock_llm
    ):
        """A weak embedding match (below the embedding-scale threshold) is NOT
        rejected when the cross-encoder scores it above
        ``reranker_confidence_threshold`` — the reranker is the gate when used."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_reranker = MagicMock()
        mock_reranker.is_available = MagicMock(return_value=True)

        def _reranked(query, chunks, top_k):
            top = self._make_chunk(confidence=0.15)
            for c in chunks[1:]:
                top.chunk.text += " " + c.chunk.text
            return [top] + [self._make_chunk(confidence=0.05) for _ in chunks[1:]]

        mock_reranker.rerank = AsyncMock(side_effect=_reranked)

        config = RagConfig(
            reranker_enabled=True,
            reranker_top_k=3,
            confidence_threshold=0.3,
            reranker_confidence_threshold=0.10,
        )
        # Embedding confidence 0.05 < embedding-scale 0.3 would previously
        # reject before reranking could intervene.
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.05) for _ in range(3)])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=mock_reranker,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert "knowledge repository" not in result.text
        assert result.confidence == 0.15

    @pytest.mark.asyncio
    async def test_answer_reranker_below_reranker_threshold_rejects(self, mock_embedder, mock_vector_store, mock_llm):
        """When a reranker ran, a uniformly-low (min-max normalized) top score
        below ``reranker_confidence_threshold`` rejects the answer."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_reranker = MagicMock()
        mock_reranker.is_available = MagicMock(return_value=True)
        # Emulate a normalized reranker: best chunk scores 0.04 (below 0.10 gate).
        mock_reranker.rerank = AsyncMock(
            side_effect=lambda query, chunks, top_k: [self._make_chunk(confidence=0.04) for _ in chunks]
        )

        config = RagConfig(
            reranker_enabled=True,
            reranker_top_k=3,
            confidence_threshold=0.3,
            reranker_confidence_threshold=0.10,
        )
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.04) for _ in range(3)])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=mock_reranker,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert "knowledge repository" in result.text
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_answer_without_reranker_uses_embedding_threshold(self, mock_embedder, mock_vector_store, mock_llm):
        """Without a reranker, the embedding-scale ``confidence_threshold`` is
        still the gate (fallback path)."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        config = RagConfig(
            reranker_enabled=True,
            reranker_top_k=3,
            confidence_threshold=0.3,
            reranker_confidence_threshold=0.50,
        )
        # reranker=None → rerank_used=False → gate on confidence_threshold=0.3.
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.4)])

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
        assert "knowledge repository" not in result.text

    @pytest.mark.asyncio
    async def test_answer_without_reranker_below_threshold_rejects(self, mock_embedder, mock_vector_store, mock_llm):
        """Fallback path: embedding-scale gate rejects a weak match when no
        reranker is available, even if the reranker threshold is lax."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        config = RagConfig(
            reranker_enabled=True,
            reranker_top_k=3,
            confidence_threshold=0.3,
            reranker_confidence_threshold=0.05,
        )
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.1)])

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
        assert "knowledge repository" in result.text
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_answer_skips_reranker_when_disabled(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_reranker = MagicMock()
        mock_reranker.is_available = MagicMock(return_value=True)
        mock_reranker.rerank = AsyncMock(return_value=[])

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
    async def test_answer_lazily_initializes_reranker(self, mock_embedder, mock_vector_store, mock_llm):
        """A configured-but-not-loaded reranker is loaded before first use."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        loaded = {"value": False}

        async def _fake_initialize():
            loaded["value"] = True

        mock_reranker = MagicMock()
        mock_reranker.is_available = MagicMock(side_effect=lambda: loaded["value"])
        mock_reranker.initialize = _fake_initialize
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, chunks, top_k: chunks)

        config = RagConfig(reranker_enabled=True, reranker_top_k=3)
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(), self._make_chunk()])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=mock_reranker,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert result.text.strip()
        assert loaded["value"] is True
        mock_reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_degrades_when_reranker_init_fails(self, mock_embedder, mock_vector_store, mock_llm):
        """A failed model load must not fail the answer — reranking is skipped."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        async def _fail_initialize():
            raise RuntimeError("model download failed")

        mock_reranker = MagicMock()
        mock_reranker.is_available = MagicMock(return_value=False)
        mock_reranker.initialize = _fail_initialize
        mock_reranker.rerank = AsyncMock()

        config = RagConfig(reranker_enabled=True, reranker_top_k=3)
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(), self._make_chunk()])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=mock_reranker,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert result.text.strip()
        mock_reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_answer_handles_non_async_reranker_initialize(self, mock_embedder, mock_vector_store, mock_llm):
        """A sync double whose initialize() is not awaitable must not crash."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_reranker = MagicMock()
        mock_reranker.is_available = MagicMock(return_value=False)
        mock_reranker.rerank = AsyncMock()

        config = RagConfig(reranker_enabled=True, reranker_top_k=3)
        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(), self._make_chunk()])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=mock_reranker,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert result.text.strip()
        mock_reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_answer_uses_injected_telemetry(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_telemetry = MagicMock()
        mock_trace = MagicMock()
        mock_telemetry.start_observation = MagicMock(return_value=mock_trace)
        mock_telemetry.flush_async = AsyncMock()

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
        mock_telemetry.flush_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_answer_low_confidence_invokes_review_dataset_hook(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_telemetry = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_trace.start_observation = MagicMock(return_value=mock_generation)
        mock_telemetry.start_observation = MagicMock(return_value=mock_trace)
        mock_telemetry.flush_async = AsyncMock()

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.05)])
        mock_llm.generate = AsyncMock(return_value="answer")

        calls = []

        def hook(trace_id, question, answer):
            calls.append((trace_id, question, answer))

        service = AsyncRagService(
            config=RagConfig(confidence_threshold=0.3),
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=mock_telemetry,
            cache=None,
            review_dataset_hook=hook,
        )

        result = await service.answer("low confidence question")

        assert len(calls) == 1
        trace_id, question, answer = calls[0]
        assert question == "low confidence question"
        assert answer == result.text
        assert trace_id is not None
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_answer_high_confidence_skips_review_dataset_hook(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_telemetry = MagicMock()
        mock_trace = MagicMock()
        mock_trace.start_observation = MagicMock(return_value=MagicMock())
        mock_telemetry.start_observation = MagicMock(return_value=mock_trace)
        mock_telemetry.flush_async = AsyncMock()

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.9)])
        mock_llm.generate = AsyncMock(return_value="answer")

        calls = []

        def hook(trace_id, question, answer):
            calls.append((trace_id, question, answer))

        service = AsyncRagService(
            config=RagConfig(confidence_threshold=0.3),
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=mock_telemetry,
            cache=None,
            review_dataset_hook=hook,
        )

        await service.answer("high confidence question")

        assert calls == []

    async def test_answer_trace_carries_tags_user_session_model(self, mock_embedder, mock_vector_store, mock_llm):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_telemetry = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_trace.start_observation = MagicMock(return_value=mock_generation)
        mock_telemetry.start_observation = MagicMock(return_value=mock_trace)
        mock_telemetry.flush_async = AsyncMock()

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk()])
        mock_llm.generate = AsyncMock(return_value="answer")
        mock_llm.model = "llama3.2:3b"

        service = AsyncRagService(
            config=RagConfig(),
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=mock_telemetry,
            cache=None,
        )

        await service.answer("what is spark", user_id="user-1", session_id="session-1")

        _, trace_kwargs = mock_telemetry.start_observation.call_args
        assert trace_kwargs["user_id"] == "user-1"
        assert trace_kwargs["session_id"] == "session-1"
        assert trace_kwargs["tags"] == ["app:data-engineering-copilot"]
        assert "metadata" in trace_kwargs

        generation_calls = [
            c.kwargs for c in mock_trace.start_observation.call_args_list if c.kwargs.get("as_type") == "generation"
        ]
        assert generation_calls, "expected a generation observation"
        assert any(c.get("model") == "llama3.2:3b" for c in generation_calls)

    def test_select_llm_client_no_code_llm_returns_primary(self, config):
        service = self._make_service(config=config)
        llm = service.llm_client
        assert service._select_llm_client("code_example") is llm
        assert service._select_llm_client("api_lookup") is llm
        assert service._select_llm_client("factual") is llm

    def test_select_llm_client_code_intent_returns_code_llm(self, config):
        code_llm = MagicMock()
        service = self._make_service(config=config, code_llm=code_llm)
        assert service._select_llm_client("code_example") is code_llm
        assert service._select_llm_client("api_lookup") is code_llm

    def test_select_llm_client_non_code_intent_returns_primary(self, config):
        code_llm = MagicMock()
        service = self._make_service(config=config, code_llm=code_llm)
        llm = service.llm_client
        assert service._select_llm_client("factual") is llm
        assert service._select_llm_client("how_to") is llm
        assert service._select_llm_client("comparative") is llm
        assert service._select_llm_client("debugging") is llm
        assert service._select_llm_client("unknown") is llm

    @pytest.mark.asyncio
    async def test_json_retry_on_malformed_structured_output(self, mock_embedder, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(return_value=[self._make_chunk()])

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                '{"citations": [], "confidence": 0.9}',
                '{"answer": "This is a retried answer about Apache Spark architecture.", "citations": []}',
            ]
        )

        service = AsyncRagService(
            config=config,
            vector_store=mock_vs,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert mock_llm.generate.await_count == 2
        assert "retried" in result.text

    @pytest.mark.asyncio
    async def test_skips_json_retry_when_answer_is_present(self, mock_embedder, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(return_value=[self._make_chunk()])

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"answer": "This is a valid answer about Apache Spark.", "citations": [{"source": "test"}]}'
        )

        service = AsyncRagService(
            config=config,
            vector_store=mock_vs,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        result = await service.answer("what is spark")
        assert mock_llm.generate.await_count == 1
        assert "valid" in result.text

    @pytest.mark.asyncio
    async def test_skips_json_retry_when_short_response(self, mock_embedder, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(return_value=[self._make_chunk()])

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="short")

        service = AsyncRagService(
            config=config,
            vector_store=mock_vs,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )

        await service.answer("what is spark")
        assert mock_llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_answer_routes_code_intent_to_code_llm(self, mock_embedder, mock_llm, config):
        from data_engineering_copilot.services.query_rewriting import QueryRewriter

        code_llm = MagicMock()
        code_llm.generate = AsyncMock(return_value="```python\nx = 1\n```")

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(return_value=[self._make_chunk()])

        mock_qr = MagicMock(spec=QueryRewriter)
        mock_qr.async_rewrite = AsyncMock()
        rewritten = MagicMock()
        rewritten.intent = "code_example"
        rewritten.decomposed_steps = []
        rewritten.hyde_query = None
        mock_qr.async_rewrite.return_value = rewritten
        mock_qr.expand_queries = AsyncMock(return_value=[])

        service = AsyncRagService(
            config=config,
            vector_store=mock_vs,
            llm_client=mock_llm,
            code_llm_client=code_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
            query_rewriter=mock_qr,
        )

        await service.answer("show me spark code")
        assert code_llm.generate.called
        assert not mock_llm.generate.called

    @pytest.mark.asyncio
    async def test_hyde_embedding_reused_not_regenerated(self, mock_embedder, config):
        """HyDE hypothesis is embedded once and reused across sub-queries, not re-generated per query."""
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(return_value=[self._make_chunk()])

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="answer")

        mock_qr = MagicMock()
        rewritten = MagicMock()
        rewritten.intent = "factual"
        rewritten.decomposed_steps = ()
        rewritten.hyde_query = "Spark SQL is a module for structured data."
        mock_qr.async_rewrite = AsyncMock(return_value=rewritten)
        mock_qr.expand_queries = AsyncMock(return_value=[])
        mock_qr.hyde = AsyncMock()

        service = AsyncRagService(
            config=config,
            vector_store=mock_vs,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
            query_rewriter=mock_qr,
        )

        await service.answer("what is spark sql")

        hyde_emb = [0.1] * 768
        mock_embedder.embed_query.assert_any_call("Spark SQL is a module for structured data.")
        for call in mock_vs.query.await_args_list:
            assert call.args[0] == hyde_emb
        mock_qr.hyde.assert_not_called()

    @pytest.mark.asyncio
    async def test_hyde_policy_suppressed_query_not_embedded_and_reason_recorded(self, mock_embedder, mock_llm, config):
        """When the HyDE policy suppresses a query, only original-query retrieval
        runs (HyDE text is never embedded) and the rewrite provenance records the
        disabled reason."""
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_rewriting import RewrittenQuery

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(return_value=[self._make_chunk()])

        mock_llm.generate = AsyncMock(return_value="answer")

        class _Rewriter:
            async def async_rewrite(self, query):
                return RewrittenQuery(
                    original_query=query,
                    intent="api_lookup",
                    decomposed_steps=(),
                    hyde_query="",
                    hyde_reason="intent:api_lookup",
                )

            async def expand_queries(self, query, max_variations):
                return []

        service = AsyncRagService(
            config=config,
            vector_store=mock_vs,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
            query_rewriter=cast("QueryRewriter", _Rewriter()),
        )

        details: dict[str, list[dict]] = {}

        def on_step_detail(kind: str, payload: dict) -> None:
            details.setdefault(kind, []).append(payload)

        await service.answer("what does DataFrame.filter do?", on_step_detail=on_step_detail)

        # Original query is embedded; the suppressed HyDE text is never embedded.
        mock_embedder.embed_query.assert_any_call("what does DataFrame.filter do?")
        embedded = [call.args[0] for call in mock_embedder.embed_query.await_args_list]
        assert all("hypothetical" not in text and "perfectly answer" not in text for text in embedded)
        assert len(embedded) == 1  # only the original query

        # Provenance records the disabled reason.
        assert details["rewrite"][0]["hyde_query"] == ""
        assert details["rewrite"][0]["hyde_reason"] == "intent:api_lookup"

        # No HyDE variant participates in the fused pool.
        assert details["embed"][0]["variants"] == 1

    @pytest.mark.asyncio
    async def test_hyde_policy_allowed_query_embedded_as_extra_variant(self, mock_embedder, mock_llm, config):
        """When the HyDE policy allows a query, the original query is embedded AND
        the HyDE text is embedded as an additional query variant with an empty
        suppression reason."""
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_rewriting import RewrittenQuery

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(return_value=[self._make_chunk()])

        mock_llm.generate = AsyncMock(return_value="answer")

        class _Rewriter:
            async def async_rewrite(self, query):
                return RewrittenQuery(
                    original_query=query,
                    intent="factual",
                    decomposed_steps=(),
                    hyde_query="A hypothetical paragraph about Spark.",
                    hyde_reason="",
                )

            async def expand_queries(self, query, max_variations):
                return []

        service = AsyncRagService(
            config=config,
            vector_store=mock_vs,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
            query_rewriter=cast("QueryRewriter", _Rewriter()),
        )

        details: dict[str, list[dict]] = {}

        def on_step_detail(kind: str, payload: dict) -> None:
            details.setdefault(kind, []).append(payload)

        await service.answer("what is spark", on_step_detail=on_step_detail)

        mock_embedder.embed_query.assert_any_call("what is spark")
        mock_embedder.embed_query.assert_any_call("A hypothetical paragraph about Spark.")
        assert details["embed"][0]["variants"] == 2
        assert details["rewrite"][0]["hyde_reason"] == ""

    @pytest.mark.asyncio
    async def test_service_close_is_idempotent(self, config):
        """close() closes every closable component exactly once across repeated calls."""
        from typing import cast

        from data_engineering_copilot.domain.protocols import (
            EmbedderProtocol,
            LLMClientProtocol,
            RerankerProtocol,
            VectorStoreProtocol,
        )
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_cache import QueryCache

        class _Closable:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        class _AClosable:
            def __init__(self) -> None:
                self.closed = False

            async def aclose(self) -> None:
                self.closed = True

        vector_store = _Closable()
        llm = _Closable()
        embedder = _Closable()
        cache = _AClosable()
        reranker = _Closable()

        service = AsyncRagService(
            config=config,
            vector_store=cast(VectorStoreProtocol, vector_store),
            llm_client=cast(LLMClientProtocol, llm),
            embedder=cast(EmbedderProtocol, embedder),
            reranker=cast(RerankerProtocol, reranker),
            cache=cast(QueryCache, cache),
        )

        await service.close()
        await service.close()

        assert vector_store.closed is True
        assert llm.closed is True
        assert embedder.closed is True
        assert reranker.closed is True
        assert cache.closed is True

    @pytest.mark.asyncio
    async def test_service_close_is_resilient_to_failing_component(self, config):
        """close() keeps closing remaining components when one raises."""
        from typing import cast

        from data_engineering_copilot.domain.protocols import (
            EmbedderProtocol,
            LLMClientProtocol,
            VectorStoreProtocol,
        )
        from data_engineering_copilot.services.async_rag import AsyncRagService

        class _Closable:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        class _Exploding:
            async def close(self) -> None:
                raise RuntimeError("boom")

        vector_store = _Exploding()
        llm = _Closable()
        embedder = _Closable()

        service = AsyncRagService(
            config=config,
            vector_store=cast(VectorStoreProtocol, vector_store),
            llm_client=cast(LLMClientProtocol, llm),
            embedder=cast(EmbedderProtocol, embedder),
        )

        await service.close()
        assert llm.closed is True
        assert embedder.closed is True


class TestPhase7Scoring:
    """Phase 7: richer score types (boolean cache_hit, categorical intent)."""

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
    def config(self):
        return RagConfig()

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
    async def test_answer_scores_cache_hit_boolean_and_intent_categorical(
        self, mock_embedder, mock_vector_store, mock_llm, config
    ):
        from unittest.mock import patch as mock_patch

        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_telemetry = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_trace.start_observation = MagicMock(return_value=mock_generation)
        mock_trace.trace_id = "trace-32hex"
        mock_telemetry.start_observation = MagicMock(return_value=mock_trace)
        mock_telemetry.flush_async = AsyncMock()

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.9)])
        mock_llm.generate = AsyncMock(return_value="answer")

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=mock_telemetry,
            cache=None,
        )

        with mock_patch(
            "data_engineering_copilot.services.async_rag._get_intent_config_id",
            return_value="intent-config-1",
        ):
            await service.answer("what is spark")

        scores = {call.kwargs["name"]: call.kwargs for call in mock_telemetry.score.call_args_list}
        assert "cache_hit" in scores
        assert scores["cache_hit"]["value"] is False
        assert scores["cache_hit"]["data_type"] == "BOOLEAN"
        assert "intent" in scores
        assert scores["intent"]["data_type"] == "CATEGORICAL"
        assert scores["intent"]["config_id"] == "intent-config-1"
        # Full pipeline traces are never cache hits.
        assert scores["cache_hit"]["value"] is False

    @pytest.mark.asyncio
    async def test_answer_intent_score_falls_back_to_numeric_without_config(
        self, mock_embedder, mock_vector_store, mock_llm, config
    ):
        from unittest.mock import patch as mock_patch

        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_telemetry = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_trace.start_observation = MagicMock(return_value=mock_generation)
        mock_trace.trace_id = "trace-32hex"
        mock_telemetry.start_observation = MagicMock(return_value=mock_trace)
        mock_telemetry.flush_async = AsyncMock()

        mock_vector_store.query = AsyncMock(return_value=[self._make_chunk(confidence=0.9)])
        mock_llm.generate = AsyncMock(return_value="answer")

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=mock_telemetry,
            cache=None,
        )

        with mock_patch(
            "data_engineering_copilot.services.async_rag._get_intent_config_id",
            return_value=None,
        ):
            await service.answer("what is spark")

        scores = {call.kwargs["name"]: call.kwargs for call in mock_telemetry.score.call_args_list}
        # "what is spark" classifies as factual -> numeric 0.0.
        assert "intent_label" in scores
        assert scores["intent_label"]["data_type"] == "CATEGORICAL"
        assert scores["intent_label"]["value"] == 0.0

    @pytest.mark.asyncio
    async def test_cache_hit_records_boolean_true_score(self, mock_embedder, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.domain.models import CachedAnswer, DocumentChunk
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_cache import QueryCache

        cache = QueryCache(exact_enabled=True, semantic_enabled=False)
        source = DocumentChunk(
            chunk_id="c1",
            source_name="test",
            title="Test",
            url="http://test.com",
            text="test content",
            doc_type="guide",
        )
        cache.set_exact(
            "what is spark",
            CachedAnswer(text="cached answer", sources=(source,), confidence=0.9),
        )

        mock_telemetry = MagicMock()
        mock_trace = MagicMock()
        mock_trace.trace_id = "cache-trace-32hex"
        mock_telemetry.start_observation = MagicMock(return_value=mock_trace)
        mock_telemetry.flush_async = AsyncMock()

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=mock_telemetry,
            cache=cache,
        )

        result = await service.answer("what is spark")

        assert result.text == "cached answer"
        # Cache-hit trace created with distinct name.
        name = mock_telemetry.start_observation.call_args.kwargs["name"]
        assert name == "rag-query-pipeline-cache-hit"
        # cache_hit scored as boolean true.
        score_call = mock_telemetry.score.call_args.kwargs
        assert score_call["name"] == "cache_hit"
        assert score_call["value"] is True
        assert score_call["data_type"] == "BOOLEAN"
        mock_telemetry.flush_async.assert_awaited()

    @pytest.mark.asyncio
    async def test_cache_hit_trace_failure_is_fail_open(self, mock_embedder, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.domain.models import CachedAnswer, DocumentChunk
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_cache import QueryCache

        cache = QueryCache(exact_enabled=True, semantic_enabled=False)
        source = DocumentChunk(
            chunk_id="c1",
            source_name="test",
            title="Test",
            url="http://test.com",
            text="test content",
            doc_type="guide",
        )
        cache.set_exact(
            "what is spark",
            CachedAnswer(text="cached answer", sources=(source,), confidence=0.9),
        )

        mock_telemetry = MagicMock()
        mock_telemetry.start_observation = MagicMock(side_effect=RuntimeError("boom"))
        mock_telemetry.flush_async = AsyncMock()

        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=mock_telemetry,
            cache=cache,
        )

        result = await service.answer("what is spark")
        assert result.text == "cached answer"


class TestStepDetailsAndArtifacts:
    """Behavioral contract for the visualizer step-detail callbacks and the
    debug artifacts the async RAG service attaches to ``Answer``.
    """

    @pytest.fixture
    def config(self):
        return RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.3,
            reranker_enabled=True,
            reranker_top_k=3,
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
        m.upsert_chunks = AsyncMock()
        return m

    @pytest.fixture
    def mock_llm(self):
        m = MagicMock()
        m.generate = AsyncMock(return_value="A concise answer for the user.")
        return m

    @pytest.fixture
    def reranker(self):
        m = MagicMock()
        m.is_available = MagicMock(return_value=True)
        m.rerank = AsyncMock(side_effect=lambda query, chunks, top_k: chunks)
        return m

    def _chunk(self, text="test content", confidence=0.9):
        chunk = MagicMock()
        chunk.chunk.chunk_id = "test-chunk"
        chunk.chunk.source_name = "test"
        chunk.chunk.title = "Test"
        chunk.chunk.url = "http://test.com"
        chunk.chunk.text = text
        chunk.chunk.word_count = len(text.split())
        chunk.confidence = confidence
        chunk.distance = 1.0 - confidence
        return chunk

    @pytest.mark.asyncio
    async def test_answer_emits_all_five_step_events_in_order(
        self, mock_embedder, mock_vector_store, mock_llm, reranker, config
    ):
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_rewriting import RewrittenQuery

        class _Rewriter:
            async def async_rewrite(self, query):
                return RewrittenQuery(
                    original_query="what is spark",
                    intent="code",
                    decomposed_steps=("rewritten step",),
                    hyde_query="hyde doc",
                )

            async def expand_queries(self, query, max_variations):
                return ["expanded variant"]

        mock_vector_store.query = AsyncMock(return_value=[self._chunk()])
        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=reranker,
            telemetry=None,
            cache=None,
            query_rewriter=cast("QueryRewriter", _Rewriter()),
        )

        events: list[str] = []
        result = await service.answer("what is spark", on_step=events.append)

        assert events == [
            "Rewriting query",
            "Embedding query",
            "Retrieving results",
            "Reranking results",
            "Generating answer",
        ]
        assert result.text == "A concise answer for the user."

    @pytest.mark.asyncio
    async def test_answer_emits_step_details_for_every_stage(
        self, mock_embedder, mock_vector_store, mock_llm, reranker, config
    ):
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_rewriting import RewrittenQuery

        class _Rewriter:
            async def async_rewrite(self, query):
                return RewrittenQuery(
                    original_query="what is spark",
                    intent="code",
                    decomposed_steps=("rewritten step",),
                    hyde_query="hyde doc",
                )

            async def expand_queries(self, query, max_variations):
                return ["expanded variant"]

        mock_vector_store.query = AsyncMock(return_value=[self._chunk()])
        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=reranker,
            telemetry=None,
            cache=None,
            query_rewriter=cast("QueryRewriter", _Rewriter()),
        )

        details: dict[str, list[dict]] = {}

        def on_step_detail(kind: str, payload: dict) -> None:
            details.setdefault(kind, []).append(payload)

        await service.answer("what is spark", on_step_detail=on_step_detail)

        assert set(details) == {"rewrite", "embed", "retrieve", "rerank", "generate"}
        assert details["rewrite"][0]["original_query"] == "what is spark"
        assert "expansions" in details["rewrite"][0]
        assert details["embed"][0]["variants"] >= 2
        assert details["retrieve"][0]["pool_size"] == 1
        assert details["retrieve"][0]["candidates"][0]["chunk_id"] == "test-chunk"
        assert details["rerank"][0]["enabled"] is False
        assert details["rerank"][0]["pool_size"] == 1
        assert details["generate"][0]["context_chunks"] == 1

    @pytest.mark.asyncio
    async def test_answer_populates_visualizer_artifacts(
        self, mock_embedder, mock_vector_store, mock_llm, reranker, config
    ):
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_rewriting import RewrittenQuery

        class _Rewriter:
            async def async_rewrite(self, query):
                return RewrittenQuery(
                    original_query="what is spark",
                    intent="code",
                    decomposed_steps=("rewritten step",),
                    hyde_query="hyde doc",
                )

            async def expand_queries(self, query, max_variations):
                return ["expanded variant"]

        mock_vector_store.query = AsyncMock(return_value=[self._chunk()])
        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=reranker,
            telemetry=None,
            cache=None,
            query_rewriter=cast("QueryRewriter", _Rewriter()),
        )

        result = await service.answer("what is spark")

        assert result.rewritten_query == "rewritten step"
        assert "what is spark" in result.query_variants
        assert result.intent == "code"
        assert len(result.retrieval_details) == 1
        assert result.retrieval_details[0]["source_name"] == "test"
        assert result.rerank_details["enabled"] is False
        assert result.context and "test content" in result.context
        assert result.prompt and "what is spark" in result.prompt
        assert "total" in result.stage_times

    @pytest.mark.asyncio
    async def test_groundedness_rebuild_preserves_artifacts(self, mock_embedder, mock_vector_store, mock_llm, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        class _Verifier:
            async def async_verify_with_score(self, result, chunks):
                return False, ["unsupported claim one"], 0.42

        mock_vector_store.query = AsyncMock(return_value=[self._chunk()])
        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
            groundedness_verifier=cast("GroundednessVerifier", _Verifier()),
        )

        result = await service.answer("what is spark")

        assert result.groundedness_score == 0.42
        assert result.groundedness_claims == ("unsupported claim one",)
        assert "[Note:" in result.text
        # dataclasses.replace must not drop the visualizer artifacts.
        assert result.stage_times and "total" in result.stage_times
        assert result.retrieval_details and result.retrieval_details[0]["source_name"] == "test"
        assert result.context is not None
        assert result.prompt is not None


class TestEmptyAnswerGuardrail:
    """The answer surface must never be empty: guardrails blanking the output
    (e.g. INSUFFICIENT_CONTEXT with an empty answer) must fall back to the raw
    LLM output or a clear default message."""

    @pytest.fixture
    def config(self):
        return RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.3,
            reranker_enabled=True,
            reranker_top_k=3,
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
        chunk = MagicMock()
        chunk.chunk.chunk_id = "test-chunk"
        chunk.chunk.source_name = "test"
        chunk.chunk.title = "Test"
        chunk.chunk.url = "http://test.com"
        chunk.chunk.text = "test content"
        chunk.chunk.word_count = 2
        chunk.confidence = 0.9
        chunk.distance = 0.1
        m.query = AsyncMock(return_value=[chunk])
        return m

    @pytest.mark.asyncio
    async def test_empty_insufficient_context_substitutes_default_message(
        self, mock_embedder, mock_vector_store, config
    ):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"status": "INSUFFICIENT_CONTEXT", "answer": "", "missing_info": null}'
        )

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

        assert result.text.strip()
        assert "No answer could be generated" in result.text
        assert "INSUFFICIENT_CONTEXT" not in result.text

    @pytest.mark.asyncio
    async def test_insufficient_context_with_info_appends_missing_info(self, mock_embedder, mock_vector_store, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=(
                '{"status": "INSUFFICIENT_CONTEXT", "answer": "Spark supports AQE.",'
                ' "missing_info": "spark 4 specific docs"}'
            )
        )

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

        assert "Spark supports AQE." in result.text
        assert "Missing information: spark 4 specific docs" in result.text

    @pytest.mark.asyncio
    async def test_raw_plain_text_falls_back_when_guardrails_blank_json(self, mock_embedder, mock_vector_store, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                '{"answer": "", "citations": []}',
                "Plain fallback answer about Apache Spark architecture.",
            ]
        )

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

        assert "Plain fallback answer" in result.text


class TestScopeGate:
    """Topic-scope gate: refuses answers when the retrieved context does not
    cover the question's topic; preserves answers when covered."""

    @pytest.fixture
    def config(self):
        return RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.3,
            reranker_enabled=True,
            reranker_top_k=3,
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
        chunk = MagicMock()
        chunk.chunk.chunk_id = "test-chunk"
        chunk.chunk.source_name = "test"
        chunk.chunk.title = "Test"
        chunk.chunk.url = "http://test.com"
        chunk.chunk.text = "test content"
        chunk.chunk.word_count = 2
        chunk.confidence = 0.9
        chunk.distance = 0.1
        m.query = AsyncMock(return_value=[chunk])
        return m

    def _make_service(self, scope_verifier, config, mock_embedder, mock_vector_store, mock_llm=None):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        return AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm or MagicMock(generate=AsyncMock(return_value="Spark SQL is a module.")),
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
            scope_verifier=scope_verifier,
        )

    @pytest.mark.asyncio
    async def test_refuses_when_topic_not_covered(self, mock_embedder, mock_vector_store, config):
        from data_engineering_copilot.services.async_rag import _SCOPE_REFUSAL_TEXT
        from data_engineering_copilot.services.scope_verifier import ScopeVerifier

        class _Verifier:
            async def verify(self, question, context):
                assert "spark" in question
                assert "test content" in context
                return False

        service = self._make_service(cast("ScopeVerifier", _Verifier()), config, mock_embedder, mock_vector_store)
        result = await service.answer("how does spark window functions work")
        assert result.text == _SCOPE_REFUSAL_TEXT
        assert "INSUFFICIENT_CONTEXT" in result.text
        assert "cannot answer" in result.text
        assert "Missing information:" in result.text

    @pytest.mark.asyncio
    async def test_preserves_answer_when_topic_covered(self, mock_embedder, mock_vector_store, config):
        from data_engineering_copilot.services.scope_verifier import ScopeVerifier

        class _Verifier:
            async def verify(self, question, context):
                return True

        mock_llm = MagicMock(generate=AsyncMock(return_value="Spark SQL is a module."))
        service = self._make_service(
            cast("ScopeVerifier", _Verifier()), config, mock_embedder, mock_vector_store, mock_llm
        )
        result = await service.answer("what is spark sql")
        assert "Spark SQL is a module." in result.text
        assert "INSUFFICIENT_CONTEXT" not in result.text

    @pytest.mark.asyncio
    async def test_no_verifier_leaves_answer_untouched(self, mock_embedder, mock_vector_store, config):
        from data_engineering_copilot.services.async_rag import AsyncRagService

        mock_llm = MagicMock(generate=AsyncMock(return_value="Spark SQL is a module."))
        service = AsyncRagService(
            config=config,
            vector_store=mock_vector_store,
            llm_client=mock_llm,
            embedder=mock_embedder,
            reranker=None,
            telemetry=None,
            cache=None,
        )
        result = await service.answer("what is spark sql")
        assert "Spark SQL is a module." in result.text
        assert "INSUFFICIENT_CONTEXT" not in result.text
