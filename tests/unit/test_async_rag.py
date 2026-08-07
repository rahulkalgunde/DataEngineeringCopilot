"""Tests for async RAG service with query caching."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.domain.exceptions import LLMGenerationError, RetrievalError
from data_engineering_copilot.domain.models import Answer, RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService


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
