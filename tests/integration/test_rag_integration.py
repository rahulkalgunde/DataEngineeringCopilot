"""Integration tests for the full RAG pipeline.

Tests the end-to-end flow: embed query → retrieve from Qdrant → rerank →
assemble context → generate answer via Ollama.

Uses testcontainers for Qdrant and external Ollama (skipped if unreachable).

Run with: pytest tests/integration/test_rag_integration.py -v -m integration
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import Answer, DocumentChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _settings(ollama_url):
    from data_engineering_copilot.config.settings import AppSettings

    return AppSettings(
        ollama_base_url=ollama_url,
        embedding_provider="ollama",
        embedding_model_name="nomic-embed-text",
        llm_provider="ollama",
        code_llm_provider="ollama",
        code_llm_model="llama3.2:3b",
        embedding_batch_size=32,
        retrieval_top_k=10,
        max_context_chars=2000,
        confidence_threshold=0.10,
        reranker_enabled=True,
    )


@pytest.fixture(scope="module")
def _embedder(_settings):
    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

    return AsyncOllamaEmbeddings(
        model_name=_settings.embedding_model_name,
        base_url=_settings.ollama_base_url,
    )


@pytest.fixture(scope="module")
def _ollama(_settings):
    from data_engineering_copilot.infrastructure.llm_client import LLMClient

    return LLMClient(
        base_url=f"{_settings.ollama_base_url}/v1",
        model=_settings.ollama_model,
        timeout_seconds=_settings.ollama_timeout_seconds,
        extra_body={
            "options": {
                "num_ctx": _settings.ollama_num_ctx,
                "num_predict": _settings.ollama_num_predict,
            }
        },
    )


@pytest.fixture(scope="module")
def _store(_settings, qdrant_url):
    import asyncio

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    coll = f"rag_mod_{__name__.replace('.', '_')}"
    store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=coll, embedding_dimension=768)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(store.initialize())
    loop.close()
    yield store
    try:
        from qdrant_client import QdrantClient

        c = QdrantClient(url=qdrant_url, prefer_grpc=False)
        c.delete_collection(collection_name=coll)
        c.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def _populated(_store, _settings):
    """Populate the store once with 10 topic chunks."""

    import asyncio

    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

    embedder = AsyncOllamaEmbeddings(
        model_name=_settings.embedding_model_name,
        base_url=_settings.ollama_base_url,
    )

    topics = [
        (
            "Apache Spark",
            "Apache Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Scala, Java, Python, and R.",
        ),
        (
            "Spark SQL",
            "Spark SQL is a Spark module for structured data processing. It provides a programming abstraction called DataFrames and SQL.",
        ),
        (
            "Spark Streaming",
            "Spark Streaming enables scalable, high-throughput, fault-tolerant stream processing of live data streams.",
        ),
        (
            "Delta Lake",
            "Delta Lake is an open-source storage framework that brings ACID transactions to Apache Spark and big data workloads.",
        ),
        (
            "Apache Airflow",
            "Apache Airflow is a platform to programmatically author, schedule and monitor workflows defined as code.",
        ),
        (
            "Airflow DAGs",
            "A DAG (Directed Acyclic Graph) in Airflow is a collection of tasks organized with dependencies and scheduling logic.",
        ),
        (
            "Databricks",
            "Databricks is a unified analytics platform built on top of Apache Spark that provides collaborative notebooks and data pipelines.",
        ),
        (
            "Data Lakehouse",
            "The data lakehouse architecture combines the best features of data lakes and data warehouses into a single platform.",
        ),
        (
            "Structured Streaming",
            "Structured Streaming is a scalable stream processing engine built on the Spark SQL engine.",
        ),
        (
            "PySpark",
            "PySpark is the Python API for Apache Spark. It allows you to write Spark applications using Python.",
        ),
    ]

    chunks = []
    texts = []
    for i, (title, text) in enumerate(topics):
        chunk = DocumentChunk(
            chunk_id=f"rag:doc{i:03d}:chunk00",
            source_name="RAG Test Docs",
            title=title,
            url=f"https://example.com/docs/{title.lower().replace(' ', '-')}.html",
            text=text,
        )
        chunks.append(chunk)
        texts.append(text)

    loop = asyncio.new_event_loop()
    try:
        all_embs = loop.run_until_complete(embedder.embed_texts(texts))
        loop.run_until_complete(_store.upsert_chunks(chunks, all_embs))
    finally:
        loop.run_until_complete(embedder.close())
        loop.close()
    return _store, chunks


@pytest.fixture(scope="module")
def _rag(_store, _ollama, _settings):
    from data_engineering_copilot.domain.models import RagConfig
    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
    from data_engineering_copilot.services.async_rag import AsyncRagService

    embedder = AsyncOllamaEmbeddings(
        model_name=_settings.embedding_model_name,
        base_url=_settings.ollama_base_url,
    )
    return AsyncRagService(
        config=RagConfig(),
        vector_store=_store,
        llm_client=_ollama,
        embedder=embedder,
    )


# ---------------------------------------------------------------------------
# RAG Pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.rag
@pytest.mark.xdist_group("qdrant")
class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_answer_returns_answer_object(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        answer = await _rag.answer("What is Apache Spark?")
        assert isinstance(answer, Answer)
        assert isinstance(answer.text, str)
        assert isinstance(answer.sources, tuple)
        assert isinstance(answer.confidence, float)

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    async def test_answer_has_substantial_text(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        answer = await _rag.answer("What is Apache Spark?")
        assert len(answer.text) > 20, f"Answer too short: {answer.text!r}"

    @pytest.mark.asyncio
    async def test_answer_cites_sources(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        answer = await _rag.answer("What is Delta Lake?")
        assert len(answer.sources) > 0, "Should cite at least one source"

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    async def test_answer_confidence_nonnegative(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        answer = await _rag.answer("What is Apache Airflow?")
        assert answer.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_multiple_sequential_questions(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        for q in [
            "What is Apache Spark?",
            "What is Delta Lake?",
        ]:
            answer = await _rag.answer(q)
            assert len(answer.text) > 10, f"Bad answer for: {q}"

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    async def test_performance_within_bounds(self, _rag, _populated):
        import time

        store, _ = _populated
        _rag.vector_store = store
        start = time.time()
        answer = await _rag.answer("What is Apache Spark?")
        elapsed = time.time() - start
        assert elapsed < 60, f"RAG query took {elapsed:.1f}s"
        assert len(answer.text) > 0


# ---------------------------------------------------------------------------
# Edge cases (these use their own empty store)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.rag
@pytest.mark.xdist_group("qdrant")
class TestRAGEdgeCases:
    @pytest.mark.asyncio
    async def test_unrelated_question_acknowledges_gap(self, _rag, _populated):
        """An unrelated question should produce an answer that acknowledges
        the docs don't cover it (either via confidence or LLM response)."""
        store, _ = _populated
        _rag.vector_store = store
        answer = await _rag.answer("What is the capital of France?")
        text_lower = answer.text.lower()
        acknowledges_gap = any(
            phrase in text_lower
            for phrase in [
                "cannot answer",
                "does not provide",
                "does not address",
                "not covered",
                "outside",
                "does not contain",
            ]
        )
        assert acknowledges_gap or answer.confidence < 0.5, (
            f"Expected the answer to acknowledge the gap. "
            f"confidence={answer.confidence:.4f}, text={answer.text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Wire-Mocked RAG Tests (respx for httpx)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.rag
class TestRAGWireMocked:
    """RAG pipeline with respx wire-mocked LLM and embeddings."""

    @pytest.mark.serial
    @pytest.mark.asyncio
    async def test_rag_cache_hit_skips_llm(self, fresh_qdrant_store, ollama_url):
        """Real Qdrant, mock LLM client — verify cache hit skips LLM."""

        from data_engineering_copilot.config.settings import AppSettings
        from data_engineering_copilot.domain.models import DocumentChunk, RagConfig
        from data_engineering_copilot.domain.protocols import LLMClientProtocol
        from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache

        settings = AppSettings(
            _env_file=None,
            ollama_base_url=ollama_url,
            embedding_provider="ollama",
            embedding_model_name="nomic-embed-text",
            llm_provider="ollama",
            code_llm_provider="",
        )

        embedder = AsyncOllamaEmbeddings(
            model_name=settings.embedding_model_name,
            base_url=settings.ollama_base_url,
        )

        chunk = DocumentChunk(
            chunk_id="test_cache_001",
            source_name="Test",
            title="Apache Spark",
            url="https://example.com/spark.html",
            text="Apache Spark is a unified analytics engine for large-scale data processing.",
        )
        emb = await embedder.embed_texts([chunk.text])
        await fresh_qdrant_store.upsert_chunks([chunk], emb)

        call_count = 0

        class MockLLM(LLMClientProtocol):
            async def generate(self, prompt: str, **kwargs) -> str:
                nonlocal call_count
                call_count += 1
                return "Spark is an analytics engine."

            async def close(self) -> None:
                pass

        llm_client = MockLLM()
        cache = TwoTierCache(similarity_threshold=0.95)
        rag_config = RagConfig(
            confidence_threshold=0.10,
            retrieval_top_k=5,
            max_context_chars=2000,
        )

        rag_service = AsyncRagService(
            config=rag_config,
            vector_store=fresh_qdrant_store,
            llm_client=llm_client,
            embedder=embedder,
            cache=cache,
        )

        result1 = await rag_service.answer("What is Spark?")
        assert "Spark" in result1.text
        assert call_count == 1, "LLM should be called once on first query"

        result2 = await rag_service.answer("What is Spark?")
        assert result2.text == result1.text
        assert call_count == 1, "LLM should NOT be called again (cache hit)"

        await embedder.close()

    @pytest.mark.asyncio
    async def test_rag_answer_with_wire_mocked_llm(self):
        """Wire-mock Ollama generate, verify answer text comes through."""
        import respx
        from httpx import Response

        from data_engineering_copilot.config.settings import AppSettings
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        settings = AppSettings()
        with respx.mock(assert_all_mocked=False) as respx_mock:
            respx_mock.post(f"{settings.ollama_base_url}/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "This is a wire-mocked answer."}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                        "model": settings.ollama_model,
                    },
                )
            )

            client = LLMClient(
                base_url=f"{settings.ollama_base_url}/v1",
                model=settings.ollama_model,
                timeout_seconds=5,
                extra_body={
                    "options": {
                        "num_ctx": 2048,
                        "num_predict": 128,
                    }
                },
            )
            result = await client.generate("What is Spark?")
            assert "wire-mocked" in result

            await client.close()

    @pytest.mark.asyncio
    async def test_rag_embedding_with_wire_mocked(self):
        """Wire-mock Ollama embed, verify vector returned."""
        import respx
        from httpx import Response

        from data_engineering_copilot.config.settings import AppSettings
        from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

        settings = AppSettings(
            embedding_provider="ollama",
            embedding_model_name="nomic-embed-text",
        )
        dim = settings.get_embedding_dimension()
        fake_embedding = [0.01] * dim

        with respx.mock(assert_all_mocked=False) as respx_mock:
            respx_mock.post(f"{settings.ollama_base_url}/api/embed").mock(
                return_value=Response(
                    200,
                    json={"embeddings": [fake_embedding]},
                )
            )

            embedder = AsyncOllamaEmbeddings(model_name=settings.embedding_model_name)
            result = await embedder.embed_query("test query")
            assert len(result) == dim
            assert abs(result[0] - 0.01) < 0.001

            await embedder.close()
