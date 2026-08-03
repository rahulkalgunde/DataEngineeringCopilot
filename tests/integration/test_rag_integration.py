"""Integration tests for the full RAG pipeline.

Tests the end-to-end flow: embed query → retrieve from Qdrant → rerank →
assemble context → generate answer.

Uses testcontainers for Qdrant and external Ollama (skipped if unreachable).

Run with: pytest tests/integration/test_rag_integration.py -v -m integration

Hermetic pipeline-logic coverage (the same assertions, fully offline) lives in
``tests/unit/test_rag_pipeline_logic.py`` using tests.doubles.
"""

from __future__ import annotations

from typing import cast

import pytest

from data_engineering_copilot.domain.models import Answer, DocumentChunk
from data_engineering_copilot.domain.protocols import VectorStoreProtocol

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _settings(ollama_url):
    from tests.conftest import make_settings

    return make_settings(
        ollama_base_url=ollama_url,
        embedding_model_name="nomic-embed-text",
        code_llm_provider="ollama",
        code_llm_model="llama3.2:3b",
        embedding_batch_size=32,
        retrieval_top_k=10,
        max_context_chars=2000,
        confidence_threshold=0.10,
        reranker_enabled=True,
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
        _store.fit_bm25(texts)
    finally:
        loop.run_until_complete(embedder.close())
        loop.close()
    return _store, chunks


@pytest.fixture
async def _rag_real(_store, _settings, _ollama):
    """Function-scoped RAG service with the real Ollama LLM (quality smoke test)."""

    from data_engineering_copilot.domain.models import RagConfig
    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.async_rag import AsyncRagService

    store = AsyncQdrantVectorStore(
        url=_store._url,
        collection_name=_store._collection_name,
        embedding_dimension=768,
    )
    await store.initialize()
    embedder = AsyncOllamaEmbeddings(
        model_name=_settings.embedding_model_name,
        base_url=_settings.ollama_base_url,
    )
    service = AsyncRagService(
        config=RagConfig(),
        vector_store=cast(VectorStoreProtocol, store),
        llm_client=_ollama,
        embedder=embedder,
    )
    yield service
    await embedder.close()
    await store.close()


# ---------------------------------------------------------------------------
# Real-LLM smoke test (quality coverage; kept minimal since model output is variable)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.rag
@pytest.mark.slow
@pytest.mark.xdist_group("qdrant")
class TestRAGPipelineRealLLM:
    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    async def test_real_llm_end_to_end(self, _rag_real, _populated):
        answer = await _rag_real.answer("What is Apache Spark?")
        assert isinstance(answer, Answer)
        assert len(answer.text) > 0


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

        from data_engineering_copilot.domain.models import DocumentChunk, RagConfig
        from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
        from tests.conftest import make_settings
        from tests.doubles.llm import StaticLLM

        settings = make_settings(
            ollama_base_url=ollama_url,
            embedding_model_name="nomic-embed-text",
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

        llm_client = StaticLLM(answer="Spark is an analytics engine.")
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
        assert llm_client.call_count == 1, "LLM should be called once on first query"

        result2 = await rag_service.answer("What is Spark?")
        assert result2.text == result1.text
        assert llm_client.call_count == 1, "LLM should NOT be called again (cache hit)"

        await embedder.close()

    @pytest.mark.asyncio
    async def test_rag_answer_with_wire_mocked_llm(self):
        """Wire-mock Ollama generate, verify answer text comes through."""
        import respx
        from httpx import Response

        from data_engineering_copilot.infrastructure.llm_client import LLMClient
        from tests.conftest import make_settings

        settings = make_settings()
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

        from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
        from tests.conftest import make_settings

        settings = make_settings(
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


class TestHybridSearch:
    """Integration tests for hybrid (dense + sparse BM25) search."""

    async def test_hybrid_query_returns_results(self, _store, _populated):
        """Hybrid query with both dense and sparse vectors returns results."""
        store = _store
        assert store._bm25 is not None

        # BM25 should be fitted
        assert store._bm25.is_frozen

        # Query should use hybrid search (dense + sparse)
        fake_emb = [0.01] * 768

        results = await store.query(
            query_embedding=fake_emb,
            top_k=5,
            query_text="Apache Spark",
        )

        assert len(results) > 0
        assert all(hasattr(r, "chunk") for r in results)
