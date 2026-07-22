"""Integration tests for the full RAG pipeline.

Tests the end-to-end flow: embed query → retrieve from Qdrant → rerank →
assemble context → generate answer via Ollama.

Uses testcontainers for Qdrant and external Ollama (skipped if unreachable).

Run with: pytest tests/integration/test_rag_integration.py -v -m integration
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import Answer, DocumentChunk
from tests.integration.conftest import require_ollama

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _settings():
    from data_engineering_copilot.config.settings import AppSettings

    return AppSettings(
        embedding_batch_size=32,
        retrieval_top_k=10,
        max_context_chars=2000,
        confidence_threshold=0.10,
        reranker_enabled=True,
    )


@pytest.fixture(scope="module")
def _embedder(_settings):
    require_ollama()
    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

    return AsyncOllamaEmbeddings(model_name=_settings.embedding_model_name)


@pytest.fixture(scope="module")
def _ollama(_settings):
    require_ollama()
    from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient

    return AsyncOllamaClient(
        base_url=_settings.ollama_base_url,
        model=_settings.ollama_model,
        timeout_seconds=_settings.ollama_timeout_seconds,
        num_ctx=_settings.ollama_num_ctx,
        num_predict=_settings.ollama_num_predict,
    )


@pytest.fixture(scope="module")
def _store(_settings, qdrant_url):
    import asyncio

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    coll = f"rag_mod_{__name__.replace('.', '_')}"
    store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=coll)
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

    embedder = AsyncOllamaEmbeddings(model_name=_settings.embedding_model_name)

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
    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
    from data_engineering_copilot.services.async_rag import AsyncRagService

    embedder = AsyncOllamaEmbeddings(model_name=_settings.embedding_model_name)
    return AsyncRagService(vector_store=_store, ollama_client=_ollama, embedder=embedder)


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
