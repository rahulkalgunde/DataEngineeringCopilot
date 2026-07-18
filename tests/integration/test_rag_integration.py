"""Integration tests for the full RAG pipeline.

Tests the end-to-end flow: embed query → retrieve from Qdrant → rerank →
assemble context → generate answer via Ollama.

Uses a module-scoped populated store to avoid re-embedding for every test.

Run with: pytest tests/test_rag_integration.py -v -m integration
"""

import time

import pytest

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import Answer, DocumentChunk
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.rag import RagAnswerService
from tests.conftest import (
    require_qdrant_and_ollama,
    unique_collection_name,
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures (created once, shared by all tests in this file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _settings():
    return AppSettings(
        embedding_batch_size=32,
        retrieval_top_k=10,
        max_context_chars=2000,
        confidence_threshold=0.10,
        reranker_enabled=True,
    )


@pytest.fixture(scope="module")
def _embedder(_settings):
    require_qdrant_and_ollama()
    return SentenceTransformerEmbeddings(
        model_name=_settings.embedding_model_name,
        cache_dir=_settings.embedding_cache_dir,
        local_files_only=False,
    )


@pytest.fixture(scope="module")
def _ollama(_settings):
    require_qdrant_and_ollama()
    return OllamaClient(
        base_url=_settings.ollama_base_url,
        model=_settings.ollama_model,
        timeout_seconds=_settings.ollama_timeout_seconds,
        num_ctx=_settings.ollama_num_ctx,
        num_predict=_settings.ollama_num_predict,
    )


@pytest.fixture(scope="module")
def _store(_settings):
    require_qdrant_and_ollama()
    from qdrant_client import QdrantClient

    coll = unique_collection_name("rag_mod")
    store = QdrantVectorStore(url=_settings.qdrant_url, collection_name=coll)
    yield store
    try:
        c = QdrantClient(url=_settings.qdrant_url, prefer_grpc=False)
        c.delete_collection(collection_name=coll)
        c.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def _populated(_store, _embedder):
    """Populate the store once with 10 topic chunks."""
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

    all_embs = _embedder.embed_texts(texts)
    _store.upsert_chunks(chunks, all_embs)
    return _store, chunks


@pytest.fixture(scope="module")
def _rag(_store, _embedder, _ollama):
    return RagAnswerService(vector_store=_store, ollama_client=_ollama, embedder=_embedder)


# ---------------------------------------------------------------------------
# RAG Pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.rag
class TestRAGPipeline:
    def test_answer_returns_answer_object(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        answer = _rag.answer("What is Apache Spark?")
        assert isinstance(answer, Answer)
        assert isinstance(answer.text, str)
        assert isinstance(answer.sources, tuple)
        assert isinstance(answer.confidence, float)

    def test_answer_has_substantial_text(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        answer = _rag.answer("What is Apache Spark?")
        assert len(answer.text) > 20, f"Answer too short: {answer.text!r}"

    def test_answer_cites_sources(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        answer = _rag.answer("What is Delta Lake?")
        assert len(answer.sources) > 0, "Should cite at least one source"

    def test_answer_confidence_nonnegative(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        answer = _rag.answer("What is Apache Airflow?")
        assert answer.confidence >= 0.0

    def test_multiple_sequential_questions(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        for q in [
            "What is Apache Spark?",
            "What is Delta Lake?",
        ]:
            answer = _rag.answer(q)
            assert len(answer.text) > 10, f"Bad answer for: {q}"

    def test_performance_within_bounds(self, _rag, _populated):
        store, _ = _populated
        _rag.vector_store = store
        start = time.time()
        answer = _rag.answer("What is Apache Spark?")
        elapsed = time.time() - start
        assert elapsed < 60, f"RAG query took {elapsed:.1f}s"
        assert len(answer.text) > 0


# ---------------------------------------------------------------------------
# Edge cases (these use their own empty store)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.rag
class TestRAGEdgeCases:
    def test_unrelated_question_acknowledges_gap(self, _rag, _populated):
        """An unrelated question should produce an answer that acknowledges
        the docs don't cover it (either via confidence or LLM response)."""
        store, _ = _populated
        _rag.vector_store = store
        answer = _rag.answer("What is the capital of France?")
        # The LLM may still generate a helpful response that acknowledges
        # the docs don't cover it, even if confidence is above threshold.
        # Accept if LLM explicitly says it's not covered.
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
