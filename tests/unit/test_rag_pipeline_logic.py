"""Hermetic RAG pipeline-logic tests (no infra, no network).

Ports of the former real-Qdrant/Ollama pipeline tests, running the full
retrieval → context → answer flow against deterministic doubles
(``InMemoryVectorStore`` + ``StubEmbedder`` + ``StubLLM``) so the suite runs
offline and stable.  Real-infra coverage lives in the single Ollama smoke test
in ``tests/integration/test_rag_integration.py``.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import Answer, DocumentChunk, RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService
from tests.doubles.embedder import StubEmbedder
from tests.doubles.llm import StubLLM
from tests.doubles.vector_store import InMemoryVectorStore

_TOPICS = [
    (
        "Apache Spark",
        "Apache Spark is a unified analytics engine for large-scale data processing. "
        "It provides high-level APIs in Scala, Java, Python, and R.",
    ),
    (
        "Spark SQL",
        "Spark SQL is a Spark module for structured data processing. It provides a "
        "programming abstraction called DataFrames and SQL.",
    ),
    (
        "Spark Streaming",
        "Spark Streaming enables scalable, high-throughput, fault-tolerant stream processing of live data streams.",
    ),
    (
        "Delta Lake",
        "Delta Lake is an open-source storage framework that brings ACID transactions "
        "to Apache Spark and big data workloads.",
    ),
    (
        "Apache Airflow",
        "Apache Airflow is a platform to programmatically author, schedule and monitor workflows defined as code.",
    ),
    (
        "Airflow DAGs",
        "A DAG (Directed Acyclic Graph) in Airflow is a collection of tasks organized "
        "with dependencies and scheduling logic.",
    ),
    (
        "Databricks",
        "Databricks is a unified analytics platform built on top of Apache Spark that "
        "provides collaborative notebooks and data pipelines.",
    ),
    (
        "Data Lakehouse",
        "The data lakehouse architecture combines the best features of data lakes and "
        "data warehouses into a single platform.",
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


def _build_chunks() -> list[DocumentChunk]:
    chunks = []
    for i, (title, text) in enumerate(_TOPICS):
        chunks.append(
            DocumentChunk(
                chunk_id=f"rag:doc{i:03d}:chunk00",
                source_name="RAG Test Docs",
                title=title,
                url=f"https://example.com/docs/{title.lower().replace(' ', '-')}.html",
                text=text,
            )
        )
    return chunks


@pytest.fixture
async def _rag():
    """Function-scoped hermetic RAG service (doubles only)."""
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)

    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    service = AsyncRagService(
        config=RagConfig(
            retrieval_top_k=10,
            confidence_threshold=0.10,
            max_context_chars=2000,
        ),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
    )
    yield service
    await embedder.close()
    await store.close()


@pytest.mark.asyncio
async def test_answer_returns_answer_object(_rag):
    answer = await _rag.answer("What is Apache Spark?")
    assert isinstance(answer, Answer)
    assert isinstance(answer.text, str)
    assert isinstance(answer.sources, tuple)
    assert isinstance(answer.confidence, float)


@pytest.mark.asyncio
async def test_answer_has_substantial_text(_rag):
    answer = await _rag.answer("What is Apache Spark?")
    assert len(answer.text) > 20, f"Answer too short: {answer.text!r}"


@pytest.mark.asyncio
async def test_answer_cites_sources(_rag):
    answer = await _rag.answer("What is Delta Lake?")
    assert len(answer.sources) > 0, "Should cite at least one source"


@pytest.mark.asyncio
async def test_answer_confidence_nonnegative(_rag):
    answer = await _rag.answer("What is Apache Airflow?")
    assert answer.confidence >= 0.0


@pytest.mark.asyncio
async def test_multiple_sequential_questions(_rag):
    for q in ["What is Apache Spark?", "What is Delta Lake?"]:
        answer = await _rag.answer(q)
        assert len(answer.text) > 10, f"Bad answer for: {q}"


@pytest.mark.asyncio
async def test_unrelated_question_acknowledges_gap(_rag):
    """An unrelated question should produce an answer that acknowledges the
    docs don't cover it (via the gap-acknowledging stub response)."""
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
        f"Expected the answer to acknowledge the gap. confidence={answer.confidence:.4f}, text={answer.text[:200]!r}"
    )
