"""Shared fixtures and health checks for all tests.

Provides:
- Service health-check functions (Qdrant, Ollama, Langfuse)
- Auto-skip decorators when services are unreachable
- Unique collection names per test for isolation
- Teardown fixtures that clean up after each test
- Reusable component fixtures (settings, embeddings, vector store, etc.)
"""

import asyncio
import json
import pathlib
import sys
import urllib.error
import urllib.request
import uuid

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# ---------------------------------------------------------------------------
# Health-check helpers
# ---------------------------------------------------------------------------


def _qdrant_is_reachable(url: str = "http://localhost:6333", timeout: int = 3) -> bool:
    """Return True if Qdrant /collections endpoint responds."""
    try:
        req = urllib.request.Request(f"{url}/collections", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_is_reachable(url: str = "http://localhost:11434", timeout: int = 3) -> bool:
    """Return True if Ollama /api/tags endpoint responds."""
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _langfuse_is_reachable(url: str = "http://localhost:3000", timeout: int = 3) -> bool:
    """Return True if Langfuse health endpoint responds OK."""
    try:
        health_url = f"{url.rstrip('/')}/api/public/health"
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "OK"
    except Exception:
        pass
    return False


# Module-level caches so we only hit each service once per test session
_qdrant_ok: bool | None = None
_ollama_ok: bool | None = None
_langfuse_ok: bool | None = None


def qdrant_available() -> bool:
    global _qdrant_ok
    if _qdrant_ok is None:
        _qdrant_ok = _qdrant_is_reachable()
    return _qdrant_ok


def ollama_available() -> bool:
    global _ollama_ok
    if _ollama_ok is None:
        _ollama_ok = _ollama_is_reachable()
    return _ollama_ok


def langfuse_available() -> bool:
    global _langfuse_ok
    if _langfuse_ok is None:
        _langfuse_ok = _langfuse_is_reachable()
    return _langfuse_ok


def require_qdrant():
    if not qdrant_available():
        pytest.skip("Qdrant is not reachable — skipping test")


def require_ollama():
    if not ollama_available():
        pytest.skip("Ollama is not reachable — skipping test")


def require_langfuse():
    if not langfuse_available():
        pytest.skip("Langfuse is not reachable — skipping test")


def require_qdrant_and_ollama():
    require_qdrant()
    require_ollama()


# ---------------------------------------------------------------------------
# Unique collection name generator
# ---------------------------------------------------------------------------


def unique_collection_name(prefix: str = "test") -> str:
    """Generate a unique collection name to isolate tests."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Pytest hooks – auto-skip integration tests when services are down
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration-marked tests when required services are unreachable."""
    for item in items:
        markers = {m.name for m in item.iter_markers()}

        if "qdrant" in markers and not qdrant_available():
            item.add_marker(pytest.mark.skip(reason="Qdrant is not reachable"))

        if "ollama" in markers and not ollama_available():
            item.add_marker(pytest.mark.skip(reason="Ollama is not reachable"))

        if "langfuse" in markers and not langfuse_available():
            item.add_marker(pytest.mark.skip(reason="Langfuse is not reachable"))

        if "rag" in markers and (not qdrant_available() or not ollama_available()):
            item.add_marker(pytest.mark.skip(reason="RAG tests require Qdrant + Ollama"))

        if "ingestion" in markers and (not qdrant_available() or not ollama_available()):
            item.add_marker(pytest.mark.skip(reason="Ingestion tests require Qdrant + Ollama"))


# ---------------------------------------------------------------------------
# Shared fixtures – Settings
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_settings():
    """AppSettings tuned for integration testing."""
    from data_engineering_copilot.config.settings import AppSettings

    return AppSettings(
        embedding_batch_size=32,
        retrieval_top_k=5,
        max_context_chars=2000,
        confidence_threshold=0.10,
        reranker_enabled=True,
        chunk_size_words=200,
        chunk_overlap_words=40,
        ingestion_batch_chunk_size=64,
    )


# ---------------------------------------------------------------------------
# Shared fixtures – Embeddings
# ---------------------------------------------------------------------------


@pytest.fixture
def embeddings_provider(integration_settings):
    """Real Ollama embeddings provider (async wrapper). Skips if Ollama unreachable."""
    require_ollama()
    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

    return AsyncOllamaEmbeddings(
        model_name=integration_settings.embedding_model_name,
    )


# ---------------------------------------------------------------------------
# Shared fixtures – Qdrant Vector Store (with unique collection + teardown)
# ---------------------------------------------------------------------------


@pytest.fixture
def qdrant_store(integration_settings):
    """Create an AsyncQdrantVectorStore with a unique collection name.

    Tears down the collection after the test.
    """
    require_qdrant()
    from qdrant_client import QdrantClient

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    coll_name = unique_collection_name("itest")
    store = AsyncQdrantVectorStore(
        url=integration_settings.qdrant_url,
        collection_name=coll_name,
    )
    asyncio.run(store.initialize())
    yield store

    # Teardown: delete the collection
    try:
        client = QdrantClient(url=integration_settings.qdrant_url, prefer_grpc=False)
        client.delete_collection(collection_name=coll_name)
        client.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared fixtures – Ollama Client
# ---------------------------------------------------------------------------


@pytest.fixture
def ollama_client(integration_settings):
    """Real Ollama async client. Skips if Ollama is unreachable."""
    require_ollama()
    from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient

    return AsyncOllamaClient(
        base_url=integration_settings.ollama_base_url,
        model=integration_settings.ollama_model,
        timeout_seconds=integration_settings.ollama_timeout_seconds,
        num_ctx=integration_settings.ollama_num_ctx,
        num_predict=integration_settings.ollama_num_predict,
    )


# ---------------------------------------------------------------------------
# Shared fixtures – HTML Parser & Chunker
# ---------------------------------------------------------------------------


@pytest.fixture
def html_parser():

    from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser

    return MarkdownParser()


@pytest.fixture
def chunker(integration_settings):
    from data_engineering_copilot.services.chunker import DocumentChunker

    return DocumentChunker(
        chunk_size_words=integration_settings.chunk_size_words,
        overlap_words=integration_settings.chunk_overlap_words,
    )


# ---------------------------------------------------------------------------
# Shared fixtures – Populated Vector Store
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_store(qdrant_store, embeddings_provider):
    """Qdrant store pre-populated with 10 diverse chunks for RAG testing.

    Returns (store, chunks) tuple.
    """
    from data_engineering_copilot.domain.models import DocumentChunk

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
            chunk_id=f"itest:doc{i:03d}:chunk00",
            source_name="Integration Test Docs",
            title=title,
            url=f"https://example.com/docs/{title.lower().replace(' ', '-')}.html",
            text=text,
        )
        chunks.append(chunk)
        texts.append(text)

    # Batch embed all texts in one call for speed
    all_embeddings = asyncio.run(embeddings_provider.embed_texts(texts))

    asyncio.run(qdrant_store.upsert_chunks(chunks, all_embeddings))
    return qdrant_store, chunks


# ---------------------------------------------------------------------------
# Shared fixtures – RAG Service
# ---------------------------------------------------------------------------


@pytest.fixture
def rag_service(integration_settings, embeddings_provider, ollama_client, qdrant_store):
    """AsyncRagService wired to real components."""
    from data_engineering_copilot.services.async_rag import AsyncRagService

    return AsyncRagService(
        vector_store=qdrant_store,
        llm_client=ollama_client,
        embedder=embeddings_provider,
    )


# ---------------------------------------------------------------------------
# Shared fixtures – FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client():
    """FastAPI TestClient for API endpoint tests."""
    from fastapi.testclient import TestClient

    from data_engineering_copilot.api.app import app

    return TestClient(app)
