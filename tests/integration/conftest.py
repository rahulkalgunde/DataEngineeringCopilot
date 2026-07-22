"""Integration test fixtures with testcontainers for Qdrant.

Provides:
- Session-scoped Qdrant container via testcontainers
- worker_id-isolated collection names for xdist parallel execution
- Fallback to external Docker Compose if testcontainers unavailable
- Existing fixtures for Ollama/Langfuse (external services)
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request

import pytest

# ---------------------------------------------------------------------------
# xdist worker_id isolation
# ---------------------------------------------------------------------------


def _worker_suffix(worker_id: str) -> str:
    """Convert xdist worker_id ('gw0', 'gw1', 'master') to a safe suffix."""
    return worker_id.replace("gw", "w").replace("master", "m") or "default"


# ---------------------------------------------------------------------------
# Qdrant testcontainer (session-scoped, shared across workers)
# ---------------------------------------------------------------------------

_qdrant_container = None
_qdrant_url = None


def _get_or_start_qdrant_container():
    """Start a Qdrant container if not already running. Returns the URL."""
    global _qdrant_container, _qdrant_url

    if _qdrant_url is not None:
        return _qdrant_url

    # First: check if Qdrant is already running (Docker Compose / local dev)
    try:
        req = urllib.request.Request("http://localhost:6333/collections", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                _qdrant_url = "http://localhost:6333"
                return _qdrant_url
    except Exception:
        pass

    # Second: try testcontainers (self-contained, portable)
    try:
        from testcontainers.qdrant import QdrantContainer

        _qdrant_container = QdrantContainer("qdrant/qdrant:latest")
        _qdrant_container.start()
        host = _qdrant_container.get_container_host_ip()
        port = _qdrant_container.get_exposed_port(6333)
        _qdrant_url = f"http://{host}:{port}"
        return _qdrant_url
    except Exception:
        pass

    return None


@pytest.fixture(scope="session")
def qdrant_url():
    """Session-scoped Qdrant URL (from testcontainers or Docker Compose)."""
    url = _get_or_start_qdrant_container()
    if url is None:
        pytest.skip("Qdrant is not available (testcontainers failed and Docker Compose not running)")
    return url


@pytest.fixture
def fresh_qdrant_store(qdrant_url, worker_id):
    """Isolated AsyncQdrantVectorStore per xdist worker."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    suffix = _worker_suffix(worker_id)
    collection = f"itest_{suffix}_{os.getpid()}"

    store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=collection)
    asyncio.run(store.initialize())
    yield store

    # Cleanup: delete collection
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, prefer_grpc=False)
        client.delete_collection(collection_name=collection)
        client.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ollama health check (external service — skip if unreachable)
# ---------------------------------------------------------------------------

_ollama_ok: bool | None = None
_ollama_models_ok: bool | None = None


def _ollama_is_reachable(url: str = "http://localhost:11434", timeout: int = 3) -> bool:
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_has_models(url: str = "http://localhost:11434", timeout: int = 3) -> bool:
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return len(data.get("models", [])) > 0
    except Exception:
        return False


def ollama_available() -> bool:
    global _ollama_ok
    if _ollama_ok is None:
        _ollama_ok = _ollama_is_reachable()
    return _ollama_ok


def ollama_has_models() -> bool:
    global _ollama_models_ok
    if _ollama_models_ok is None:
        _ollama_models_ok = _ollama_has_models()
    return _ollama_models_ok


def require_ollama():
    if not ollama_available():
        pytest.skip("Ollama is not reachable — skipping test")
    if not ollama_has_models():
        pytest.skip("Ollama has no models pulled — skipping test")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_settings():
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


@pytest.fixture
def embeddings_provider(integration_settings):
    require_ollama()
    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

    return AsyncOllamaEmbeddings(model_name=integration_settings.embedding_model_name)


@pytest.fixture
def ollama_client(integration_settings):
    require_ollama()
    from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient

    return AsyncOllamaClient(
        base_url=integration_settings.ollama_base_url,
        model=integration_settings.ollama_model,
        timeout_seconds=integration_settings.ollama_timeout_seconds,
        num_ctx=integration_settings.ollama_num_ctx,
        num_predict=integration_settings.ollama_num_predict,
    )
