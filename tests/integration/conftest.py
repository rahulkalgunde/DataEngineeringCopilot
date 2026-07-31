"""Integration test fixtures with testcontainers for Qdrant, Redis, and Ollama.

Provides:
- Session-scoped Qdrant container via testcontainers
- Session-scoped Redis container via testcontainers
- Session-scoped Ollama container via testcontainers
- worker_id-isolated collection names for xdist parallel execution
- Fallback to external Docker Compose if testcontainers unavailable
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib

import pytest

logger = logging.getLogger(__name__)

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
    """Start a Qdrant testcontainer. Returns the URL."""
    global _qdrant_container, _qdrant_url

    if _qdrant_url is not None:
        return _qdrant_url

    # Always use testcontainers for isolation — never connect to Docker Compose
    try:
        from testcontainers.qdrant import QdrantContainer

        _qdrant_container = QdrantContainer("qdrant/qdrant:v1.18.3")
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

    # Use Ollama dimension (768) for test isolation
    store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=collection, embedding_dimension=768)
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
# Redis testcontainer (session-scoped, shared across workers)
# ---------------------------------------------------------------------------

_redis_container = None
_redis_url = None


def _get_or_start_redis_container():
    """Start a Redis testcontainer. Returns the URL."""
    global _redis_container, _redis_url

    if _redis_url is not None:
        return _redis_url

    # Always use testcontainers for isolation — never connect to Docker Compose
    try:
        from testcontainers.redis import RedisContainer

        _redis_container = RedisContainer("redis:7-alpine")
        _redis_container.start()
        host = _redis_container.get_container_host_ip()
        port = _redis_container.get_exposed_port(6379)
        _redis_url = f"redis://{host}:{port}/0"
        return _redis_url
    except Exception:
        pass

    return None


@pytest.fixture(scope="session")
def redis_url():
    """Session-scoped Redis URL (from testcontainers or Docker Compose)."""
    url = _get_or_start_redis_container()
    if url is None:
        pytest.skip("Redis is not available (testcontainers failed and Docker Compose not running)")
    return url


@pytest.fixture
def fresh_redis_client(redis_url):
    """Isolated Redis client per test. Flushes test keys on teardown."""
    import redis as redis_lib

    client = redis_lib.from_url(redis_url, decode_responses=False, client_name="itest")
    yield client

    # Teardown: delete only keys created during this test
    try:
        for key in client.scan_iter("ingestion:*"):
            client.delete(key)
        for key in client.scan_iter("itest:*"):
            client.delete(key)
        for key in client.scan_iter("ratelimit:*"):
            client.delete(key)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# PostgreSQL testcontainer (session-scoped, shared across workers)
# ---------------------------------------------------------------------------

_pg_container = None
_pg_dsn = None


def _get_or_start_pg_container():
    """Start a PostgreSQL testcontainer. Returns the DSN."""
    global _pg_container, _pg_dsn

    if _pg_dsn is not None:
        return _pg_dsn

    try:
        from testcontainers.postgres import PostgresContainer

        _pg_container = PostgresContainer(
            "postgres:16-alpine",
            username="copilot",
            password="local_secure_password_123",
            dbname="crawl_frontier",
        )
        _pg_container.start()
        host = _pg_container.get_container_host_ip()
        port = _pg_container.get_exposed_port(5432)
        _pg_dsn = f"postgresql://copilot:local_secure_password_123@{host}:{port}/crawl_frontier"
        return _pg_dsn
    except Exception:
        pass

    return None


@pytest.fixture(scope="session")
def pg_dsn():
    """Session-scoped PostgreSQL DSN (from testcontainer)."""
    dsn = _get_or_start_pg_container()
    if dsn is None:
        pytest.skip("PostgreSQL testcontainer could not be started")
    return dsn


# ---------------------------------------------------------------------------
# Ollama testcontainer (session-scoped, shared across workers)
# ---------------------------------------------------------------------------

_OLLAMA_IMAGE = "ollama/ollama:0.32.4"
_OLLAMA_HOME = pathlib.Path.home() / ".ollama"
_OLLAMA_MODELS = ["nomic-embed-text", "llama3.2:3b", "qwen2.5-coder:7b"]

_ollama_container = None
_ollama_url: str | None = None


def _pull_model_with_retry(container, model: str, retries: int = 3) -> None:
    """Pull an Ollama model with retry and diagnostic logging."""
    import time

    for attempt in range(1, retries + 1):
        try:
            logger.info("Pulling Ollama model %s (attempt %d/%d)", model, attempt, retries)
            container.pull_model(model)
            logger.info("Successfully pulled Ollama model %s", model)
            return
        except Exception as exc:
            logger.warning(
                "Failed to pull Ollama model %s (attempt %d/%d): %s",
                model,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(2**attempt)
    logger.error("Exhausted retries pulling Ollama model %s", model)
    raise RuntimeError(f"Failed to pull Ollama model {model} after {retries} attempts")


def _get_or_start_ollama_container() -> str | None:
    """Start an Ollama testcontainer. Returns the URL."""
    global _ollama_container, _ollama_url

    if _ollama_url is not None:
        return _ollama_url

    try:
        from testcontainers.ollama import OllamaContainer

        _ollama_container = OllamaContainer(
            image=_OLLAMA_IMAGE,
            ollama_home=str(_OLLAMA_HOME),
        )
        _ollama_container.start()
        _ollama_url = _ollama_container.get_endpoint()
        logger.info("Ollama container started at %s", _ollama_url)

        # Pull models needed by tests
        existing = set()
        try:
            for m in _ollama_container.list_models():
                name = m["name"]
                existing.add(name)
                existing.add(name.split(":")[0])
        except Exception as exc:
            logger.warning("Failed to list existing Ollama models: %s", exc)

        for model in _OLLAMA_MODELS:
            if model not in existing:
                _pull_model_with_retry(_ollama_container, model)

        return _ollama_url
    except Exception as exc:
        logger.error("Failed to start Ollama container: %s", exc, exc_info=True)
        pass

    return None


@pytest.fixture(scope="session")
def ollama_url():
    """Session-scoped Ollama URL (from testcontainer)."""
    url = _get_or_start_ollama_container()
    if url is None:
        pytest.skip("Ollama testcontainer could not be started")
    return url


# ---------------------------------------------------------------------------
# Eager container start so shared conftest's collection-time check passes
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Start the Ollama container before test collection so the shared conftest's
    ``pytest_collection_modifyitems`` hook sees Ollama as available."""
    url = _get_or_start_ollama_container()
    if url:
        import tests.conftest as shared_conftest

        shared_conftest._ollama_ok = True


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_settings(ollama_url):
    from data_engineering_copilot.config.settings import AppSettings

    return AppSettings(
        ollama_base_url=ollama_url,
        embedding_provider="ollama",
        embedding_model_name="nomic-embed-text",
        llm_provider="ollama",
        code_llm_provider="ollama",
        code_llm_model="qwen2.5-coder:7b",
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
    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

    return AsyncOllamaEmbeddings(
        model_name=integration_settings.embedding_model_name,
        base_url=integration_settings.ollama_base_url,
    )


@pytest.fixture
def ollama_client(integration_settings):
    from data_engineering_copilot.infrastructure.llm_client import LLMClient

    return LLMClient(
        base_url=f"{integration_settings.ollama_base_url}/v1",
        model=integration_settings.ollama_model,
        timeout_seconds=integration_settings.ollama_timeout_seconds,
        extra_body={
            "options": {
                "num_ctx": integration_settings.ollama_num_ctx,
                "num_predict": integration_settings.ollama_num_predict,
            }
        },
    )
