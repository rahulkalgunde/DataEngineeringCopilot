"""E2E test fixtures using isolated testcontainers for Qdrant, Redis, and Ollama.

Qdrant, Redis, and Ollama run in ephemeral testcontainers (never Docker Compose).
FastAPI is tested in-process via ASGITransport (no live server needed).
"""

from __future__ import annotations

import pathlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import httpx
import pytest

from tests.conftest import infra_unavailable

if TYPE_CHECKING:
    from data_engineering_copilot.config.settings import AppSettings

# ---------------------------------------------------------------------------
# Qdrant testcontainer
# ---------------------------------------------------------------------------

_qdrant_container = None
_qdrant_url: str | None = None


def _get_or_start_qdrant_container() -> str | None:
    """Start a Qdrant testcontainer. Returns the URL."""
    global _qdrant_container, _qdrant_url

    if _qdrant_url is not None:
        return _qdrant_url

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
def e2e_qdrant_url() -> str:
    """Session-scoped Qdrant URL from testcontainer."""
    url = _get_or_start_qdrant_container()
    if url is None:
        infra_unavailable("Qdrant testcontainer could not be started")
    return url


# ---------------------------------------------------------------------------
# Redis testcontainer
# ---------------------------------------------------------------------------

_redis_container = None
_redis_url: str | None = None


def _get_or_start_redis_container() -> str | None:
    """Start a Redis testcontainer. Returns the URL."""
    global _redis_container, _redis_url

    if _redis_url is not None:
        return _redis_url

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
def e2e_redis_url() -> str:
    """Session-scoped Redis URL from testcontainer."""
    url = _get_or_start_redis_container()
    if url is None:
        infra_unavailable("Redis testcontainer could not be started")
    return url


# ---------------------------------------------------------------------------
# Ollama testcontainer (session-scoped, shared across workers)
# ---------------------------------------------------------------------------

_OLLAMA_IMAGE = "ollama/ollama:0.32.4"
_OLLAMA_HOME = pathlib.Path.home() / ".ollama"
_OLLAMA_MODELS = ["llama3.2:3b"]

_ollama_container = None
_e2e_ollama_url: str | None = None


def _get_or_start_ollama_container() -> str | None:
    """Start an Ollama testcontainer. Returns the URL."""
    global _ollama_container, _e2e_ollama_url

    if _e2e_ollama_url is not None:
        return _e2e_ollama_url

    try:
        from testcontainers.ollama import OllamaContainer

        _ollama_container = OllamaContainer(
            image=_OLLAMA_IMAGE,
            ollama_home=str(_OLLAMA_HOME),
        )
        _ollama_container.start()
        _e2e_ollama_url = _ollama_container.get_endpoint()

        existing = set()
        for m in _ollama_container.list_models():
            name = m["name"]
            existing.add(name)
            existing.add(name.split(":")[0])
        for model in _OLLAMA_MODELS:
            if model not in existing:
                _ollama_container.pull_model(model)

        return _e2e_ollama_url
    except Exception:
        pass

    return None


@pytest.fixture(scope="session")
def e2e_ollama_url() -> str:
    """Session-scoped Ollama URL from testcontainer."""
    url = _get_or_start_ollama_container()
    if url is None:
        infra_unavailable("Ollama testcontainer could not be started")
    return url


# ---------------------------------------------------------------------------
# PostgreSQL testcontainer (crawl frontier)
# ---------------------------------------------------------------------------

_pg_container = None
_pg_dsn: str | None = None


def _get_or_start_pg_container() -> str | None:
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
def e2e_pg_dsn() -> str:
    """Session-scoped PostgreSQL DSN from testcontainer."""
    dsn = _get_or_start_pg_container()
    if dsn is None:
        infra_unavailable("PostgreSQL testcontainer could not be started")
    return dsn


# ---------------------------------------------------------------------------
# Settings (Ollama provider, testcontainer URLs)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def e2e_settings(
    e2e_qdrant_url: str,
    e2e_redis_url: str,
    e2e_ollama_url: str,
    e2e_pg_dsn: str,
) -> AppSettings:
    """AppSettings tuned for E2E testing with isolated testcontainer URLs."""
    import uuid

    from data_engineering_copilot.config.settings import AppSettings

    collection = f"e2e_test_{uuid.uuid4().hex[:8]}"

    return AppSettings(
        qdrant_url=e2e_qdrant_url,
        redis_url=e2e_redis_url,
        collection_name=collection,
        crawl_db_url=e2e_pg_dsn,
        ollama_local_base_url=e2e_ollama_url,
        llm_provider="ollama",
        embedding_provider="local-hf",
        code_llm_provider="ollama",
        code_llm_model="llama3.2:3b",
        local_hf_embedding_model="nvidia/Nemotron-3-Embed-1B-BF16",
        embedding_batch_size=32,
        retrieval_top_k=5,
        max_context_chars=2000,
        confidence_threshold=0.10,
        reranker_enabled=False,
        query_rewrite_enabled=False,
    )


# ---------------------------------------------------------------------------
# Qdrant cleanup (session-scoped, autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _e2e_qdrant_cleanup(e2e_settings: AppSettings):
    """Delete the unique test collection after all E2E tests complete."""
    yield
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=e2e_settings.qdrant_url, prefer_grpc=False)
        client.delete_collection(collection_name=e2e_settings.collection_name)
        client.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Redis (function-scoped async client)
# ---------------------------------------------------------------------------


@pytest.fixture
async def e2e_redis(e2e_redis_url: str) -> AsyncGenerator:
    """Real async Redis client connected to testcontainer Redis."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(e2e_redis_url, decode_responses=True)
    yield client
    await client.aclose()


# ---------------------------------------------------------------------------
# Ollama fixtures (session-scoped, use testcontainer)
# ---------------------------------------------------------------------------


@pytest.fixture
async def e2e_embedder(e2e_settings: AppSettings):
    """In-process local-hf embeddings (function-scoped)."""
    from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import (
        LocalSentenceTransformerEmbeddings,
    )

    embedder = LocalSentenceTransformerEmbeddings(
        model_name=e2e_settings.local_hf_embedding_model,
        embedding_dimension=e2e_settings.get_embedding_dimension(),
    )
    yield embedder
    await embedder.close()


@pytest.fixture
async def e2e_llm(e2e_settings: AppSettings):
    """Real Ollama LLM client from testcontainer (function-scoped)."""
    from data_engineering_copilot.infrastructure.llm_client import LLMClient

    client = LLMClient(
        base_url=f"{e2e_settings.ollama_local_base_url}/v1",
        model=e2e_settings.ollama_model,
        timeout_seconds=e2e_settings.ollama_timeout_seconds,
        max_tokens=e2e_settings.ollama_num_predict,
    )
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# API client (in-process ASGI transport, no live server needed)
# ---------------------------------------------------------------------------


@pytest.fixture
async def e2e_api_client() -> AsyncGenerator:
    """In-process FastAPI client using ASGITransport (no Docker Compose backend-api needed)."""
    from data_engineering_copilot.api.app import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client
