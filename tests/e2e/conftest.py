"""E2E test fixtures using real Docker services (Redis, Qdrant, Ollama, API).

Assumes Docker services are running (the standard docker-compose stack).
Creates isolated Qdrant collections per test session and cleans up on teardown.

Note: Docker Redis requires auth. Use password from docker-compose.yml.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import httpx
import pytest

REDIS_URL = "redis://:local_secure_password_123@localhost:6379/0"


@pytest.fixture(scope="session")
def e2e_settings():
    """AppSettings tuned for E2E testing with unique collection name."""
    import uuid

    from data_engineering_copilot.config.settings import AppSettings

    collection = f"e2e_test_{uuid.uuid4().hex[:8]}"
    return AppSettings(
        qdrant_url="http://localhost:6333",
        redis_url=REDIS_URL,
        collection_name=collection,
        ollama_base_url="http://localhost:11434",
        embedding_batch_size=32,
        retrieval_top_k=5,
        max_context_chars=2000,
        confidence_threshold=0.10,
        reranker_enabled=False,
        query_rewrite_enabled=False,
    )


@pytest.fixture(scope="session", autouse=True)
def _e2e_qdrant_cleanup(e2e_settings):
    """Delete the unique test collection after all E2E tests complete."""
    import urllib.request

    try:
        req = urllib.request.Request("http://localhost:6333/collections", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.status == 200
    except Exception:
        pytest.skip("Qdrant is not reachable")
        yield
        return

    yield
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
        client.delete_collection(collection_name=e2e_settings.collection_name)
        client.close()
    except Exception:
        pass


@pytest.fixture
def e2e_redis() -> AsyncGenerator:
    """Real async Redis client connected to Docker Redis."""
    import redis.asyncio as aioredis

    try:
        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        yield client
    except Exception:
        pytest.skip("Redis is not reachable")
        return
    asyncio.ensure_future(client.aclose())


@pytest.fixture(scope="session")
def e2e_embedder(e2e_settings):
    """Real Ollama embeddings provider (session-scoped)."""
    import urllib.request

    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.status == 200
    except Exception:
        pytest.skip("Ollama is not reachable")
        yield None
        return

    from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

    embedder = AsyncOllamaEmbeddings(model_name=e2e_settings.embedding_model_name)
    yield embedder
    asyncio.run(embedder.close())


@pytest.fixture(scope="session")
def e2e_llm(e2e_settings):
    """Real Ollama LLM client (session-scoped)."""
    import urllib.request

    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.status == 200
    except Exception:
        pytest.skip("Ollama is not reachable")
        yield None
        return

    from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient

    client = AsyncOllamaClient(
        base_url=e2e_settings.ollama_base_url,
        model=e2e_settings.ollama_model,
        timeout_seconds=e2e_settings.ollama_timeout_seconds,
        num_ctx=e2e_settings.ollama_num_ctx,
        num_predict=e2e_settings.ollama_num_predict,
    )
    yield client
    asyncio.run(client.close())


@pytest.fixture
async def e2e_api_client() -> AsyncGenerator:
    """Real httpx client hitting the live FastAPI at localhost:8000."""
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        yield client
