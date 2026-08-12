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
import os
import pathlib
import sys
import urllib.error
import urllib.request
import uuid
from typing import TYPE_CHECKING, NoReturn
from unittest.mock import patch

import dotenv
import pytest

if TYPE_CHECKING:
    from data_engineering_copilot.config.settings import AppSettings

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------
# Kill ambient environment pollution at the source instead of restoring
# ``os.environ`` after every test.  Third-party libraries (notably
# ``crawl4ai``) call ``load_dotenv()`` at import time, permanently injecting
# the developer's ``.env`` (e.g. ``LLM_PROVIDER=openrouter``) into
# ``os.environ``, which pydantic-settings reads *above* env files — this would
# defeat ``AppSettings(_env_file=None)`` isolation.  Patch it once for the
# whole test process so no import (ours or third-party) can pollute.
def _noop_load_dotenv(*args: object, **kwargs: object) -> bool:
    """No-op replacement for ``dotenv.load_dotenv`` during tests."""
    return False


dotenv.load_dotenv = _noop_load_dotenv


@pytest.fixture(autouse=True)
def _isolate_rate_limiter():
    """Isolate the rate limiter between tests.

    ``RateLimiter`` falls back to a module-global in-memory store when Redis
    is unavailable (rate_limiter.py ``_IN_MEMORY_STORE``), and would share a
    live Redis otherwise.  Without isolation the per-path bucket (60 req/60 s
    for ``/api/v1/ask``) is exhausted by earlier tests in the suite, causing
    later requests to spuriously return HTTP 429.

    Force the in-memory fallback and clear it before/after every test so the
    outcome is deterministic and independent of test order.  Tests that need
    the Redis path opt in by patching ``_redis_client`` themselves.
    """
    import data_engineering_copilot.services.rate_limiter as rate_limiter_mod

    rate_limiter_mod._IN_MEMORY_STORE.clear()
    with patch.object(rate_limiter_mod, "_redis_client", return_value=None):
        yield
    rate_limiter_mod._IN_MEMORY_STORE.clear()


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


def _redis_is_reachable(url: str = "redis://:local_secure_password_123@localhost:6379/0", timeout: int = 3) -> bool:
    """Return True if Redis responds to PING (authenticating if the URL has a password).

    The default matches the repo's compose Redis, which requires ``--requirepass``.
    """
    try:
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        password = parsed.password or ""

        sock = socket.create_connection((host, port), timeout=timeout)
        sock.sendall(b"PING\r\n")
        response = sock.recv(1024)
        if b"NOAUTH" in response and password:
            sock.sendall(f"AUTH {password}\r\n".encode())
            sock.recv(1024)
            sock.sendall(b"PING\r\n")
            response = sock.recv(1024)
        sock.close()
        return b"PONG" in response
    except Exception:
        return False


# Module-level caches so we only hit each service once per test session
_qdrant_ok: bool | None = None
_ollama_ok: bool | None = None
_langfuse_ok: bool | None = None
_redis_ok: bool | None = None


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


def redis_available() -> bool:
    global _redis_ok
    if _redis_ok is None:
        _redis_ok = _redis_is_reachable()
    return _redis_ok


_REQUIRE_INFRA = os.environ.get("REQUIRE_INFRA") == "1"


def infra_unavailable(reason: str) -> NoReturn:
    """Fail when REAL infra is required (``REQUIRE_INFRA=1``), else skip.

    Every infrastructure-availability guard in the test suite routes through
    here so that ``make test-real`` / CI never silently pass with all tests
    skipped because a service was down.
    """
    if _REQUIRE_INFRA:
        raise RuntimeError(f"REQUIRE_INFRA=1: required infra unavailable: {reason}")
    pytest.skip(reason)


def require_qdrant(url: str | None = None):
    target = url or "http://localhost:6333"
    if not _qdrant_is_reachable(target):
        infra_unavailable(f"Qdrant not reachable at {target}")


def require_ollama():
    if not ollama_available():
        infra_unavailable("Ollama not reachable")


def require_langfuse():
    if not langfuse_available():
        infra_unavailable("Langfuse not reachable")


def require_redis(url: str | None = None):
    target = url or "redis://:local_secure_password_123@localhost:6379/0"
    if not _redis_is_reachable(target):
        infra_unavailable(f"Redis not reachable at {target}")


def require_qdrant_and_ollama(qdrant_url: str | None = None):
    require_qdrant(qdrant_url)
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


def pytest_configure(config):
    """Enforce hermetic ``AppSettings`` construction during tests.

    ``AppSettings`` must never observe the developer's ``.env`` / ``.env.secrets``
    (real provider choices + API keys).  Three layers, in order of defense:
      1. ``dotenv.load_dotenv`` is no-op'd process-wide (see module top), so no
         third-party import can inject ``.env`` into ``os.environ``.
      2. ``_env_file=None`` is forced, so env *files* are never read.
      3. Ambient provider env vars in ``os.environ`` (shell exports) fail loudly
         instead of silently overriding settings.

    Provider choice defaults to Ollama: explicit non-Ollama providers raise
    unless the test deliberately opts in with ``_test_allow_non_ollama=True``
    and placeholder API keys (factory/wiring tests).  Use ``make_settings()``
    to build settings explicitly.
    """

    from data_engineering_copilot.config.settings import AppSettings

    _original_init = AppSettings.__init__

    _ALLOWED = frozenset({"ollama", ""})

    _PROVIDER_FIELDS = [
        "llm_provider",
        "embedding_provider",
        "code_llm_provider",
        "answer_llm_provider",
        "rewrite_llm_provider",
        "groundedness_llm_provider",
        "intent_llm_provider",
        "enrichment_llm_provider",
        "evaluation_llm_provider",
    ]

    _AMBIENT_PROVIDER_VARS = [
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "CODE_LLM_PROVIDER",
        "ANSWER_LLM_PROVIDER",
        "REWRITE_LLM_PROVIDER",
        "GROUNDEDNESS_LLM_PROVIDER",
        "INTENT_LLM_PROVIDER",
        "ENRICHMENT_LLM_PROVIDER",
        "EVALUATION_LLM_PROVIDER",
        "OPENROUTER_API_KEY",
        "NVIDIA_API_KEY",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "GEMINI_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ]

    def _patched_init(self, *args, **kwargs):
        # Tests must never read .env / .env.secrets / .env.local.  os.environ
        # vars (e.g. monkeypatch.setenv) still take precedence over defaults.
        kwargs.setdefault("_env_file", None)

        leak = sorted(v for v in _AMBIENT_PROVIDER_VARS if os.environ.get(v))
        if leak:
            raise RuntimeError(
                "Ambient provider environment variable(s) present during tests:\n  "
                + "\n  ".join(leak)
                + "\nTests must be hermetic — do not export these from .env or your shell."
            )

        # Several per-purpose providers default to 'groq'/'openrouter', so a
        # hermetic default construction needs placeholder keys (never real ones)
        # to pass key validation.  make_settings() is the explicit alternative.
        kwargs.setdefault("openrouter_api_key", "placeholder")
        kwargs.setdefault("groq_api_key", "placeholder")

        # Check explicit kwargs — catches test code that deliberately passes a
        # non-Ollama provider.  Tests that intentionally exercise provider
        # routing pass ``_test_allow_non_ollama=True`` (with placeholder keys).
        allow_non_ollama = kwargs.pop("_test_allow_non_ollama", False)
        if not allow_non_ollama:
            bad = [f for f in _PROVIDER_FIELDS if (val := kwargs.get(f, "")) and val.lower() not in _ALLOWED]
            if bad:
                raise RuntimeError(
                    "Test configuration uses non-Ollama LLM provider(s) in explicit kwargs:\n  "
                    + "\n  ".join(f"{f}={kwargs.get(f)!r}" for f in bad)
                    + "\nOnly 'ollama' is allowed in tests by default to avoid costly external API "
                    "calls. If you deliberately test provider routing, pass "
                    "_test_allow_non_ollama=True with placeholder API keys."
                )
        _original_init(self, *args, **kwargs)

    AppSettings.__init__ = _patched_init


# ---------------------------------------------------------------------------
# Settings factory — the single way to build AppSettings in tests
# ---------------------------------------------------------------------------


def make_settings(**overrides) -> "AppSettings":
    """Build hermetic ``AppSettings`` for tests.

    Never reads ``.env`` / ``.env.secrets`` / ``.env.local`` (forces
    ``_env_file=None``), defaults every provider to Ollama, and clears
    per-purpose overrides.  Provider API keys default to *empty* (providers
    unconfigured), so the fallback chain terminates at Ollama unless a test
    explicitly supplies a placeholder key for the provider it selects.

    The rare tests that deliberately exercise non-Ollama provider routing
    (factory/wiring tests) pass ``_test_allow_non_ollama=True`` alongside
    placeholder API keys — never real ones.
    """
    defaults = {
        "_env_file": None,
        "llm_provider": "ollama",
        "llm_model": "llama3.2:3b",
        "embedding_provider": "ollama",
        "embedding_model_name": "nomic-embed-text",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama3.2:3b",
        "answer_llm_provider": "",
        "rewrite_llm_provider": "",
        "groundedness_llm_provider": "",
        "intent_llm_provider": "",
        "enrichment_llm_provider": "",
        "evaluation_llm_provider": "",
        "code_llm_provider": "",
        "code_llm_model": "",
        "openrouter_api_key": "",
        "nvidia_api_key": "",
        "groq_api_key": "",
        "cerebras_api_key": "",
        "gemini_api_key": "",
        "api_key": "",
    }
    defaults.update(overrides)

    from data_engineering_copilot.config.settings import AppSettings

    settings = AppSettings(**defaults)
    settings.validate_all()
    return settings


def _needed_infra(markers: set[str]) -> set[str]:
    """Map pytest markers onto the infra services they require."""
    needed: set[str] = set()
    if "qdrant" in markers:
        needed.add("Qdrant")
    if "ollama" in markers:
        needed.add("Ollama")
    if "langfuse" in markers:
        needed.add("Langfuse")
    if "redis" in markers:
        needed.add("Redis")
    if "rag" in markers or "ingestion" in markers:
        needed.update(("Qdrant", "Ollama"))
    return needed


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration-marked tests when required services are unreachable.

    With ``REQUIRE_INFRA=1`` the run fails at collection instead of silently
    skipping — real infra is mandatory (``make test-real``, CI). Only the
    services actually needed by the collected tests are required (e.g. the e2e
    suite never needs Langfuse).
    """
    needed: set[str] = set()
    for item in items:
        needed |= _needed_infra({m.name for m in item.iter_markers()})

    if _REQUIRE_INFRA and needed:
        available = {
            "Qdrant": qdrant_available(),
            "Ollama": ollama_available(),
            "Langfuse": langfuse_available(),
            "Redis": redis_available(),
        }
        missing = [svc for svc in needed if not available[svc]]
        if missing:
            raise pytest.UsageError(
                "REQUIRE_INFRA=1 but required service(s) unavailable: "
                + ", ".join(missing)
                + ". Start them with 'make docker-up' (or 'make docker-setup') before running real-infra tests."
            )

    for item in items:
        markers = {m.name for m in item.iter_markers()}

        if "qdrant" in markers and not qdrant_available():
            item.add_marker(pytest.mark.skip(reason="Qdrant is not reachable"))

        if "ollama" in markers and not ollama_available():
            item.add_marker(pytest.mark.skip(reason="Ollama is not reachable"))

        if "langfuse" in markers and not langfuse_available():
            item.add_marker(pytest.mark.skip(reason="Langfuse is not reachable"))

        if "redis" in markers and not redis_available():
            item.add_marker(pytest.mark.skip(reason="Redis is not reachable"))

        if "rag" in markers and (not qdrant_available() or not ollama_available()):
            item.add_marker(pytest.mark.skip(reason="RAG tests require Qdrant + Ollama"))

        if "ingestion" in markers and (not qdrant_available() or not ollama_available()):
            item.add_marker(pytest.mark.skip(reason="Ingestion tests require Qdrant + Ollama"))


# ---------------------------------------------------------------------------
# Shared fixtures – Settings
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_settings():
    """AppSettings tuned for integration testing (hermetic, Ollama-only)."""
    return make_settings(
        embedding_model_name="nomic-embed-text",
        embedding_batch_size=32,
        retrieval_top_k=5,
        reranker_top_k=3,
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
    from data_engineering_copilot.infrastructure.llm_client import LLMClient

    return LLMClient(
        base_url=f"{integration_settings.ollama_base_url}/v1",
        model=integration_settings.ollama_model,
        timeout_seconds=integration_settings.ollama_timeout_seconds,
        max_tokens=integration_settings.ollama_num_predict,
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
        chunk_size_chars=integration_settings.chunk_size_words * 5,
        chunk_overlap_chars=integration_settings.chunk_overlap_words * 5,
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
    from data_engineering_copilot.domain.models import RagConfig
    from data_engineering_copilot.services.async_rag import AsyncRagService

    return AsyncRagService(
        config=RagConfig(),
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
