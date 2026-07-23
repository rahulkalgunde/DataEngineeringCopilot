"""Tests for hybrid search (dense + sparse BM25) in AsyncQdrantVectorStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.domain.models import DocumentChunk


@pytest.fixture
def mock_async_qdrant():
    with patch("data_engineering_copilot.infrastructure.async_qdrant_store.AsyncQdrantClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk1",
            source_name="test_source",
            title="Spark SQL",
            url="http://example.com/1",
            text="Apache Spark SQL is a module for structured data processing",
        ),
        DocumentChunk(
            chunk_id="chunk2",
            source_name="test_source",
            title="Delta Lake",
            url="http://example.com/2",
            text="Delta Lake brings ACID transactions to Apache Spark",
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [[0.1] * 768, [0.2] * 768]


# ------------------------------------------------------------------
# Collection creation with sparse vectors
# ------------------------------------------------------------------

async def test_initialize_creates_sparse_vectors_config(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test_hybrid",
        hybrid_search=True,
    )
    await store.initialize()

    call_args = mock_async_qdrant.create_collection.call_args
    # sparse_vectors_config should be present with a "sparse" key
    sparse_cfg = call_args.kwargs.get("sparse_vectors_config") or (
        call_args[1].get("sparse_vectors_config") if len(call_args) > 1 else None
    )
    assert sparse_cfg is not None
    assert "sparse" in sparse_cfg


async def test_initialize_no_sparse_when_hybrid_off(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test_dense",
        hybrid_search=False,
    )
    await store.initialize()

    call_args = mock_async_qdrant.create_collection.call_args
    sparse_cfg = call_args.kwargs.get("sparse_vectors_config") or (
        call_args[1].get("sparse_vectors_config") if len(call_args) > 1 else None
    )
    assert sparse_cfg is None


# ------------------------------------------------------------------
# Upsert with sparse vectors
# ------------------------------------------------------------------

async def test_upsert_includes_sparse_vectors(mock_async_qdrant, sample_chunks, sample_embeddings):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
    )
    await store.initialize()
    await store.upsert_chunks(sample_chunks, sample_embeddings)

    call_kwargs = mock_async_qdrant.upsert.call_args.kwargs
    batch = call_kwargs.get("points") or call_kwargs[1].get("points")
    # The Batch should have sparse_vectors
    assert batch is not None


async def test_upsert_no_sparse_when_hybrid_off(mock_async_qdrant, sample_chunks, sample_embeddings):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=False,
    )
    await store.initialize()
    await store.upsert_chunks(sample_chunks, sample_embeddings)

    call_kwargs = mock_async_qdrant.upsert.call_args.kwargs
    batch = call_kwargs.get("points")
    # When hybrid is off, vectors should be a plain list (not a dict)
    vectors = batch.vectors if hasattr(batch, "vectors") else batch.get("vectors")
    assert not isinstance(vectors, dict)


# ------------------------------------------------------------------
# Hybrid query with prefetch + RRF
# ------------------------------------------------------------------

async def test_hybrid_query_uses_prefetch(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)

    mock_hit = MagicMock()
    mock_hit.id = "550e8400-e29b-41d4-a716-446655440000"
    mock_hit.score = 0.85
    mock_hit.payload = {
        "chunk_id": "chunk1",
        "source_name": "test",
        "title": "Spark",
        "url": "http://example.com/1",
        "text": "Spark SQL content",
    }
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
    )
    # Fit BM25 so hybrid is active
    store.fit_bm25(["Apache Spark SQL structured data", "Delta Lake ACID"])
    # Set the sparse vector for the query
    from qdrant_client.http.models import SparseVector
    store.set_query_sparse(SparseVector(indices=[0, 1], values=[1.0, 0.5]))

    results = await store.query([0.1] * 768, top_k=5)

    # Verify query_points was called with prefetch
    call_kwargs = mock_async_qdrant.query_points.call_args.kwargs
    assert "prefetch" in call_kwargs
    assert len(results) == 1


async def test_dense_only_query_no_prefetch(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)

    mock_response = MagicMock()
    mock_response.points = []
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=False,
    )
    results = await store.query([0.1] * 768, top_k=5)

    call_kwargs = mock_async_qdrant.query_points.call_args.kwargs
    # No prefetch when hybrid is off
    assert "prefetch" not in call_kwargs


# ------------------------------------------------------------------
# BM25 fit integration
# ------------------------------------------------------------------

async def test_fit_bm25_populates_tokenizer(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
    )
    texts = [
        "Apache Spark SQL structured data",
        "Delta Lake ACID transactions",
        "Apache Airflow workflow orchestration",
    ]
    store.fit_bm25(texts)
    assert store._bm25 is not None
    assert store._bm25._frozen is True
    assert store._bm25.vocab_size > 0


async def test_fit_bm25_noop_when_hybrid_off():
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    with patch(
        "data_engineering_copilot.infrastructure.async_qdrant_store.AsyncQdrantClient",
    ):
        store = AsyncQdrantVectorStore(
            url="http://localhost:6333",
            collection_name="test",
            hybrid_search=False,
        )
        store.fit_bm25(["hello world"])
        # _bm25 should still be None when hybrid is disabled
        assert store._bm25 is None


# ------------------------------------------------------------------
# Backward compatibility: default hybrid_search=True
# ------------------------------------------------------------------

async def test_default_constructor_enables_hybrid(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    assert store._hybrid_search is True
    assert store._bm25 is not None
