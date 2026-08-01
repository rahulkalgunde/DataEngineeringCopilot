"""Tests for AsyncQdrantVectorStore — async wrapper around AsyncQdrantClient."""

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
            title="Title 1",
            url="http://example.com/1",
            text="content 1",
        ),
        DocumentChunk(
            chunk_id="chunk2",
            source_name="test_source",
            title="Title 2",
            url="http://example.com/2",
            text="content 2",
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [[0.1] * 768, [0.2] * 768]


async def test_init_creates_client_and_collection(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    await store.initialize()
    assert store._url == "http://localhost:6333"
    assert store._collection_name == "test"
    mock_async_qdrant.collection_exists.assert_awaited_once_with("test")
    mock_async_qdrant.create_collection.assert_awaited_once()


async def test_init_skips_creation_if_exists(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=True)
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="existing")
    await store.initialize()
    mock_async_qdrant.collection_exists.assert_awaited_once_with("existing")
    mock_async_qdrant.create_collection.assert_not_awaited()


async def test_upsert_chunks_success(mock_async_qdrant, sample_chunks, sample_embeddings):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    await store.initialize()
    await store.upsert_chunks(sample_chunks, sample_embeddings)
    mock_async_qdrant.upsert.assert_awaited_once()


async def test_query_success(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)

    mock_hit = MagicMock()
    mock_hit.id = "550e8400-e29b-41d4-a716-446655440000"
    mock_hit.score = 0.8
    mock_hit.payload = {
        "chunk_id": "chunk1",
        "source_name": "test_source",
        "title": "Test Title",
        "url": "http://example.com/1",
        "text": "Test content",
    }
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    results = await store.query([0.1] * 768, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "550e8400-e29b-41d4-a716-446655440000"
    assert results[0].confidence == pytest.approx(0.8)
    assert results[0].distance == pytest.approx(0.2)


async def _rrf_mock_response(mock_async_qdrant, score: float):
    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_hit = MagicMock()
    mock_hit.id = "550e8400-e29b-41d4-a716-446655440000"
    mock_hit.score = score
    mock_hit.payload = {
        "chunk_id": "chunk1",
        "source_name": "test_source",
        "title": "Test Title",
        "url": "http://example.com/1",
        "text": "Test content",
    }
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)


async def test_query_hybrid_rrf_normalizes_confidence(mock_async_qdrant):
    """RRF fused scores (1/(k+rank)) must be normalized to a 0..1 confidence."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    await _rrf_mock_response(mock_async_qdrant, score=0.0167)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        hybrid_rrf_k=60,
    )
    assert store._bm25 is not None
    store._bm25._frozen = True
    store._last_query_sparse = {"indices": [1], "values": [1.0]}

    results = await store.query([0.1] * 768, top_k=1, query_text="apache spark")

    assert len(results) == 1
    expected = 0.0167 * 61 / 2
    assert results[0].confidence == pytest.approx(expected)
    assert results[0].distance == pytest.approx(1.0 - expected)


async def test_query_hybrid_dense_fallback_keeps_raw_score(mock_async_qdrant):
    """Without a sparse vector the hybrid path falls back to raw cosine score."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    await _rrf_mock_response(mock_async_qdrant, score=0.8)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        hybrid_rrf_k=60,
    )
    assert store._bm25 is not None
    store._bm25._frozen = True

    results = await store.query([0.1] * 768, top_k=1)

    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.8)


async def test_query_empty_results(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_response = MagicMock()
    mock_response.points = []
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    results = await store.query([0.1] * 768, top_k=5)
    assert results == []


async def test_count_success(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_info = MagicMock()
    mock_info.points_count = 42
    mock_async_qdrant.get_collection = AsyncMock(return_value=mock_info)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    count = await store.count()
    assert count == 42


async def test_count_urls_filters_by_source(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_result = MagicMock()
    mock_result.count = 7
    mock_async_qdrant.count = AsyncMock(return_value=mock_result)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.count_urls("test_source")

    assert result == 7
    mock_async_qdrant.count.assert_awaited_once()
    call = mock_async_qdrant.count.await_args
    assert call is not None
    assert call.kwargs["collection_name"] == "test"
    assert call.kwargs["exact"] is True


async def test_count_urls_zero_for_missing_source(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_result = MagicMock()
    mock_result.count = 0
    mock_async_qdrant.count = AsyncMock(return_value=mock_result)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.count_urls("missing_source")
    assert result == 0


async def test_count_urls_client_none_returns_zero():
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    store._client = None
    result = await store.count_urls("test_source")
    assert result == 0


async def test_delete_by_url_success(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    await store.delete_by_url("http://example.com/page1")
    mock_async_qdrant.delete.assert_awaited_once()


async def test_get_content_hash_found(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_point = MagicMock()
    mock_point.payload = {"content_hash": "abc123"}
    mock_async_qdrant.scroll = AsyncMock(return_value=([mock_point], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.get_content_hash_for_url("http://example.com/page1")
    assert result == "abc123"


async def test_get_content_hash_not_found(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_async_qdrant.scroll = AsyncMock(return_value=([], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.get_content_hash_for_url("http://example.com/nonexistent")
    assert result is None


async def test_query_404_raises_vector_store_error(mock_async_qdrant):
    from data_engineering_copilot.domain.exceptions import VectorStoreError
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    exc = Exception("HTTP 404: collection 'test' not found")
    mock_async_qdrant.query_points = AsyncMock(side_effect=exc)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    with pytest.raises(VectorStoreError, match="not found"):
        await store.query([0.1] * 768, top_k=5)


async def test_query_non_404_exception_reraised(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_async_qdrant.query_points = AsyncMock(side_effect=RuntimeError("connection lost"))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    with pytest.raises(RuntimeError, match="connection lost"):
        await store.query([0.1] * 768, top_k=5)


async def test_scroll_urls_returns_distinct_urls(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)

    mock_point_1 = MagicMock()
    mock_point_1.payload = {"url": "http://example.com/1"}
    mock_point_2 = MagicMock()
    mock_point_2.payload = {"url": "http://example.com/2"}

    mock_async_qdrant.scroll = AsyncMock()
    mock_async_qdrant.scroll.side_effect = [
        ([mock_point_1, mock_point_2], None),
    ]

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    urls = await store.scroll_urls("test_source")
    assert sorted(urls) == ["http://example.com/1", "http://example.com/2"]


async def test_scroll_urls_empty_returns_empty_list(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_async_qdrant.scroll = AsyncMock(return_value=([], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    urls = await store.scroll_urls("missing_source")
    assert urls == []


async def test_scroll_urls_client_none_returns_empty():
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    store._client = None
    urls = await store.scroll_urls("test_source")
    assert urls == []


async def test_client_not_initialized_returns_safe_defaults():
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    with (
        patch(
            "data_engineering_copilot.infrastructure.async_qdrant_store.AsyncQdrantClient",
            side_effect=Exception("Connection refused"),
        ),
        pytest.raises(Exception, match="Connection refused"),
    ):
        AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
