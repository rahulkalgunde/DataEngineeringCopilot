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
    return [[0.1] * 2048, [0.2] * 2048]


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
    results = await store.query([0.1] * 2048, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk1"
    assert results[0].confidence == pytest.approx(0.8)
    assert results[0].distance == pytest.approx(0.2)


async def test_query_falls_back_to_point_id_when_payload_lacks_chunk_id(mock_async_qdrant):
    """Retrieved chunks must carry the payload's real ``chunk_id``; the Qdrant
    point UUID is only a fallback for legacy points without the field."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)

    mock_hit = MagicMock()
    mock_hit.id = "550e8400-e29b-41d4-a716-446655440000"
    mock_hit.score = 0.8
    mock_hit.payload = {"source_name": "test_source", "title": "T", "url": "http://example.com", "text": "c"}
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    results = await store.query([0.1] * 2048, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "550e8400-e29b-41d4-a716-446655440000"


async def test_query_deserializes_deployment_mode(mock_async_qdrant):
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
        "deployment_mode": "yarn",
    }
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    results = await store.query([0.1] * 2048, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.deployment_mode == "yarn"


async def test_query_deployment_mode_defaults_empty(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)

    mock_hit = MagicMock()
    mock_hit.id = "550e8400-e29b-41d4-a716-446655440000"
    mock_hit.score = 0.8
    mock_hit.payload = {"chunk_id": "chunk1", "text": "legacy point without mode"}
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    results = await store.query([0.1] * 2048, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.deployment_mode == ""


async def test_chunk_to_payload_round_trips_deployment_mode():
    from data_engineering_copilot.infrastructure.async_qdrant_store import chunk_to_payload

    chunk = DocumentChunk(
        chunk_id="chunk1",
        source_name="test_source",
        title="Title",
        url="http://example.com",
        text="content",
        deployment_mode="kubernetes",
        token_count=42,
        character_count=7,
        representation="native",
    )
    payload = chunk_to_payload(chunk)
    assert payload["deployment_mode"] == "kubernetes"
    assert payload["token_count"] == 42
    assert payload["character_count"] == 7
    assert payload["representation"] == "native"


async def test_chunk_to_payload_deployment_mode_defaults_empty():
    from data_engineering_copilot.infrastructure.async_qdrant_store import chunk_to_payload

    chunk = DocumentChunk(chunk_id="chunk1", source_name="s", title="t", url="u", text="c")
    payload = chunk_to_payload(chunk)
    assert payload["deployment_mode"] == ""
    assert payload["token_count"] == 0
    assert payload["character_count"] == 0
    assert payload["representation"] == ""


async def test_empty_source_filter_rejected(mock_async_qdrant):
    """An empty source_filter must raise, never silently mean 'all sources'."""
    from data_engineering_copilot.domain.exceptions import VectorStoreError
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")

    with pytest.raises(VectorStoreError):
        await store.query([0.1] * 2048, top_k=1, source_filter=[])

    mock_async_qdrant.query_points.assert_not_called()


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

    results = await store.query([0.1] * 2048, top_k=1, query_text="apache spark")

    assert len(results) == 1
    expected = 0.0167 * 61 / 2
    assert results[0].confidence == pytest.approx(expected)
    assert results[0].distance == pytest.approx(1.0 - expected)


async def test_query_equal_rrf_profile_has_no_weights(mock_async_qdrant):
    """Default profile keeps current RRF behavior: k unchanged, no weights."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    await _rrf_mock_response(mock_async_qdrant, score=0.1)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        hybrid_rrf_k=60,
    )
    assert store._bm25 is not None
    store._bm25._frozen = True
    store._last_query_sparse = {"indices": [1], "values": [1.0]}

    await store.query(
        [0.1] * 2048,
        top_k=5,
        query_text="apache spark",
        source_filter=["spark"],
        chunk_type_filter="text",
        fused_limit=40,
    )

    call_kwargs = mock_async_qdrant.query_points.await_args.kwargs
    query = call_kwargs["query"]
    assert query.rrf.k == 60
    assert query.rrf.weights is None
    assert call_kwargs["limit"] == 40
    assert len(call_kwargs["prefetch"]) == 2
    # Dense and sparse prefetches keep the same filters as before.
    assert call_kwargs["prefetch"][0].filter is not None
    assert call_kwargs["prefetch"][1].filter is not None
    assert call_kwargs["prefetch"][0].using == "dense"
    assert call_kwargs["prefetch"][1].using == "sparse"


async def test_query_identifier_sparse_rrf_changes_only_weights(mock_async_qdrant):
    """Weighted profile: candidate limits, filters, RRF k, and query vectors
    are unchanged; only the RRF weights differ (sparse boosted to 1.25)."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import (
        RRF_IDENTIFIER_SPARSE_PROFILE,
        AsyncQdrantVectorStore,
    )

    await _rrf_mock_response(mock_async_qdrant, score=0.1)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        hybrid_rrf_k=60,
    )
    assert store._bm25 is not None
    store._bm25._frozen = True
    store._last_query_sparse = {"indices": [1], "values": [1.0]}

    await store.query(
        [0.1] * 2048,
        top_k=5,
        query_text="apache spark",
        source_filter=["spark"],
        chunk_type_filter="text",
        fused_limit=40,
        rrf_profile=RRF_IDENTIFIER_SPARSE_PROFILE,
    )

    call_kwargs = mock_async_qdrant.query_points.await_args.kwargs
    query = call_kwargs["query"]
    assert query.rrf.k == 60
    assert query.rrf.weights == [1.0, 1.25]
    # Identical request shape to the equal profile except for weights.
    assert call_kwargs["limit"] == 40
    assert len(call_kwargs["prefetch"]) == 2
    assert call_kwargs["prefetch"][0].using == "dense"
    assert call_kwargs["prefetch"][1].using == "sparse"
    assert call_kwargs["prefetch"][0].query == [0.1] * 2048
    assert call_kwargs["prefetch"][0].filter is not None
    assert call_kwargs["prefetch"][1].filter is not None


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

    results = await store.query([0.1] * 2048, top_k=1)

    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.8)


async def test_query_empty_results(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_response = MagicMock()
    mock_response.points = []
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    results = await store.query([0.1] * 2048, top_k=5)
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


async def test_delete_by_url_scoped_to_source(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    await store.delete_by_url("http://example.com/page1", "Source A")

    filter_arg = mock_async_qdrant.delete.await_args.kwargs["points_selector"].filter
    keys = [c.key for c in filter_arg.must]
    assert keys == ["url", "source_name"]
    match_values = [c.match.value for c in filter_arg.must]
    assert match_values == ["http://example.com/page1", "Source A"]


async def test_delete_by_url_unsourced_keeps_url_only_filter(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    await store.delete_by_url("http://example.com/page1")

    filter_arg = mock_async_qdrant.delete.await_args.kwargs["points_selector"].filter
    keys = [c.key for c in filter_arg.must]
    assert keys == ["url"]


async def test_get_content_hash_found(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_point = MagicMock()
    mock_point.payload = {"content_hash": "abc123"}
    mock_async_qdrant.scroll = AsyncMock(return_value=([mock_point], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.get_content_hash_for_url("http://example.com/page1")
    assert result == "abc123"


async def test_get_content_hash_scoped_to_source(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_point = MagicMock()
    mock_point.payload = {"content_hash": "abc123"}
    mock_async_qdrant.scroll = AsyncMock(return_value=([mock_point], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.get_content_hash_for_url("http://example.com/page1", "Source A")
    assert result == "abc123"

    call = mock_async_qdrant.scroll.await_args
    assert call is not None
    scroll_filter = call.kwargs["scroll_filter"]
    keys = [c.key for c in scroll_filter.must]
    assert keys == ["url", "source_name"]
    match_values = [c.match.value for c in scroll_filter.must]
    assert match_values == ["http://example.com/page1", "Source A"]


async def test_get_content_hash_not_found(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_async_qdrant.scroll = AsyncMock(return_value=([], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.get_content_hash_for_url("http://example.com/nonexistent")
    assert result is None


async def test_get_content_hash_skips_empty_payload(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    empty_point = MagicMock()
    empty_point.payload = {"content_hash": ""}
    good_point = MagicMock()
    good_point.payload = {"content_hash": "abc123"}
    mock_async_qdrant.scroll = AsyncMock(return_value=([empty_point, good_point], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.get_content_hash_for_url("http://example.com/page1")
    assert result == "abc123"


async def test_get_content_hash_returns_none_when_only_empty_payloads(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    empty_point = MagicMock()
    empty_point.payload = {"content_hash": ""}
    mock_async_qdrant.scroll = AsyncMock(return_value=([empty_point], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    result = await store.get_content_hash_for_url("http://example.com/page1")
    assert result is None


async def test_scroll_chunks_by_parent_hash_orders_by_segment_index(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    seg0 = MagicMock()
    seg0.id = "id-0"
    seg0.payload = {
        "chunk_id": "doc:seg:0",
        "source_name": "spark",
        "title": "T",
        "url": "u",
        "text": "first",
        "parent_content_hash": "P",
        "segment_index": 0,
        "segment_total": 2,
    }
    seg1 = MagicMock()
    seg1.id = "id-1"
    seg1.payload = {
        "chunk_id": "doc:seg:1",
        "source_name": "spark",
        "title": "T",
        "url": "u",
        "text": "second",
        "parent_content_hash": "P",
        "segment_index": 1,
        "segment_total": 2,
    }
    # Return out of order to prove ordering is by segment_index.
    mock_async_qdrant.scroll = AsyncMock(return_value=([seg1, seg0], None))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    siblings = await store.scroll_chunks_by_parent_hash("P", source_name="spark")

    assert [s.chunk_id for s in siblings] == ["doc:seg:0", "doc:seg:1"]
    assert "".join(s.text for s in siblings) == "firstsecond"
    # The scroll filter must include the parent hash and source_name.
    _, kwargs = mock_async_qdrant.scroll.call_args
    assert kwargs["scroll_filter"].must[0].key == "parent_content_hash"
    assert kwargs["scroll_filter"].must[1].key == "source_name"


async def test_scroll_chunks_by_parent_hash_fail_open_on_error(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_async_qdrant.scroll = AsyncMock(side_effect=RuntimeError("qdrant down"))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    assert await store.scroll_chunks_by_parent_hash("P") == []


async def test_query_404_raises_vector_store_error(mock_async_qdrant):
    from data_engineering_copilot.domain.exceptions import VectorStoreError
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    exc = Exception("HTTP 404: collection 'test' not found")
    mock_async_qdrant.query_points = AsyncMock(side_effect=exc)

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    with pytest.raises(VectorStoreError, match="not found"):
        await store.query([0.1] * 2048, top_k=5)


async def test_query_non_404_exception_reraised(mock_async_qdrant):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_async_qdrant.query_points = AsyncMock(side_effect=RuntimeError("connection lost"))

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    with pytest.raises(RuntimeError, match="connection lost"):
        await store.query([0.1] * 2048, top_k=5)


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


async def test_chunk_to_payload_round_trips_parent_chunk_id():
    from data_engineering_copilot.infrastructure.async_qdrant_store import chunk_to_payload

    chunk = DocumentChunk(
        chunk_id="child1",
        source_name="s",
        title="t",
        url="u",
        text="c",
        parent_chunk_id="parent1",
    )
    payload = chunk_to_payload(chunk)
    assert payload["parent_chunk_id"] == "parent1"

    no_parent = DocumentChunk(chunk_id="p", source_name="s", title="t", url="u", text="c")
    assert chunk_to_payload(no_parent)["parent_chunk_id"] == ""


async def test_query_substitutes_parent_context(mock_async_qdrant):
    """A retrieved child chunk must carry its parent's text when available."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    child_hit = MagicMock()
    child_hit.id = "child-uuid"
    child_hit.score = 0.8
    child_hit.payload = {
        "chunk_id": "doc:p0:c0",
        "source_name": "test_source",
        "title": "Title",
        "url": "http://example.com/1",
        "text": "child text",
        "parent_chunk_id": "doc:p0",
    }
    mock_response = MagicMock()
    mock_response.points = [child_hit]
    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)

    parent_hit = MagicMock()
    parent_hit.id = str(__import__("uuid").uuid5(__import__("uuid").NAMESPACE_DNS, "doc:p0"))
    parent_hit.payload = {"text": "PARENT CONTEXT TEXT"}

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    store._client.retrieve = AsyncMock(return_value=[parent_hit])

    results = await store.query([0.1] * 2048, top_k=1, query_text="apache spark")

    assert len(results) == 1
    assert results[0].chunk.text == "PARENT CONTEXT TEXT"
    assert results[0].chunk.parent_chunk_id == "doc:p0"


async def test_query_keeps_child_text_when_parent_missing(mock_async_qdrant):
    """If the parent point cannot be fetched, the child text is kept."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    child_hit = MagicMock()
    child_hit.id = "child-uuid"
    child_hit.score = 0.8
    child_hit.payload = {
        "chunk_id": "doc:p0:c0",
        "source_name": "test_source",
        "title": "Title",
        "url": "http://example.com/1",
        "text": "child text",
        "parent_chunk_id": "doc:p0",
    }
    mock_response = MagicMock()
    mock_response.points = [child_hit]
    mock_async_qdrant.collection_exists = AsyncMock(return_value=False)
    mock_async_qdrant.query_points = AsyncMock(return_value=mock_response)
    mock_async_qdrant.retrieve = AsyncMock(return_value=[])

    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test")
    results = await store.query([0.1] * 2048, top_k=1, query_text="apache spark")

    assert len(results) == 1
    assert results[0].chunk.text == "child text"


# ------------------------------------------------------------------
# BM25 namespace mode + tokenizer version enforcement (plan Task 7)
# ------------------------------------------------------------------


async def test_store_namespace_mode_creates_namespace_tokenizer(mock_async_qdrant, tmp_path):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="ns-test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
        bm25_namespace=True,
    )
    assert store._bm25 is not None
    assert store._bm25.namespace_enabled is True
    assert store._bm25.version == BM25Tokenizer.TOKENIZER_VERSION
    assert store._bm25_version_mismatch is False


async def test_store_loads_matching_namespace_cache(mock_async_qdrant, tmp_path):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

    tok = BM25Tokenizer(namespace=True)
    tok.fit(["spark.sql.functions"])
    cache = tmp_path / "bm25.json"
    tok.save(cache)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="ns-test",
        hybrid_search=True,
        bm25_persist_path=cache,
        bm25_namespace=True,
    )
    assert store._bm25 is not None
    assert store._bm25_loaded_from_disk is True
    assert store._bm25_version_mismatch is False


async def test_store_legacy_cache_with_namespace_mode_marks_mismatch(mock_async_qdrant, tmp_path):
    from data_engineering_copilot.infrastructure.async_qdrant_store import (
        AsyncQdrantVectorStore,
    )
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

    tok = BM25Tokenizer()
    tok.fit(["apache spark"])
    cache = tmp_path / "bm25.json"
    tok.save(cache)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="ns-test",
        hybrid_search=True,
        bm25_persist_path=cache,
        bm25_namespace=True,
    )
    assert store._bm25 is None
    assert store._bm25_version_mismatch is True
    assert store.is_hybrid_ready() is False


async def test_store_version_mismatch_fails_before_query(mock_async_qdrant, tmp_path):
    from data_engineering_copilot.infrastructure.async_qdrant_store import (
        AsyncQdrantVectorStore,
        VectorStoreError,
    )
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

    tok = BM25Tokenizer()
    tok.fit(["apache spark"])
    cache = tmp_path / "bm25.json"
    tok.save(cache)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="ns-test",
        hybrid_search=True,
        bm25_persist_path=cache,
        bm25_namespace=True,
    )
    with pytest.raises(VectorStoreError, match="version mismatch"):
        await store.query([0.1] * 2048, top_k=1, query_text="spark.sql.functions")
    mock_async_qdrant.query_points.assert_not_awaited()


async def test_store_version_mismatch_fails_before_upsert(mock_async_qdrant, tmp_path):
    from data_engineering_copilot.domain.models import DocumentChunk
    from data_engineering_copilot.infrastructure.async_qdrant_store import (
        AsyncQdrantVectorStore,
        VectorStoreError,
    )
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

    tok = BM25Tokenizer()
    tok.fit(["apache spark"])
    cache = tmp_path / "bm25.json"
    tok.save(cache)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="ns-test",
        hybrid_search=True,
        bm25_persist_path=cache,
        bm25_namespace=True,
    )
    chunk = DocumentChunk(
        chunk_id="c1",
        source_name="s",
        title="t",
        url="http://example.com/1",
        text="spark.sql.functions",
        content_hash="hash",
        index_generation="gen-1",
    )
    with pytest.raises(VectorStoreError, match="version mismatch"):
        await store.upsert_frozen_chunks([chunk], [[0.1] * 2048])
    mock_async_qdrant.upsert.assert_not_awaited()


async def test_store_unsupported_version_cache_marks_mismatch(mock_async_qdrant, tmp_path):
    import json

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    cache = tmp_path / "bm25.json"
    cache.write_text(
        json.dumps(
            {
                "tokenizer_version": "namespace-v2",
                "k1": 1.2,
                "b": 0.75,
                "vocab": {},
                "doc_freq": {},
                "corpus_size": 0,
                "avg_doc_len": 0.0,
                "frozen": True,
            }
        )
    )

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="ns-test",
        hybrid_search=True,
        bm25_persist_path=cache,
        bm25_namespace=True,
    )
    assert store._bm25_version_mismatch is True
    assert store.is_hybrid_ready() is False


async def test_store_namespace_mismatch_fails_before_fit(mock_async_qdrant, tmp_path):
    from data_engineering_copilot.infrastructure.async_qdrant_store import (
        AsyncQdrantVectorStore,
        VectorStoreError,
    )
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

    tok = BM25Tokenizer(namespace=True)
    tok.fit(["spark.sql.functions"])
    cache = tmp_path / "bm25.json"
    tok.save(cache)

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="ns-test",
        hybrid_search=True,
        bm25_persist_path=cache,
        bm25_namespace=False,
    )
    assert store._bm25_version_mismatch is True
    with pytest.raises(VectorStoreError, match="version mismatch"):
        store.fit_bm25(["apache spark"])
