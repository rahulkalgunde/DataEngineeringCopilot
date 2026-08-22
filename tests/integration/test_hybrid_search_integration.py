"""Integration tests for AsyncQdrantVectorStore hybrid search and filtering.

Exercises the real Qdrant testcontainer with upsert + query round-trips,
validating source filtering, chunk_type filtering, and combined filters.

Run with: pytest tests/integration/test_hybrid_search_integration.py -v -m integration
"""

from __future__ import annotations

import random

import pytest

from data_engineering_copilot.domain.models import DocumentChunk

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]


def _fake_embedding(dim: int = 2048) -> list[float]:
    return [random.random() for _ in range(dim)]


def _chunk(
    chunk_id: str,
    source_name: str,
    text: str,
    url: str = "https://example.com",
    title: str = "Test Page",
    chunk_type: str = "text",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name=source_name,
        title=title,
        url=url,
        text=text,
        content_hash=f"hash-{chunk_id}",
        section_header="## Section",
        chunk_type=chunk_type,
        word_count=len(text.split()),
        heading_path=("Section",),
    )


# ---------------------------------------------------------------------------
# Source filtering
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSourceFilter:
    @pytest.mark.asyncio
    async def test_source_filter_returns_only_matching_chunks(self, fresh_qdrant_store):
        store = fresh_qdrant_store
        dim = store._embedding_dim()

        chunks = [
            _chunk("s1-c1", "source_alpha", "Alpha doc"),
            _chunk("s1-c2", "source_beta", "Beta doc"),
            _chunk("s1-c3", "source_alpha", "Another alpha"),
        ]
        embeddings = [_fake_embedding(dim) for _ in chunks]
        await store.upsert_chunks(chunks, embeddings)

        results = await store.query(
            query_embedding=_fake_embedding(dim),
            top_k=10,
            source_filter=["source_alpha"],
        )
        assert len(results) == 2
        assert all(r.chunk.source_name == "source_alpha" for r in results)

    @pytest.mark.asyncio
    async def test_source_filter_multi(self, fresh_qdrant_store):
        store = fresh_qdrant_store
        dim = store._embedding_dim()

        chunks = [
            _chunk("s2-c1", "alpha", "doc a1"),
            _chunk("s2-c2", "beta", "doc b1"),
            _chunk("s2-c3", "gamma", "doc g1"),
        ]
        embeddings = [_fake_embedding(dim) for _ in chunks]
        await store.upsert_chunks(chunks, embeddings)

        results = await store.query(
            query_embedding=_fake_embedding(dim),
            top_k=10,
            source_filter=["alpha", "gamma"],
        )
        sources = {r.chunk.source_name for r in results}
        assert sources == {"alpha", "gamma"}

    @pytest.mark.asyncio
    async def test_no_filter_returns_all_sources(self, fresh_qdrant_store):
        store = fresh_qdrant_store
        dim = store._embedding_dim()

        chunks = [
            _chunk("s3-c1", "source_x", "doc x"),
            _chunk("s3-c2", "source_y", "doc y"),
        ]
        embeddings = [_fake_embedding(dim) for _ in chunks]
        await store.upsert_chunks(chunks, embeddings)

        results = await store.query(query_embedding=_fake_embedding(dim), top_k=10)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Chunk type filtering
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestChunkTypeFilter:
    @pytest.mark.asyncio
    async def test_chunk_type_filter_returns_only_matching(self, fresh_qdrant_store):
        store = fresh_qdrant_store
        dim = store._embedding_dim()

        chunks = [
            _chunk("ct-c1", "src", "def foo(): pass", chunk_type="code"),
            _chunk("ct-c2", "src", "Description text", chunk_type="text"),
            _chunk("ct-c3", "src", "GET /api/v1/items", chunk_type="api"),
        ]
        embeddings = [_fake_embedding(dim) for _ in chunks]
        await store.upsert_chunks(chunks, embeddings)

        results = await store.query(
            query_embedding=_fake_embedding(dim),
            top_k=10,
            chunk_type_filter="api",
        )
        assert len(results) == 1
        assert results[0].chunk.chunk_type == "api"


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCombinedFilters:
    @pytest.mark.asyncio
    async def test_source_and_chunk_type_combined(self, fresh_qdrant_store):
        store = fresh_qdrant_store
        dim = store._embedding_dim()

        chunks = [
            _chunk("cf-c1", "src_a", "code a", chunk_type="code"),
            _chunk("cf-c2", "src_a", "text a", chunk_type="text"),
            _chunk("cf-c3", "src_b", "code b", chunk_type="code"),
        ]
        embeddings = [_fake_embedding(dim) for _ in chunks]
        await store.upsert_chunks(chunks, embeddings)

        results = await store.query(
            query_embedding=_fake_embedding(dim),
            top_k=10,
            source_filter=["src_a"],
            chunk_type_filter="code",
        )
        assert len(results) == 1
        assert results[0].chunk.source_name == "src_a"
        assert results[0].chunk.chunk_type == "code"

    @pytest.mark.asyncio
    async def test_filter_returns_empty_when_no_match(self, fresh_qdrant_store):
        store = fresh_qdrant_store
        dim = store._embedding_dim()

        chunks = [_chunk("nf-c1", "src", "doc", chunk_type="text")]
        embeddings = [_fake_embedding(dim) for _ in chunks]
        await store.upsert_chunks(chunks, embeddings)

        results = await store.query(
            query_embedding=_fake_embedding(dim),
            top_k=10,
            source_filter=["nonexistent_source"],
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# count() and payload integrity
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPayloadIntegrity:
    @pytest.mark.asyncio
    async def test_count_matches_upserted_chunks(self, fresh_qdrant_store):
        store = fresh_qdrant_store
        dim = store._embedding_dim()

        chunks = [
            _chunk("pi-c1", "src", "hello world"),
            _chunk("pi-c2", "src", "foo bar"),
        ]
        await store.upsert_chunks(chunks, [_fake_embedding(dim), _fake_embedding(dim)])
        count = await store.count()
        assert count >= 2

    @pytest.mark.asyncio
    async def test_retrieved_chunk_contains_expected_fields(self, fresh_qdrant_store):
        store = fresh_qdrant_store
        dim = store._embedding_dim()

        chunk = _chunk("fi-c1", "src", "the quick brown fox", title="Fox Page")
        await store.upsert_chunks([chunk], [_fake_embedding(dim)])

        results = await store.query(
            query_embedding=_fake_embedding(dim),
            top_k=1,
            source_filter=["src"],
        )
        assert len(results) == 1
        retrieved = results[0].chunk
        assert retrieved.source_name == "src"
        assert retrieved.title == "Fox Page"
        assert "quick brown fox" in retrieved.text
        assert results[0].confidence >= 0.0
