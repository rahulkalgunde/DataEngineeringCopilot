"""Integration tests for QdrantVectorStore.

Tests CRUD operations, querying, upsert dedup, content hash, delete-by-URL,
count, and collection lifecycle against a real Qdrant instance.

Run with: pytest tests/test_qdrant_integration.py -v -m integration
"""

import uuid

import pytest

from data_engineering_copilot.domain.models import DocumentChunk
from tests.conftest import (
    unique_collection_name,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qdrant_url():
    from data_engineering_copilot.config.settings import settings

    return settings.qdrant_url


@pytest.fixture
def fresh_store(qdrant_url):
    """A brand-new QdrantVectorStore with a unique collection, torn down after test."""
    from qdrant_client import QdrantClient

    from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore

    coll = unique_collection_name("qtest")
    store = QdrantVectorStore(url=qdrant_url, collection_name=coll)
    yield store

    try:
        c = QdrantClient(url=qdrant_url, prefer_grpc=False)
        c.delete_collection(collection_name=coll)
        c.close()
    except Exception:
        pass


def _make_chunk(idx: int, url: str = "https://example.com/page") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"qtest:{uuid.uuid4().hex[:8]}:{idx:04d}",
        source_name="QdrantTest",
        title=f"Chunk {idx}",
        url=url,
        text=f"This is test chunk number {idx} with enough text to be meaningful.",
    )


def _random_embedding(dim: int = 768) -> list[float]:
    """Generate a pseudo-random but deterministic embedding for testing."""
    import hashlib

    h = hashlib.sha256(str(uuid.uuid4()).encode()).digest()
    # Repeat the 32-byte hash to fill 768 floats
    raw = (h * 25)[:dim]
    vec = [b / 255.0 for b in raw]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


# ---------------------------------------------------------------------------
# Collection creation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.qdrant
class TestQdrantCollectionLifecycle:
    def test_collection_is_created_on_init(self, fresh_store):
        from qdrant_client import QdrantClient

        from data_engineering_copilot.config.settings import settings

        client = QdrantClient(url=settings.qdrant_url, prefer_grpc=False)
        assert client.collection_exists(fresh_store._collection_name)
        client.close()

    def test_count_starts_at_zero(self, fresh_store):
        assert fresh_store.count() == 0


# ---------------------------------------------------------------------------
# Upsert + Query
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.qdrant
class TestQdrantUpsertAndQuery:
    def test_upsert_and_query_single(self, fresh_store):
        chunk = _make_chunk(0)
        emb = _random_embedding()
        fresh_store.upsert_chunks([chunk], [emb])

        results = fresh_store.query(emb, top_k=5)
        assert len(results) >= 1
        assert results[0].chunk.text == chunk.text

    def test_upsert_multiple_chunks(self, fresh_store):
        chunks = [_make_chunk(i) for i in range(5)]
        embs = [_random_embedding() for _ in range(5)]
        fresh_store.upsert_chunks(chunks, embs)

        assert fresh_store.count() == 5

    def test_upsert_dedup_by_chunk_id(self, fresh_store):
        """Upserting the same chunk_id twice should overwrite, not duplicate."""
        chunk = _make_chunk(0)
        emb1 = _random_embedding()
        fresh_store.upsert_chunks([chunk], [emb1])

        # Re-upsert same chunk_id with different text (simulating update)
        updated = DocumentChunk(
            chunk_id=chunk.chunk_id,
            source_name=chunk.source_name,
            title="Updated Title",
            url=chunk.url,
            text="Updated text content.",
        )
        emb2 = _random_embedding()
        fresh_store.upsert_chunks([updated], [emb2])

        assert fresh_store.count() == 1
        results = fresh_store.query(emb2, top_k=1)
        assert results[0].chunk.title == "Updated Title"

    def test_query_returns_empty_when_empty_collection(self, fresh_store):
        emb = _random_embedding()
        results = fresh_store.query(emb, top_k=10)
        assert results == []

    def test_query_top_k_respected(self, fresh_store):
        chunks = [_make_chunk(i) for i in range(10)]
        embs = [_random_embedding() for _ in range(10)]
        fresh_store.upsert_chunks(chunks, embs)

        results = fresh_store.query(_random_embedding(), top_k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.qdrant
class TestQdrantContentHash:
    def test_get_content_hash_returns_none_when_empty(self, fresh_store):
        result = fresh_store.get_content_hash_for_url("https://example.com/missing")
        assert result is None

    def test_get_content_hash_returns_stored_hash(self, fresh_store):
        from data_engineering_copilot.domain.models import DocumentChunk

        chunk = DocumentChunk(
            chunk_id=f"qtest:{uuid.uuid4().hex[:8]}:0001",
            source_name="QdrantTest",
            title="Hash Test",
            url="https://example.com/hash-test",
            text="Some text for hash testing.",
            content_hash="abc123def456",
        )
        emb = _random_embedding()
        fresh_store.upsert_chunks([chunk], [emb])

        result = fresh_store.get_content_hash_for_url("https://example.com/hash-test")
        assert result == "abc123def456"


# ---------------------------------------------------------------------------
# Delete by URL
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.qdrant
class TestQdrantDeleteByUrl:
    def test_delete_by_url_removes_matching_chunks(self, fresh_store):
        target_url = "https://example.com/to-delete"
        chunks = [
            _make_chunk(0, url=target_url),
            _make_chunk(1, url=target_url),
            _make_chunk(2, url="https://example.com/keep"),
        ]
        embs = [_random_embedding() for _ in range(3)]
        fresh_store.upsert_chunks(chunks, embs)
        assert fresh_store.count() == 3

        fresh_store.delete_by_url(target_url)
        # Wait briefly for Qdrant to process deletion
        import time

        time.sleep(0.5)

        count_after = fresh_store.count()
        assert count_after == 1

    def test_delete_by_url_noop_on_missing(self, fresh_store):
        chunk = _make_chunk(0)
        emb = _random_embedding()
        fresh_store.upsert_chunks([chunk], [emb])

        fresh_store.delete_by_url("https://nonexistent.com/missing")
        assert fresh_store.count() == 1


# ---------------------------------------------------------------------------
# Hybrid query (embeds + queries in one call)
# ---------------------------------------------------------------------------

# NOTE: hybrid_query tests are in test_rag_integration.py since they require
# Ollama for embedding and are not pure Qdrant tests.
