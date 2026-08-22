"""MRL multistage retrieval helpers and store wiring (plan phase B)."""

from __future__ import annotations

import math

import pytest

from data_engineering_copilot.infrastructure.async_qdrant_store import _mrl_small_vector


class TestMrlSmallVector:
    def test_slices_prefix_and_renormalizes(self):
        # 4-dim vector; small dim = 2 -> first two entries renormalized.
        vec = [0.6, 0.8, 999.0, 999.0]
        out = _mrl_small_vector(vec, 2)
        assert len(out) == 2
        norm = math.sqrt(sum(x * x for x in out))
        assert abs(norm - 1.0) < 1e-6
        assert abs(out[0] - 0.6) < 1e-6 or True  # direction preserved
        # direction must match the prefix's direction
        raw = math.sqrt(0.36 + 0.64)
        assert abs(out[0] - 0.6 / raw) < 1e-9
        assert abs(out[1] - 0.8 / raw) < 1e-9

    def test_dim_must_not_exceed_source(self):
        with pytest.raises(ValueError, match="exceeds"):
            _mrl_small_vector([0.1, 0.2], 4)

    def test_zero_prefix_returns_zeros(self):
        assert _mrl_small_vector([0.0, 0.0, 0.5], 2) == [0.0, 0.0]


class _CapturingClient:
    """Records upsert payloads; collection_exists False so initialize() creates."""

    def __init__(self):
        self.upserts: list = []
        self.create_kwargs: dict | None = None

    async def collection_exists(self, name):
        return False

    async def create_collection(self, **kwargs):
        self.create_kwargs = kwargs

    async def upsert(self, collection_name, points, **kwargs):
        self.upserts.append(points)

    async def get_collection(self, name):
        class C:
            config = None

        return C()


def _make_store(**overrides):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="t",
        hybrid_search=True,
        embedding_dimension=768,
        mrl_multistage_enabled=True,
        mrl_small_dim=4,
        **overrides,
    )
    store._client = _CapturingClient()  # type: ignore[assignment]
    return store


def _chunk(text="hello world"):
    from data_engineering_copilot.domain.models import DocumentChunk

    return DocumentChunk(chunk_id="c1", source_name="s", title="t", url="u", text=text)


class TestUpsertWritesSmallVector:
    @pytest.mark.asyncio
    async def test_upsert_chunks_includes_dense_small(self):
        store = _make_store()
        await store.upsert_chunks([_chunk()], [[0.6, 0.8, 0.1, 0.2, 0.3, 0.4]])
        points = store._client.upserts[-1]  # type: ignore[attr-defined]
        vectors = points.vectors
        assert "dense" in vectors
        small = vectors["dense_small"]
        norm = math.sqrt(sum(x * x for x in small[0]))
        assert len(small[0]) == 4
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_disabled_store_has_no_dense_small(self):
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        store = AsyncQdrantVectorStore(
            url="http://localhost:6333",
            collection_name="t",
            hybrid_search=True,
            embedding_dimension=768,
        )
        store._client = _CapturingClient()  # type: ignore[assignment]
        await store.upsert_chunks([_chunk()], [[0.6, 0.8, 0.1, 0.2, 0.3, 0.4]])
        vectors = store._client.upserts[-1].vectors  # type: ignore[attr-defined]
        assert "dense_small" not in vectors
