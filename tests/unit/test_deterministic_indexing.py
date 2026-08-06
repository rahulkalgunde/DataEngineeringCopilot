"""Phase 7 tests: deterministic two-stage indexing (BM25 corpus + frozen upsert)."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.domain.models import DocumentChunk


@pytest.fixture
def mock_async_qdrant():
    with patch("data_engineering_copilot.infrastructure.async_qdrant_store.AsyncQdrantClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        yield mock_client


def _chunk(
    text: str,
    chunk_id: str,
    generation: str = "gen-1",
    source_commit: str = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
) -> DocumentChunk:
    from data_engineering_copilot.infrastructure.token_budget import count_tokens

    segment_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text=text,
        content_hash=segment_hash,
        doc_type="guide",
        language="conceptual",
        index_generation=generation,
        source_commit=source_commit,
        parent_content_hash=segment_hash,
        segment_index=0,
        segment_total=1,
        token_count=count_tokens(text),
        character_count=len(text),
    )


def test_fit_bm25_corpus_requires_hybrid(tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=False,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    with pytest.raises(ValueError, match="hybrid_search"):
        store.fit_bm25_corpus(["spark sql"])


def test_fit_bm25_corpus_empty_raises(tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    with pytest.raises(ValueError, match="non-empty"):
        store.fit_bm25_corpus([])


def test_fit_bm25_corpus_fits_and_persists(tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.fit_bm25_corpus(["Apache Spark SQL structured data", "Delta Lake ACID transactions"])
    assert store._bm25 is not None
    assert store._bm25._frozen is True
    assert store.is_hybrid_ready() is True
    assert (tmp_path / "bm25.json").exists()


async def test_upsert_frozen_chunks_length_mismatch(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.fit_bm25_corpus(["spark sql"])
    with pytest.raises(ValueError, match="equal lengths"):
        await store.upsert_frozen_chunks([_chunk("a", "c1")], [[0.1] * 768, [0.2] * 768])


async def test_upsert_frozen_chunks_requires_frozen(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    with pytest.raises(ValueError, match="frozen"):
        await store.upsert_frozen_chunks([_chunk("a", "c1")], [[0.1] * 768])


async def test_upsert_frozen_chunks_requires_generation(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.fit_bm25_corpus(["spark sql"])
    chunk = _chunk("a", "c1", generation="")
    with pytest.raises(ValueError, match="index_generation"):
        await store.upsert_frozen_chunks([chunk], [[0.1] * 768])


async def test_upsert_frozen_chunks_writes_sparse(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.fit_bm25_corpus(["spark sql window functions"])
    chunks = [_chunk("spark sql window functions dense_rank", "c1")]
    await store.upsert_frozen_chunks(chunks, [[0.1] * 768])
    call_kwargs = mock_async_qdrant.upsert.call_args.kwargs
    batch = call_kwargs.get("points") or call_kwargs[1].get("points")
    assert batch is not None
    assert "sparse" in batch.vectors


async def test_validate_index_generation_ok(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.fit_bm25_corpus(["spark sql"])

    collection_info = MagicMock()
    collection_info.points_count = 3
    config = MagicMock()
    config.params.sparse_vectors = {"sparse": {}}
    collection_info.config = config
    mock_async_qdrant.get_collection = AsyncMock(return_value=collection_info)

    report = await store.validate_index_generation(expected_points=3)
    assert report["point_count"] == 3
    assert report["sparse_configured"] is True
    assert report["bm25_ready"] is True
    assert report["passed"] is True


async def test_validate_index_generation_mismatch_raises(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.domain.exceptions import VectorStoreError
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.fit_bm25_corpus(["spark sql"])

    collection_info = MagicMock()
    collection_info.points_count = 5
    config = MagicMock()
    config.params.sparse_vectors = {"sparse": {}}
    collection_info.config = config
    mock_async_qdrant.get_collection = AsyncMock(return_value=collection_info)

    with pytest.raises(VectorStoreError, match="Point count mismatch"):
        await store.validate_index_generation(expected_points=3)


async def test_verify_payload_texts_ok(mock_async_qdrant, tmp_path) -> None:
    """Payload text matching chunks.jsonl proves persisted == embedded text."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )

    point = MagicMock()
    point.id = "uuid-1"
    point.payload = {"chunk_id": "c1", "text": "spark sql window functions"}
    mock_async_qdrant.scroll = AsyncMock(return_value=([point], None))

    mismatches = await store.verify_payload_texts({"c1": "spark sql window functions"})
    assert mismatches == []


async def test_verify_payload_texts_detects_truncation(mock_async_qdrant, tmp_path) -> None:
    """A truncated payload text is reported as a mismatch."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )

    point = MagicMock()
    point.id = "uuid-1"
    point.payload = {"chunk_id": "c1", "text": "spark sql window"}
    mock_async_qdrant.scroll = AsyncMock(return_value=([point], None))

    mismatches = await store.verify_payload_texts({"c1": "spark sql window functions"})
    assert len(mismatches) == 1
    assert "differs from persisted chunks.jsonl text" in mismatches[0]


async def test_verify_payload_texts_reports_unexpected_chunk(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )

    point = MagicMock()
    point.id = "uuid-1"
    point.payload = {"chunk_id": "c-unknown", "text": "x"}
    mock_async_qdrant.scroll = AsyncMock(return_value=([point], None))

    mismatches = await store.verify_payload_texts({"c1": "x"})
    assert len(mismatches) == 1
    assert "unexpected chunk_id" in mismatches[0]


# ------------------------------------------------------------------
# SparkIndexBuilder chunk normalization (content_hash + lossless segmentation)
# ------------------------------------------------------------------


def test_split_chunk_within_budget_is_single_segment(tmp_path) -> None:
    from data_engineering_copilot.config.settings import SparkSourceConfig
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    config = SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(),
    )
    builder = SparkIndexBuilder(
        config=config,
        resolver=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=None,  # type: ignore[arg-type]
        store=store,
        generation="gen-1",
        embedder=None,  # type: ignore[arg-type]
    )
    chunk = DocumentChunk(
        chunk_id="c1",
        source_name="Apache Spark 4.0.0",
        title="Window",
        url="http://x",
        text="window dense_rank content",
        content_hash="",
        doc_type="guide",
    )
    segments = builder._normalize_chunk(chunk)
    assert len(segments) == 1
    assert segments[0].text == chunk.text
    assert segments[0].segment_index == 0
    assert segments[0].segment_total == 1
    assert segments[0].parent_content_hash
    assert segments[0].index_generation == "gen-1"
    assert segments[0].chunker_version == "spark-chunker-v1"
    assert segments[0].content_hash


def test_normalize_chunk_splits_long_text_losslessly(tmp_path) -> None:
    from data_engineering_copilot.config.settings import SparkSourceConfig
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.infrastructure.token_budget import count_tokens
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    config = SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(),
    )
    builder = SparkIndexBuilder(
        config=config,
        resolver=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=None,  # type: ignore[arg-type]
        store=store,
        generation="gen-1",
        embedder=None,  # type: ignore[arg-type]
        max_embed_tokens=100,
        max_embed_chars=2000,
    )
    text = " ".join(["word"] * 5000)
    chunk = DocumentChunk(
        chunk_id="c2",
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text=text,
        content_hash="",
        doc_type="guide",
    )
    segments = builder._normalize_chunk(chunk)
    assert len(segments) > 1
    for segment in segments:
        assert count_tokens(segment.text) <= 100
        assert len(segment.text) <= 2000
        assert segment.parent_content_hash
        assert segment.segment_total == len(segments)
    assert [s.segment_index for s in segments] == list(range(len(segments)))
    assert "".join(s.text for s in segments).strip() == text.strip()
    # Segment IDs are derived from the parent chunk ID deterministically.
    assert segments[0].chunk_id.startswith("c2:seg:")


def test_normalize_chunk_computes_missing_content_hash(tmp_path) -> None:
    from data_engineering_copilot.config.settings import SparkSourceConfig
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    config = SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(),
    )
    builder = SparkIndexBuilder(
        config=config,
        resolver=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=None,  # type: ignore[arg-type]
        store=store,
        generation="gen-1",
        embedder=None,  # type: ignore[arg-type]
    )
    # Guide chunks from HeaderAwareChunker have empty content_hash.
    chunk = DocumentChunk(
        chunk_id="c1",
        source_name="Apache Spark 4.0.0",
        title="Window",
        url="http://x",
        text="window dense_rank content",
        content_hash="",
        doc_type="guide",
    )
    normalized = builder._normalize_chunk(chunk)
    assert len(normalized) == 1
    segment = normalized[0]
    assert segment.content_hash
    assert segment.index_generation == "gen-1"
    assert segment.chunker_version == "spark-chunker-v1"


def test_normalize_chunk_truncates_long_text(tmp_path) -> None:
    from data_engineering_copilot.config.settings import SparkSourceConfig
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    config = SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(),
    )
    builder = SparkIndexBuilder(
        config=config,
        resolver=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=None,  # type: ignore[arg-type]
        store=store,
        generation="gen-1",
        embedder=None,  # type: ignore[arg-type]
        max_embed_tokens=100,
        max_embed_chars=2000,
    )
    chunk = DocumentChunk(
        chunk_id="c2",
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text=" ".join(["word"] * 5000),
        content_hash="",
        doc_type="guide",
    )
    normalized = builder._normalize_chunk(chunk)
    assert len(normalized) > 1
    for segment in normalized:
        assert len(segment.text.split()) <= 100
        assert segment.segment_total == len(normalized)


# ------------------------------------------------------------------
# Full build: BM25 + embeddings receive exactly the persisted segments
# ------------------------------------------------------------------


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[0.1, 0.2, 0.3]] * len(texts)

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def close(self) -> None:
        return None


def test_build_embeds_and_fits_exactly_the_persisted_segments(tmp_path) -> None:
    """BM25 corpus and embedder receive exactly the texts that get upserted."""
    import asyncio

    from data_engineering_copilot.config.settings import SparkSourceConfig, SparkStreamConfig
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
    from data_engineering_copilot.infrastructure.spark_source_resolver import (
        SparkFileRecord,
        SparkManifest,
    )
    from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
    from data_engineering_copilot.services.spark_chunker import SparkChunker
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder

    # A guide long enough to require lossless splitting.
    guide = tmp_path / "big_guide.md"
    guide.write_text(
        "# Big Guide\n\n"
        + "\n\n".join(f"Paragraph {i} with a decent number of words to grow the size." for i in range(300)),
        encoding="utf-8",
    )
    commit = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"
    manifest = SparkManifest(
        source_name="Apache Spark 4.0.0",
        ref="v4.0.0",
        commit=commit,
        root=tmp_path,
        files=(
            SparkFileRecord(
                stream="guides",
                relative_path="big_guide.md",
                absolute_path=guide,
                doc_type="guide",
                language="conceptual",
                source_url=f"https://raw.githubusercontent.com/apache/spark/{commit}/big_guide.md",
            ),
        ),
        manifest_hash="fixture-hash",
    )
    config = SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit=commit,
        license="Apache-2.0",
        streams=(SparkStreamConfig("guides", "guide", ("**/*.md",), (), "conceptual", "header_aware"),),
    )

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.initialize = AsyncMock(return_value=None)  # type: ignore[method-assign]
    store.fit_bm25_corpus = MagicMock()
    store.upsert_frozen_chunks = AsyncMock(return_value=None)
    store.validate_index_generation = AsyncMock(return_value={"passed": True})
    store._bm25 = MagicMock()
    store._bm25.vocab_size = 42

    embedder = _RecordingEmbedder()
    builder = SparkIndexBuilder(
        config=config,
        resolver=None,  # type: ignore[arg-type]
        parser=NativeDocumentParser(),
        chunker=SparkChunker(header_chunker=HeaderAwareChunker()),
        store=store,
        generation="gen-1",
        embedder=embedder,
        max_embed_tokens=3800,
        max_embed_chars=6000,
    )

    report = asyncio.run(builder.build_from_manifest(manifest))

    persisted_texts = [c.text for c in store.upsert_frozen_chunks.call_args.args[0]]
    assert store.upsert_frozen_chunks.call_args.args[0]
    # BM25 corpus == embedded texts == persisted segment texts.
    bm25_texts = store.fit_bm25_corpus.call_args.args[0]
    embedded_texts = [t for batch in embedder.batches for t in batch]
    assert bm25_texts == embedded_texts
    assert embedded_texts == persisted_texts
    assert report.chunk_count == len(persisted_texts)
    for text in persisted_texts:
        assert len(text) <= 6000


# ------------------------------------------------------------------
# FallbackEmbedder adapter
# ------------------------------------------------------------------


def test_fallback_embedder_delegates_to_single_embedder() -> None:
    from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder

    class _Stub:
        async def embed_texts(self, texts):
            return [[0.1] * 4 for _ in texts]

        async def close(self):
            return None

    embedder = FallbackEmbedder(_Stub())

    async def _run():
        result = await embedder.embed_texts(["a", "b"])
        q = await embedder.embed_query("a")
        return result, q

    import asyncio

    result, q = asyncio.run(_run())
    assert len(result) == 2
    assert len(q) == 4


def test_fallback_embedder_delegates_to_chain_execute() -> None:
    from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder

    class _Chain:
        async def execute(self, request):
            return [[0.5] * 4 for _ in request]

        async def close(self):
            return None

    embedder = FallbackEmbedder(_Chain())

    async def _run():
        return await embedder.embed_texts(["a", "b"])

    import asyncio

    result = asyncio.run(_run())
    assert len(result) == 2
    assert result[0] == [0.5] * 4


# ------------------------------------------------------------------
# Task 7: per-file coverage records + strict generation validation
# ------------------------------------------------------------------

from data_engineering_copilot.services.spark_index_builder import CoverageRecord  # noqa: E402
from tests.unit.test_spark_hybrid_merge import (  # noqa: E402
    _COMMIT,
    _builder,
    _config,
    _native_manifest,
    _rendered_config,
    _rendered_manifest,
)


def _coverage_record(
    relative_path: str = "docs/x.md",
    representation: str = "native",
    doc_type: str = "guide",
    canonical_url: str = "https://spark.apache.org/docs/4.0.0/x.html",
    status: str = "indexed",
    chunk_count: int = 1,
    failure_reason: str = "",
) -> CoverageRecord:
    return CoverageRecord(
        relative_path=relative_path,
        representation=representation,
        doc_type=doc_type,
        canonical_url=canonical_url,
        status=status,
        chunk_count=chunk_count,
        content_hash="h",
        failure_reason=failure_reason,
    )


def test_validate_generation_artifacts_ok() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    chunks = [_chunk("text one", "a"), _chunk("text two", "b")]
    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=chunks,
        coverage=[_coverage_record(chunk_count=2)],
        native_manifest_paths=["docs/x.md"],
        rendered_manifest_paths=["x.html"],
        qdrant_point_count=2,
        bm25_ready=True,
        sparse_configured=True,
    )
    assert failures == []


def test_validate_generation_artifacts_fails_on_zero_chunks(tmp_path) -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=[],
        coverage=[_coverage_record(status="zero_chunks", failure_reason="selected non-empty file produced no chunks")],
        native_manifest_paths=["docs/x.md"],
        qdrant_point_count=0,
        bm25_ready=True,
        sparse_configured=True,
    )
    assert any("produced no chunks" in failure for failure in failures)


def test_validate_generation_artifacts_fails_on_missing_output() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=[_chunk("text", "a")],
        coverage=[_coverage_record(status="missing_output", representation="rendered", failure_reason="missing")],
        native_manifest_paths=[],
        rendered_manifest_paths=["missing.html"],
        qdrant_point_count=1,
        bm25_ready=True,
        sparse_configured=True,
    )
    assert any("rendered output missing" in failure for failure in failures)


def test_validate_generation_artifacts_fails_on_duplicate_manifest_path() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=[_chunk("text", "a")],
        coverage=[_coverage_record()],
        native_manifest_paths=["docs/x.md", "docs/x.md"],
        qdrant_point_count=1,
        bm25_ready=True,
        sparse_configured=True,
    )
    assert any("manifest path duplicated" in failure for failure in failures)


def test_validate_generation_artifacts_fails_on_duplicate_chunk_id() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=[_chunk("text", "a"), _chunk("text two", "a")],
        coverage=[_coverage_record(chunk_count=2)],
        native_manifest_paths=["docs/x.md"],
        qdrant_point_count=2,
        bm25_ready=True,
        sparse_configured=True,
    )
    assert any("duplicate chunk_id" in failure for failure in failures)


def test_validate_generation_artifacts_fails_on_point_count_mismatch() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=[_chunk("text", "a")],
        coverage=[_coverage_record()],
        native_manifest_paths=["docs/x.md"],
        qdrant_point_count=5,
        bm25_ready=True,
        sparse_configured=True,
    )
    assert any("differs from chunks.jsonl count" in failure for failure in failures)


def test_validate_generation_artifacts_fails_on_wrong_generation_or_commit() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    chunks = [_chunk("text", "a", generation="other-gen")]
    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=chunks,
        coverage=[_coverage_record()],
        native_manifest_paths=["docs/x.md"],
        qdrant_point_count=1,
        bm25_ready=True,
        sparse_configured=True,
    )
    assert any("generation 'other-gen', expected 'gen-1'" in failure for failure in failures)


def test_validate_generation_artifacts_fails_without_bm25_or_sparse() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=[_chunk("text", "a")],
        coverage=[_coverage_record()],
        native_manifest_paths=["docs/x.md"],
        qdrant_point_count=1,
        bm25_ready=False,
        sparse_configured=False,
    )
    assert any("BM25" in failure for failure in failures)
    assert any("Sparse" in failure for failure in failures)


def test_build_coverage_records_native_and_rendered(tmp_path) -> None:
    """Coverage marks rendered-replaced native as 'replaced', others indexed."""
    import asyncio

    manifest, _ = _native_manifest(tmp_path)
    rendered = _rendered_manifest(tmp_path)
    builder = _builder(tmp_path, rendered)

    chunks, coverage = asyncio.run(builder._chunk_all(manifest))

    assert chunks
    statuses = {record.status for record in coverage}
    assert "replaced" in statuses  # window.md superseded by window.html
    assert "indexed" in statuses  # example file indexed natively
    # Every native + rendered selected record produced chunks or was replaced.
    for record in coverage:
        assert record.status in ("indexed", "replaced", "no_content", "zero_chunks", "missing_output")
        assert record.representation in ("native", "rendered")
    # The nav-only reference/index.html is dropped from the merged corpus but
    # still appears as a coverage record for the rendered manifest, classified
    # as no_content (navigation-only), not a zero-chunk failure.
    nav = [r for r in coverage if r.canonical_url.endswith("/reference/index.html")]
    assert nav and nav[0].status == "no_content"
    assert "navigation-only" in nav[0].failure_reason


def test_is_redirect_stub_classifies_native_stubs(tmp_path) -> None:
    """Redirect / under-construction stubs are no_content, not zero_chunks."""
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
    from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
    from data_engineering_copilot.services.spark_chunker import SparkChunker
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder
    from tests.unit.test_spark_hybrid_merge import _config

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    builder = SparkIndexBuilder(
        config=_config(),
        resolver=None,  # type: ignore[arg-type]
        parser=NativeDocumentParser(),
        chunker=SparkChunker(header_chunker=HeaderAwareChunker()),
        store=store,
        generation="gen-1",
        embedder=None,  # type: ignore[arg-type]
    )

    assert builder._is_redirect_stub("---\nlayout: global\nredirect: rdd.html\n---\n\nThis document has moved here.")
    assert builder._is_redirect_stub("**This page is under construction**")
    assert not builder._is_redirect_stub("Real guide content with headings and substantive paragraphs about Spark.")


def test_build_writes_coverage_and_build_report(tmp_path) -> None:
    import asyncio

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
    from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
    from data_engineering_copilot.services.spark_chunker import SparkChunker
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder

    manifest, _ = _native_manifest(tmp_path)
    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.initialize = AsyncMock(return_value=None)  # type: ignore[method-assign]
    store.fit_bm25_corpus = MagicMock()
    store.upsert_frozen_chunks = AsyncMock(return_value=None)
    store.validate_index_generation = AsyncMock(return_value={"passed": True, "point_count": 3})
    store._bm25 = MagicMock()
    store._bm25.vocab_size = 42

    builder = SparkIndexBuilder(
        config=_config(),
        resolver=None,  # type: ignore[arg-type]
        parser=NativeDocumentParser(),
        chunker=SparkChunker(header_chunker=HeaderAwareChunker()),
        store=store,
        generation="gen-1",
        embedder=_RecordingEmbedder(),
        rendered_config=_rendered_config(),
        rendered_manifest=None,
        chunks_path=tmp_path / "artifacts" / "chunks.jsonl",
    )

    report = asyncio.run(builder.build_from_manifest(manifest))
    assert report.chunk_count > 0
    assert report.coverage_count >= 3

    chunks_jsonl = tmp_path / "artifacts" / "chunks.jsonl"
    coverage_json = tmp_path / "artifacts" / "coverage.json"
    build_report_json = tmp_path / "artifacts" / "build_report.json"
    assert chunks_jsonl.is_file()
    assert coverage_json.is_file()
    assert build_report_json.is_file()

    import json

    coverage = json.loads(coverage_json.read_text(encoding="utf-8"))
    assert all(record["status"] in ("indexed", "no_content", "zero_chunks") for record in coverage)
    build_report = json.loads(build_report_json.read_text(encoding="utf-8"))
    assert build_report["generation"] == "gen-1"
    assert build_report["source_commit"] == _COMMIT
    assert build_report["final_chunk_count"] == report.chunk_count
    assert build_report["bm25_vocabulary_size"] == 42
    assert build_report["validation_result"] is True


# ------------------------------------------------------------------
# Task 10: token-budget validation in generation validation
# ------------------------------------------------------------------


def _segment_chunk(text: str, chunk_id: str, parent_hash: str, index: int, total: int) -> DocumentChunk:
    from data_engineering_copilot.infrastructure.token_budget import count_tokens

    return DocumentChunk(
        chunk_id=chunk_id,
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        doc_type="guide",
        language="conceptual",
        index_generation="gen-1",
        source_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        parent_content_hash=parent_hash,
        segment_index=index,
        segment_total=total,
        token_count=count_tokens(text),
        character_count=len(text),
    )


def test_segment_validation_ok_for_multi_segment_generation(tmp_path) -> None:
    """A losslessly-split parent (via _normalize_chunk) passes Task 10 checks."""
    from data_engineering_copilot.config.settings import SparkSourceConfig
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder, validate_generation_artifacts

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    config = SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(),
    )
    builder = SparkIndexBuilder(
        config=config,
        resolver=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=None,  # type: ignore[arg-type]
        store=store,
        generation="gen-1",
        embedder=None,  # type: ignore[arg-type]
        max_embed_tokens=100,
        max_embed_chars=2000,
    )
    text = " ".join(["word"] * 5000)
    chunk = DocumentChunk(
        chunk_id="c2",
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text=text,
        content_hash="",
        doc_type="guide",
        source_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        index_generation="gen-1",
    )
    segments = builder._normalize_chunk(chunk)
    assert len(segments) > 1

    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=segments,
        coverage=[],
        native_manifest_paths=["docs/x.md"],
    )
    assert failures == []


def test_segment_validation_fails_on_over_token_budget() -> None:
    from data_engineering_copilot.infrastructure.token_budget import DEFAULT_MAX_TOKENS
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    text = "word " * DEFAULT_MAX_TOKENS * 2
    chunk = DocumentChunk(
        chunk_id="over",
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text=text,
        content_hash="",
        doc_type="guide",
        parent_content_hash="ph",
        segment_index=0,
        segment_total=1,
        token_count=DEFAULT_MAX_TOKENS + 1,
        character_count=len(text),
    )
    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=[chunk],
        coverage=[],
        native_manifest_paths=["docs/x.md"],
    )
    assert any("token_count" in failure and "exceeds budget" in failure for failure in failures)


def test_segment_validation_fails_on_over_character_budget() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    text = "a" * 7000
    chunk = DocumentChunk(
        chunk_id="over-char",
        source_name="Apache Spark 4.0.0",
        title="T",
        url="http://x",
        text=text,
        content_hash="",
        doc_type="guide",
        parent_content_hash="ph",
        segment_index=0,
        segment_total=1,
        token_count=10,
        character_count=7000,
    )
    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=[chunk],
        coverage=[],
        native_manifest_paths=["docs/x.md"],
    )
    assert any("character_count" in failure and "exceeds budget" in failure for failure in failures)


def test_segment_validation_fails_on_missing_segment_index() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    parent_hash = hashlib.sha256(b"parent source text").hexdigest()
    segments = [
        _segment_chunk("segment zero", "s0", parent_hash, 0, 2),
        _segment_chunk("segment two", "s2", parent_hash, 2, 2),
    ]
    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=segments,
        coverage=[],
        native_manifest_paths=["docs/x.md"],
    )
    assert any("missing segment index 1" in failure for failure in failures)


def test_segment_validation_fails_on_inconsistent_segment_total() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    parent_hash = hashlib.sha256(b"parent source text").hexdigest()
    segments = [
        _segment_chunk("segment zero", "s0", parent_hash, 0, 3),
        _segment_chunk("segment one", "s1", parent_hash, 1, 3),
    ]
    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=segments,
        coverage=[],
        native_manifest_paths=["docs/x.md"],
    )
    assert any("segment_total" in failure for failure in failures)


def test_segment_validation_fails_on_truncated_reconstruction() -> None:
    """A parent whose joined segments do not reproduce the parent hash fails."""
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    # parent hash recorded from full text, but a segment's text was truncated.
    full_text = "the quick brown fox jumps over the lazy dog"
    parent_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    segments = [
        _segment_chunk("the quick brown fox", "t0", parent_hash, 0, 1),
    ]
    failures = validate_generation_artifacts(
        generation="gen-1",
        expected_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        chunks=segments,
        coverage=[],
        native_manifest_paths=["docs/x.md"],
    )
    assert any("truncation detected" in failure for failure in failures)
