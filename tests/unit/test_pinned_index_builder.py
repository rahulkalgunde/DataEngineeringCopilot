"""Phase 2 tests: generic pinned index builder."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.pinned_index_builder import PinnedIndexBuilder
from data_engineering_copilot.services.prepared_source import PreparedSource
from data_engineering_copilot.services.spark_index_builder import CoverageRecord
from tests.doubles.vector_store import InMemoryVectorStore


def _builder(
    store: InMemoryVectorStore, embedder: _StubEmbedder, generation: str, output_dir=None
) -> PinnedIndexBuilder:
    return PinnedIndexBuilder(
        cast(AsyncQdrantVectorStore, store),
        embedder,
        generation,
        output_dir=output_dir,
    )


def _chunk(source_name: str, commit: str, generation: str, text: str) -> DocumentChunk:
    parsed = ParsedDocument(
        source_name=source_name,
        title="Doc",
        url="https://example.com/doc",
        text=text,
        doc_type="guide",
        language="conceptual",
        source_commit=commit,
        file_path="docs/doc.md",
        license="Apache-2.0",
    )
    chunks = asyncio.run(HeaderAwareChunker().chunk(parsed))
    return replace(
        chunks[0],
        source_commit=commit,
        index_generation=generation,
    )


_LONG_BODY = "# Heading\n\n" + "word " * 60 + "\n"


def _long(prefix: str) -> str:
    return f"# {prefix}\n\n{prefix} content: " + "word " * 60 + "\n"


class _StubEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(i + 1), float(len(texts) - i)] for i in range(len(texts))]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0]

    async def close(self) -> None: ...


def _package(slug: str, name: str, commit: str, generation: str, texts: list[str]) -> PreparedSource:
    chunks = [_chunk(name, commit, generation, text) for text in texts]
    coverage = tuple(
        CoverageRecord(
            relative_path="docs/doc.md",
            representation="native",
            doc_type="guide",
            canonical_url="https://example.com/doc",
            status="indexed",
            chunk_count=1,
            content_hash="",
        )
        for _ in texts
    )
    return PreparedSource(
        slug=slug,
        source_name=name,
        generation=generation,
        commit=commit,
        chunks=tuple(chunks),
        coverage=coverage,
    )


def test_build_combines_and_embeds_packages() -> None:
    store = InMemoryVectorStore()
    embedder = _StubEmbedder()
    generation = "gen-airflow-delta-abc123"
    airflow = _package("airflow", "Apache Airflow Documentation", "a" * 40, generation, [_long("Airflow")])
    delta = _package("delta", "Delta Lake Documentation", "b" * 40, generation, [_long("Delta")])

    report = asyncio.run(_builder(store, embedder, generation, output_dir=None).build([airflow, delta]))

    assert report.generation == generation
    assert report.chunk_count == 2
    assert report.source_file_count == 2
    assert report.validation_passed is True
    assert store._bm25_fit_count == 1
    assert len(embedder.calls) == 1
    assert len(store._chunks) == 2
    for chunk in store._chunks.values():
        assert chunk.index_generation == generation
        assert chunk.source_commit in {"a" * 40, "b" * 40}


def test_build_rejects_mismatched_source_commit() -> None:
    store = InMemoryVectorStore()
    embedder = _StubEmbedder()
    generation = "gen-x"
    package = _package("airflow", "Apache Airflow Documentation", "a" * 40, generation, [_LONG_BODY])
    bad = replace(package, commit="b" * 40)

    with pytest.raises(ValueError, match="airflow"):
        asyncio.run(_builder(store, embedder, generation).build([bad]))


def test_build_rejects_wrong_generation() -> None:
    store = InMemoryVectorStore()
    embedder = _StubEmbedder()
    package = _package("delta", "Delta Lake Documentation", "b" * 40, "other-gen", [_LONG_BODY])

    with pytest.raises(ValueError, match="delta"):
        asyncio.run(_builder(store, embedder, "gen-x").build([package]))


def test_build_dedups_identical_content_across_sources() -> None:
    store = InMemoryVectorStore()
    embedder = _StubEmbedder()
    generation = "gen-x"
    airflow = _package("airflow", "Apache Airflow Documentation", "a" * 40, generation, [_LONG_BODY])
    delta = _package("delta", "Delta Lake Documentation", "b" * 40, generation, [_LONG_BODY])

    report = asyncio.run(_builder(store, embedder, generation).build([airflow, delta]))

    assert report.chunk_count == 1
    assert len(store._chunks) == 1


def test_build_writes_artifacts(tmp_path) -> None:
    store = InMemoryVectorStore()
    embedder = _StubEmbedder()
    generation = "gen-x"
    package = _package("airflow", "Apache Airflow Documentation", "a" * 40, generation, [_LONG_BODY])

    asyncio.run(_builder(store, embedder, generation, output_dir=tmp_path).build([package]))

    assert (tmp_path / "chunks.jsonl").is_file()
    assert (tmp_path / "coverage.json").is_file()
    assert (tmp_path / "build_report.json").is_file()
