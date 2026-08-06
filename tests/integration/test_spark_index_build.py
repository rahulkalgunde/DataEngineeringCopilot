"""Phase 7 integration test: build a real Spark index generation in Qdrant."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_engineering_copilot.config.settings import SparkSourceConfig, SparkStreamConfig
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
from data_engineering_copilot.infrastructure.spark_source_resolver import (
    SparkFileRecord,
    SparkManifest,
)
from data_engineering_copilot.services.spark_chunker import SparkChunker
from data_engineering_copilot.services.spark_index_builder import SparkIndexBuilder

pytestmark = pytest.mark.qdrant

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "spark_v4_fixture"
_COMMIT = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"


class _FakeEmbedder:
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        return [[float(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 10 + 1) / 10.0] * self.dim for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        import hashlib

        return [float(int(hashlib.sha256(text.encode()).hexdigest(), 16) % 10 + 1) / 10.0] * self.dim

    async def close(self) -> None:
        return None


def _build_manifest() -> SparkManifest:
    files = [
        SparkFileRecord(
            stream="guides",
            relative_path="docs/window.md",
            absolute_path=_FIXTURE / "docs" / "window.md",
            doc_type="guide",
            language="conceptual",
            source_url=f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/docs/window.md",
        ),
        SparkFileRecord(
            stream="api",
            relative_path="python/pyspark/sql/functions.py",
            absolute_path=_FIXTURE / "python" / "pyspark" / "sql" / "functions.py",
            doc_type="api_reference",
            language="python",
            source_url=f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/python/pyspark/sql/functions.py",
        ),
        SparkFileRecord(
            stream="examples",
            relative_path="examples/src/main/python/nested_arrays.py",
            absolute_path=_FIXTURE / "examples" / "src" / "main" / "python" / "nested_arrays.py",
            doc_type="code_example",
            language="python",
            source_url=f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/examples/src/main/python/nested_arrays.py",
        ),
    ]
    return SparkManifest(
        source_name="Apache Spark 4.0.0",
        ref="v4.0.0",
        commit=_COMMIT,
        root=_FIXTURE.resolve(),
        files=tuple(files),
        manifest_hash="fixture-hash",
    )


def _config() -> SparkSourceConfig:
    return SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit=_COMMIT,
        license="Apache-2.0",
        streams=(
            SparkStreamConfig("guides", "guide", ("docs/**/*.md",), (), "conceptual", "header_aware"),
            SparkStreamConfig("api", "api_reference", ("python/pyspark/**/*.py",), (), "python", "api"),
            SparkStreamConfig("examples", "code_example", ("examples/src/main/**/*.py",), (), "mixed", "code"),
        ),
    )


class _FakeHeaderChunker:
    """Minimal header-aware chunker that splits on Markdown H1/H2/H3."""

    import re as _re

    async def chunk(self, document, precomputed_embeddings=None):
        import hashlib

        from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser

        parser = NativeDocumentParser()
        native = parser.parse_markdown(document.text, document.file_path)
        chunks = []
        for idx, (header, body) in enumerate(native.sections):
            if not body.strip():
                continue
            from data_engineering_copilot.domain.models import DocumentChunk

            chunks.append(
                DocumentChunk(
                    chunk_id=f"fixture-guide-{idx}-{len(body)}",
                    source_name=document.source_name,
                    title=header or document.title,
                    url=document.url,
                    text=body,
                    content_hash=hashlib.sha256(body.encode()).hexdigest(),
                    chunk_type="text",
                    doc_type=document.doc_type,
                    language=document.language,
                    source_commit=document.source_commit,
                    file_path=document.file_path,
                )
            )
        return chunks

    def extract_sentences(self, text):
        return None


@pytest.mark.asyncio
async def test_build_generation_in_qdrant(qdrant_url, tmp_path):
    collection = f"itest_spark_{abs(hash(tmp_path)) % 10_000_000}"
    store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=collection, embedding_dimension=768)
    await store.initialize()

    try:
        parser = NativeDocumentParser()
        chunker = SparkChunker(header_chunker=_FakeHeaderChunker())
        builder = SparkIndexBuilder(
            config=_config(),
            resolver=None,  # type: ignore[arg-type]  # fixture manifest used directly
            parser=parser,
            chunker=chunker,
            store=store,
            generation="spark-4.0.0-fa33ea00-test",
            embedder=_FakeEmbedder(),
        )
        report = await builder.build_from_manifest(_build_manifest())

        assert report.chunk_count >= 3
        assert report.validation_passed is True
        assert store.is_hybrid_ready() is True
        assert await store.count() == report.chunk_count
    finally:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, prefer_grpc=False)
        client.delete_collection(collection_name=collection)
        client.close()
