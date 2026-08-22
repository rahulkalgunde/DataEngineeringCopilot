"""Phase 9 integration test: Spark retrieval recall against a seeded index."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, LLMUsage, RagConfig, RetrievedChunk
from data_engineering_copilot.domain.protocols import LLMClientProtocol, VectorStoreProtocol
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.context_assembler import ContextAssembler

pytestmark = pytest.mark.qdrant

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "spark_v4_fixture"
_COMMIT = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"

_GUIDE_URL = f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/docs/window.md"
_API_URL = f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/python/pyspark/sql/functions.py"
_EXAMPLE_URL = f"https://raw.githubusercontent.com/apache/spark/{_COMMIT}/examples/src/main/python/nested_arrays.py"


class _FakeLLM(LLMClientProtocol):
    """Deterministic LLM double that returns a fixed answer."""

    def __init__(self) -> None:
        self._usage = LLMUsage()

    async def generate(self, prompt: str) -> str:
        return "A Spark answer using dense_rank, filter, transform, aggregate."

    def generate_stream(self, prompt: str):
        async def _gen():
            yield "A Spark answer."

        return _gen()

    @property
    def last_usage(self) -> LLMUsage:
        return self._usage


class _StubEmbedder:
    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        return [[float(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 10 + 1) / 10.0] * self.dim for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        import hashlib

        return [float(int(hashlib.sha256(text.encode()).hexdigest(), 16) % 10 + 1) / 10.0] * self.dim

    async def close(self) -> None:
        return None


def _seed_chunks() -> list[DocumentChunk]:
    from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser

    parser = NativeDocumentParser()

    guide_text = parser.parse_markdown(
        (_FIXTURE / "docs" / "window.md").read_text(),
        "docs/window.md",
    ).text
    guide = DocumentChunk(
        chunk_id="guide-window-1",
        source_name="Apache Spark 4.0.0",
        title="Window Functions",
        url=_GUIDE_URL,
        text=guide_text,
        content_hash="g1",
        doc_type="guide",
        language="conceptual",
        spark_version="4.0.0",
        source_commit=_COMMIT,
        file_path="docs/window.md",
        license="Apache-2.0",
    )

    api_text = parser.parse_code(
        (_FIXTURE / "python" / "pyspark" / "sql" / "functions.py").read_text(),
        "python/pyspark/sql/functions.py",
        "python",
    ).text
    api = DocumentChunk(
        chunk_id="api-functions-1",
        source_name="Apache Spark 4.0.0",
        title="functions",
        url=_API_URL,
        text=api_text,
        content_hash="a1",
        chunk_type="api",
        doc_type="api_reference",
        language="python",
        spark_version="4.0.0",
        module="pyspark.sql.functions",
        source_commit=_COMMIT,
        file_path="python/pyspark/sql/functions.py",
        license="Apache-2.0",
    )

    example_text = parser.parse_code(
        (_FIXTURE / "examples" / "src" / "main" / "python" / "nested_arrays.py").read_text(),
        "examples/src/main/python/nested_arrays.py",
        "python",
    ).text
    example = DocumentChunk(
        chunk_id="example-nested-1",
        source_name="Apache Spark 4.0.0",
        title="nested_arrays",
        url=_EXAMPLE_URL,
        text=example_text,
        content_hash="e1",
        chunk_type="code",
        doc_type="code_example",
        language="python",
        spark_version="4.0.0",
        module="pyspark.sql.functions",
        source_commit=_COMMIT,
        file_path="examples/src/main/python/nested_arrays.py",
        license="Apache-2.0",
    )

    return [guide, api, example]


async def _seed_store(store: AsyncQdrantVectorStore, chunks: list[DocumentChunk]) -> None:
    embedder = _StubEmbedder()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    store.fit_bm25_corpus([c.text for c in chunks])
    tagged = [
        DocumentChunk(
            **{**c.__dict__, "index_generation": "spark-4.0.0-fa33ea00-test"},
        )
        for c in chunks
    ]
    await store.upsert_frozen_chunks(tagged, vectors)


@pytest.mark.asyncio
async def test_window_query_assembles_window_evidence(qdrant_url, tmp_path):
    collection = f"itest_rag_window_{abs(hash(tmp_path)) % 10_000_000}"
    store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=collection, embedding_dimension=2048)
    await store.initialize()
    try:
        await _seed_store(store, _seed_chunks())

        service = AsyncRagService(
            config=RagConfig(retrieval_top_k=10, reranker_enabled=False, max_context_chars=4000),
            vector_store=cast(VectorStoreProtocol, store),
            llm_client=_FakeLLM(),
            embedder=_StubEmbedder(),
        )
        answer = await service.answer(
            "Calculate the 7-day rolling total spend per customer using dense_rank in Spark SQL."
        )
        urls = {c.url for c in answer.sources}
        assert _GUIDE_URL in urls or _API_URL in urls

        assembler = ContextAssembler(max_context_chars=4000)
        context, _, _ = assembler.assemble(
            [RetrievedChunk(chunk=c, distance=0.1, confidence=0.5) for c in answer.sources],
            deduplicate=False,
        )
        assert "dense_rank" in context or "Window" in context
    finally:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, prefer_grpc=False)
        client.delete_collection(collection_name=collection)
        client.close()


@pytest.mark.asyncio
async def test_nested_array_query_retrieves_filter_transform_aggregate(qdrant_url, tmp_path):
    collection = f"itest_rag_array_{abs(hash(tmp_path)) % 10_000_000}"
    store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=collection, embedding_dimension=2048)
    await store.initialize()
    try:
        await _seed_store(store, _seed_chunks())

        service = AsyncRagService(
            config=RagConfig(retrieval_top_k=10, reranker_enabled=False, max_context_chars=4000),
            vector_store=cast(VectorStoreProtocol, store),
            llm_client=_FakeLLM(),
            embedder=_StubEmbedder(),
        )
        answer = await service.answer(
            "Filter an array of structs where discount > 0.20, then transform and aggregate to compute net_total."
        )
        urls = {c.url for c in answer.sources}
        assert _API_URL in urls or _EXAMPLE_URL in urls

        combined_text = "\n".join(c.text for c in answer.sources)
        assert "filter" in combined_text or "Filter" in combined_text
        assert "transform" in combined_text
        assert "aggregate" in combined_text
    finally:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, prefer_grpc=False)
        client.delete_collection(collection_name=collection)
        client.close()
