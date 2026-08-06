"""In-memory vector store double implementing VectorStoreProtocol.

Mimics the scoring semantics of AsyncQdrantVectorStore (cosine similarity →
``confidence`` clamped to 0..1, ``distance = 1 - confidence``) so RAG
pipeline-logic tests run fully offline with stable results.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.domain.protocols import VectorStoreProtocol


def _cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    if denom == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False)) / denom


class InMemoryVectorStore(VectorStoreProtocol):
    """Thread-of-record store for hermetic RAG tests.

    Only dense cosine retrieval is implemented; BM25 fitting is recorded and
    no-ops (mirroring a hybrid-disabled Qdrant collection).
    """

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._bm25_fit_count = 0
        self._closed = False

    async def upsert_chunks(self, chunks: Iterable[DocumentChunk], vectors: Iterable[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=False):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = list(vector)

    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
        query_text: str | None = None,
        source_filter: list[str] | None = None,
        chunk_type_filter: str | None = None,
        metadata_filters: object | None = None,
        fused_limit: int | None = None,
    ) -> list[RetrievedChunk]:
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk_id, vector in self._vectors.items():
            chunk = self._chunks[chunk_id]
            if source_filter and chunk.source_name not in source_filter:
                continue
            if chunk_type_filter and chunk.chunk_type != chunk_type_filter:
                continue
            if metadata_filters is not None and not _matches_metadata(chunk, metadata_filters):
                continue
            scored.append((_cosine(query_embedding, vector), chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        retrieved: list[RetrievedChunk] = []
        for score, chunk in scored[:top_k]:
            confidence = max(0.0, min(1.0, score))
            retrieved.append(RetrievedChunk(chunk=chunk, distance=1.0 - confidence, confidence=confidence))
        return retrieved

    async def count(self) -> int:
        return len(self._chunks)

    async def count_urls(self, source_name: str) -> int:
        return sum(1 for c in self._chunks.values() if c.source_name == source_name)

    async def get_content_hash_for_url(self, url: str) -> str | None:
        for chunk in self._chunks.values():
            if chunk.url == url and chunk.content_hash:
                return chunk.content_hash
        return None

    async def delete_by_url(self, url: str) -> None:
        for chunk_id, chunk in list(self._chunks.items()):
            if chunk.url == url:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk_id, None)

    def fit_bm25(self, texts: list[str]) -> None:
        self._bm25_fit_count += 1

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True


def _matches_metadata(chunk: DocumentChunk, filters: object) -> bool:
    """Apply structured metadata filters to an in-memory chunk.

    Mirrors AsyncQdrantVectorStore._build_query_filter semantics: each non-empty
    tuple is an OR match (MatchAny) on the corresponding chunk field, and all
    non-empty tuples must match (AND).
    """
    from data_engineering_copilot.domain.models import RetrievalFilters

    if not isinstance(filters, RetrievalFilters):
        return True
    if filters.source_names and chunk.source_name not in filters.source_names:
        return False
    if filters.doc_types and chunk.doc_type not in filters.doc_types:
        return False
    if filters.languages and chunk.language not in filters.languages:
        return False
    if filters.versions and chunk.spark_version not in filters.versions:
        return False
    if filters.modules and chunk.module not in filters.modules:
        return False
    return not (filters.chunk_types and chunk.chunk_type not in filters.chunk_types)
