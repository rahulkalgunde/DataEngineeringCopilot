"""Protocol interfaces for dependency inversion.

These structural-typing protocols define the contracts between layers.
Any concrete class that implements the required methods satisfies the
protocol — no inheritance required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any, Protocol

from data_engineering_copilot.domain.models import (
    DocumentChunk,
    IngestionEvent,
    LLMUsage,
    ParsedDocument,
    RawDocument,
    RetrievedChunk,
)


class CrawlerProtocol(Protocol):
    async def crawl(
        self,
        source: Any,
        max_pages: int,
        on_event: Callable[[IngestionEvent], None] | None = ...,
    ) -> AsyncIterator[RawDocument]: ...


class ParserProtocol(Protocol):
    def parse(self, raw: RawDocument) -> ParsedDocument: ...


class ChunkerProtocol(Protocol):
    async def chunk(
        self, document: ParsedDocument, precomputed_embeddings: list[list[float]] | None = None
    ) -> list[DocumentChunk]: ...


class EmbedderProtocol(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    async def upsert_chunks(self, chunks: Iterable[DocumentChunk], vectors: Iterable[list[float]]) -> None: ...
    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
        query_text: str | None = ...,
        source_filter: list[str] | None = ...,
        chunk_type_filter: str | None = ...,
    ) -> list[RetrievedChunk]: ...
    async def count(self) -> int: ...
    async def get_content_hash_for_url(self, url: str) -> str | None: ...
    async def delete_by_url(self, url: str) -> None: ...
    def fit_bm25(self, texts: list[str]) -> None: ...
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...


class SyncRedisProtocol(Protocol):
    """Minimal protocol for synchronous Redis clients used in async contexts."""

    def hget(self, key: str | bytes, field: str | bytes) -> bytes | None: ...
    def hset(self, key: str | bytes, field: str | bytes, value: str | bytes) -> None: ...
    def delete(self, *keys: str | bytes) -> int: ...
    def close(self) -> None: ...


class LLMClientProtocol(Protocol):
    async def generate(self, prompt: str) -> str: ...

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]: ...

    @property
    def last_usage(self) -> LLMUsage: ...


class RerankerProtocol(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...
    def is_available(self) -> bool: ...
    async def initialize(self) -> None: ...


class TelemetryTracerProtocol(Protocol):
    def start_observation(
        self,
        name: str,
        input: Any = ...,
        as_type: str = ...,
        model: str | None = ...,
    ) -> Any: ...
    def flush(self) -> None: ...
