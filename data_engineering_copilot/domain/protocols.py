"""Protocol interfaces for dependency inversion.

These structural-typing protocols define the contracts between layers.
Any concrete class that implements the required methods satisfies the
protocol — no inheritance required.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from data_engineering_copilot.domain.models import (
    DocumentChunk,
    IngestionEvent,
    ParsedDocument,
    RawDocument,
    RetrievedChunk,
)


class CrawlerProtocol(Protocol):
    def crawl(
        self,
        source: Any,
        max_pages: int,
        on_event: Callable[[IngestionEvent], None] | None = ...,
    ) -> Iterable[RawDocument]: ...


class ParserProtocol(Protocol):
    def parse(self, raw: RawDocument) -> ParsedDocument: ...


class ChunkerProtocol(Protocol):
    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]: ...


class EmbedderProtocol(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    async def upsert_chunks(self, chunks: Iterable[DocumentChunk], vectors: Iterable[list[float]]) -> None: ...
    async def query(
        self, query_embedding: list[float], top_k: int, query_text: str | None = ...
    ) -> list[RetrievedChunk]: ...
    async def count(self) -> int: ...
    async def get_content_hash_for_url(self, url: str) -> str | None: ...
    async def delete_by_url(self, url: str) -> None: ...
    def fit_bm25(self, texts: list[str]) -> None: ...
    def set_query_sparse(self, sparse_vector) -> None: ...
    def clear_query_sparse(self) -> None: ...


class LLMClientProtocol(Protocol):
    async def generate(self, prompt: str) -> str: ...


class RerankerProtocol(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...
    def is_available(self) -> bool: ...


class TelemetryTracerProtocol(Protocol):
    def start_observation(
        self,
        name: str,
        input: Any = ...,
        as_type: str = ...,
        model: str | None = ...,
    ) -> Any: ...
    def flush(self) -> None: ...
