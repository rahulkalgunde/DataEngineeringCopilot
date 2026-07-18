"""Protocol interfaces for dependency inversion.

These structural-typing protocols define the contracts between layers.
Any concrete class that implements the required methods satisfies the
protocol — no inheritance required.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from data_engineering_copilot.config.settings import DocumentationSource
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
        source: DocumentationSource,
        max_pages: int,
        on_event: Callable[[IngestionEvent], None] | None = ...,
    ) -> Iterable[RawDocument]: ...


class ParserProtocol(Protocol):
    def parse(self, raw: RawDocument) -> ParsedDocument: ...


class ChunkerProtocol(Protocol):
    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]: ...


class EmbedderProtocol(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    def upsert_chunks(self, chunks: Iterable[DocumentChunk], vectors: Iterable[list[float]]) -> None: ...
    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...
    def get_content_hash_for_url(self, url: str) -> str | None: ...
    def delete_by_url(self, url: str) -> None: ...


class LLMClientProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...
