from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawDocument:
    source_name: str
    url: str
    html: str


@dataclass(frozen=True)
class IngestionEvent:
    event_type: str
    source_name: str
    message: str
    url: str | None = None
    title: str | None = None
    chunks_indexed: int = 0
    pages_fetched: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    source_name: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_name: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    distance: float
    confidence: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[DocumentChunk, ...]
    confidence: float
