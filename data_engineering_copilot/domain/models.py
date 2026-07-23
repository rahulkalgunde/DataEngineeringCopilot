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
    timestamp: float = 0.0
    total_pages_fetched: int = 0
    total_chunks_indexed: int = 0
    elapsed_seconds: float = 0.0
    batch_size: int = 0
    current_phase: str = ""


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
    content_hash: str = ""
    extracted_entities: tuple[str, ...] = ()
    source_type: str = ""
    content_quality_score: float = 0.0


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    distance: float
    confidence: float


@dataclass(frozen=True)
class DocumentationSourceConfig:
    name: str
    start_url: str
    allowed_domains: tuple[str, ...]
    sitemap_url: str | None = None


@dataclass(frozen=True)
class RagConfig:
    retrieval_top_k: int = 5
    confidence_threshold: float = 0.3
    reranker_enabled: bool = False
    reranker_model: str = "ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 3
    max_context_chars: int = 4000


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[DocumentChunk, ...]
    confidence: float
    groundedness_score: float = 1.0
