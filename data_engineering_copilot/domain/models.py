from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CacheScope:
    """Identifies the isolation scope a cache entry belongs to.

    Two scopes differing in any field (tenant, role, source filter, embedding
    model, or collection) must never share cached answers. The fingerprint is
    embedded in every cache key so cross-tenant / cross-filter leakage is
    structurally impossible.
    """

    tenant_id: str = "default"
    role: str = "reader"
    source_filter: tuple[str, ...] = ()
    embedding_model: str = ""
    collection_name: str = ""


@dataclass(frozen=True)
class CachedAnswer:
    """Serializable envelope stored in the query cache.

    Unlike the bare answer string, this preserves sources, confidence, and
    groundedness so a cache hit reconstructs the full ``Answer`` instead of a
    fabricated ``confidence=1.0`` / empty ``sources``.
    """

    text: str
    sources: tuple[DocumentChunk, ...] = ()
    confidence: float = 1.0
    groundedness_score: float = 1.0
    cached_at: float = 0.0


@dataclass(frozen=True)
class RawDocument:
    source_name: str
    url: str
    html: str
    content_type: str = "text/html"


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
class DocumentSection:
    """A structured section extracted from a parsed document."""

    header: str
    level: int  # 1 for #, 2 for ##, etc.
    heading_path: tuple[str, ...]
    text: str
    code_blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedDocument:
    source_name: str
    title: str
    url: str
    text: str
    sections: tuple[DocumentSection, ...] = ()


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_name: str
    title: str
    url: str
    text: str
    content_hash: str = ""
    section_header: str = ""
    chunk_type: str = "text"  # one of: text, code, api, table
    word_count: int = 0
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    distance: float
    confidence: float


@dataclass(frozen=True)
class RagConfig:
    retrieval_top_k: int = 5
    confidence_threshold: float = 0.3
    reranker_enabled: bool = False
    reranker_model: str = "ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 3
    max_context_chars: int = 4000
    max_expansion_queries: int = 2


@dataclass(frozen=True)
class LLMUsage:
    """Unified token usage metadata returned by all LLM providers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    duration_ms: int = 0
    tokens_per_second: float = 0.0


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[DocumentChunk, ...]
    confidence: float
    groundedness_score: float = 1.0
    stage_times: dict[str, float] = field(default_factory=dict)
    trace_id: str | None = None


@dataclass(frozen=True)
class UserPermissions:
    """RBAC permissions associated with an API key."""

    api_key_prefix: str
    allowed_sources: tuple[str, ...] = ()
    role: str = "reader"  # reader | admin
