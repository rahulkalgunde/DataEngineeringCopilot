from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from data_engineering_copilot.config.settings import settings


@dataclass(frozen=True)
class RetrievalFilters:
    """Structured metadata filters applied during retrieval.

    Empty tuples mean "no constraint" for that field. The default instance is
    the canonical "no filter" value.

    ``modules`` are hard filters (exact module match, e.g. an explicitly named
    ``pyspark.sql.functions.filter``). ``preferred_modules`` are soft ranking
    preferences derived from function terms (e.g. ``dense_rank``) — they are
    never applied as hard filters so cross-doc-type content (guides, examples)
    remains retrievable.
    """

    source_names: tuple[str, ...] = ()
    doc_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    preferred_modules: tuple[str, ...] = ()
    chunk_types: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.source_names
            or self.doc_types
            or self.languages
            or self.versions
            or self.modules
            or self.preferred_modules
            or self.chunk_types
        )


@dataclass(frozen=True)
class CacheScope:
    """Identifies the isolation scope a cache entry belongs to.

    Two scopes differing in any field (tenant, role, source filter, embedding
    model, collection, index generation, or answer-config fingerprint) must
    never share cached answers. The fingerprint is embedded in every cache key
    so cross-tenant / cross-filter / cross-generation / cross-config leakage is
    structurally impossible.
    """

    tenant_id: str = "default"
    role: str = "reader"
    source_filter: tuple[str, ...] = ()
    embedding_model: str = ""
    collection_name: str = ""
    index_generation: str = ""
    config_fingerprint: str = ""


@dataclass(frozen=True)
class CachedAnswer:
    """Serializable envelope stored in the query cache.

    Unlike the bare answer string, this preserves sources, confidence, and
    groundedness so a cache hit reconstructs the full ``Answer`` instead of a
    fabricated ``confidence=1.0`` / empty ``sources``. ``suggestions`` caches
    the ChatGPT-style follow-up questions so a cache hit is fully instant
    (answer + chips, no regeneration).
    """

    text: str
    sources: tuple[DocumentChunk, ...] = ()
    confidence: float = 1.0
    groundedness_score: float = 1.0
    cached_at: float = 0.0
    suggestions: tuple[str, ...] = ()


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
    doc_type: str = ""
    language: str = ""
    spark_version: str = ""
    module: str = ""
    source_commit: str = ""
    file_path: str = ""
    license: str = ""


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
    chunk_index: int = 0
    total_chunks: int = 0
    crawled_at: str = ""  # ISO 8601 UTC timestamp when the page was crawled
    doc_type: str = ""
    language: str = ""
    spark_version: str = ""
    module: str = ""
    source_commit: str = ""
    file_path: str = ""
    license: str = ""
    parser_version: str = ""
    chunker_version: str = ""
    index_generation: str = ""
    # Deployment mode the chunk documents (e.g. "yarn", "kubernetes",
    # "standalone"); empty for non-mode-specific content.
    deployment_mode: str = ""
    # Lossless token-budget segmentation metadata. Empty parent_content_hash
    # means the chunk was never split (it is its own segment).
    parent_content_hash: str = ""
    segment_index: int = -1
    segment_total: int = 1
    token_count: int = 0
    character_count: int = 0
    # Source representation for the hybrid corpus: "native" (raw repo file) or
    # "rendered" (locally built HTML). Empty for legacy/non-Spark chunks.
    representation: str = ""
    # Hierarchical parent-child link: chunk_id of the parent context chunk this
    # child was split from. Empty for parent chunks and for corpora built
    # without hierarchical chunking.
    parent_chunk_id: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    distance: float
    confidence: float


@dataclass(frozen=True)
class EmbeddingRequest:
    """A batch of texts plus the retrieval role for dual-mode embedding models.

    Some embedding models (e.g. ``nemotron-3-embed-1b``) encode passages and
    queries in different, non-interchangeable modes. The provider chain is
    mode-agnostic, so this request carries the role so clients can set
    ``input_type`` on the provider API. ``input_type`` is ``"passage"`` for
    index-time chunks, ``"query"`` for live search prompts, and ``None`` when
    the target model has no mode (client skips the field).
    """

    input_type: str | None
    texts: list[str]


@dataclass(frozen=True)
class RerankRequest:
    """A batch of documents to rerank against a query.

    Provider-agnostic request passed through the fallback chain. Clients
    translate to their native payload shapes (NVIDIA ``passages``, OpenRouter
    ``documents``, HF text-classification inputs).
    """

    query: str
    documents: list[str]
    top_n: int = 10


@dataclass(frozen=True)
class RerankResult:
    """Normalized rerank output: ``(index_into_documents, score)`` pairs.

    Scores are normalized to ``[0, 1]`` so the rerank confidence gate
    semantics hold across providers. ``index`` refers to the position in the
    original ``RerankRequest.documents`` list (sorted descending by score).
    """

    rankings: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class RagConfig:
    retrieval_top_k: int = 5
    confidence_threshold: float = 0.3
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 3
    # Cross-encoder scores sit on a different scale than embedding/fused
    # confidence (relevant pairs often land ~0.10-0.15). Gate on this value
    # when a reranker ran for the query; otherwise fall back to
    # ``confidence_threshold`` (the embedding scale).
    reranker_confidence_threshold: float = 0.10
    max_context_chars: int = 8000
    max_chunks_per_source: int = 2
    max_expansion_queries: int = 2
    cache_enabled: bool = True
    # Phase F: smart-cache recall tier. When enabled, follow-up chat turns try
    # to reuse similar cached (question→answer) pairs via local synthesis,
    # gated by scope verify, before falling through to the full pipeline.
    chat_cache_recall_enabled: bool = False
    chat_cache_top_k: int = 3
    chat_cache_recall_threshold: float = 0.70
    chat_cache_max_age_seconds: int = 86400
    # ChatGPT-style clickable follow-up suggestions.
    chat_suggestions_enabled: bool = True
    chat_suggestions_count: int = 3
    chat_suggestions_mode: str = "hybrid"


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
    stage_times: Mapping[str, float] = field(default_factory=dict)
    trace_id: str | None = None
    # --- Debug/visualizer artifacts (populated by AsyncRagService.answer) ---
    # Effective query actually used for retrieval after rewriting/expansion.
    rewritten_query: str | None = None
    # Query variants sent to retrieval: original + decomposed + expanded.
    query_variants: tuple[str, ...] = ()
    intent: str | None = None
    # Per-candidate retrieval details for the visualizer: one dict per ranked
    # result with rank, chunk_id, source_name, title, url, distance,
    # confidence, word_count and a short text snippet.
    retrieval_details: tuple[dict, ...] = ()
    # Rerank metadata: enabled, pool_size, top_k, final_top_k, dropped count.
    rerank_details: dict = field(default_factory=dict)
    # Assembled context string injected into the prompt (PII-safe subset).
    context: str | None = None
    # Final LLM prompt (may be large; used only by the visualizer).
    prompt: str | None = None
    # Generation token usage (from llm_client.last_usage) when available.
    token_usage: Mapping[str, object] = field(default_factory=dict)
    # Unsupported claims flagged by the groundedness verifier.
    groundedness_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserPermissions:
    """RBAC permissions associated with an API key."""

    api_key_prefix: str
    allowed_sources: tuple[str, ...] = ()
    role: str = "reader"  # reader | admin


class IngestRequest(BaseModel):
    """Shared ingestion request validated at both API and broker trust boundaries.

    ``max_pages`` is clamped to ``settings.max_pages_hard_cap`` so the contract
    value has a single source of truth across the API route, the Celery broker,
    and the CLI.
    """

    source_names: list[str] | None = Field(default=None, max_length=20)
    max_pages: int | None = Field(default=None, ge=1, le=settings.max_pages_hard_cap)


@dataclass(frozen=True)
class ChatMessage:
    """A single conversational turn within a chat session.

    ``sources`` holds JSON-safe provenance dicts (``_chunk_provenance_ref``
    shape) rather than full ``DocumentChunk`` objects so the message is
    directly serializable by both the Redis and Postgres stores.
    """

    message_id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = 0.0
    sources: tuple[dict, ...] = ()
    token_count: int = 0
    groundedness_score: float = 1.0
    groundedness_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatSession:
    """Metadata for a multi-turn conversational RAG session."""

    session_id: str
    user_id: str = "anonymous"
    title: str = "New Chat"
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict = field(default_factory=dict)
