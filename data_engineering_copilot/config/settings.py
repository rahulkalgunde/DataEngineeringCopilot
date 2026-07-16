from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DocumentationSource:
    name: str
    start_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    url_prefixes: tuple[str, ...] = ()


def load_documentation_sources(config_path: Path) -> tuple[DocumentationSource, ...]:
    with config_path.open("r", encoding="utf-8") as file:
        raw_sources = json.load(file)

    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"Documentation source config must contain a non-empty list: {config_path}")

    sources: list[DocumentationSource] = []
    for index, raw_source in enumerate(raw_sources, start=1):
        if not isinstance(raw_source, dict):
            raise ValueError(f"Documentation source #{index} must be an object.")

        name = _required_string(raw_source, "name", index)
        start_urls = _required_string_tuple(raw_source, "start_urls", index)
        allowed_domains = _required_string_tuple(raw_source, "allowed_domains", index)
        url_prefixes = _optional_string_tuple(raw_source, "url_prefixes", index)

        sources.append(
            DocumentationSource(
                name=name,
                start_urls=start_urls,
                allowed_domains=allowed_domains,
                url_prefixes=url_prefixes,
            )
        )

    return tuple(sources)


def _required_string(raw_source: dict, field_name: str, index: int) -> str:
    value = raw_source.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Documentation source #{index} must define a non-empty `{field_name}` string.")
    return value.strip()


def _required_string_tuple(raw_source: dict, field_name: str, index: int) -> tuple[str, ...]:
    value = _optional_string_tuple(raw_source, field_name, index)
    if not value:
        raise ValueError(f"Documentation source #{index} must define at least one `{field_name}` value.")
    return value


def _optional_string_tuple(raw_source: dict, field_name: str, index: int) -> tuple[str, ...]:
    value = raw_source.get(field_name, [])
    if not isinstance(value, list):
        raise ValueError(f"Documentation source #{index} field `{field_name}` must be a list.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Documentation source #{index} field `{field_name}` must contain only non-empty strings.")
    return tuple(item.strip() for item in value)


@dataclass(frozen=True)
class AppSettings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    # Legacy Chroma directory – retained for compatibility; not used in production.
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    # Qdrant persistence directory (mounted volume for Docker). Used by Qdrant container.
    qdrant_dir: Path = PROJECT_ROOT / "qdrant_db"
    documentation_sources_path: Path = PROJECT_ROOT / "data_engineering_copilot" / "config" / "documentation_sources.json"
    embedding_cache_dir: Path = PROJECT_ROOT / "data" / "embedding_models"
    collection_name: str = "data_engineering_docs"
    
    # URLs accessed from localhost
    qdrant_url: str = "http://localhost:6333"
    ollama_base_url: str = "http://localhost:11434"

    # URLs accessed within docker
    redis_url: str = "redis://redis:6379/0"
    langfuse_url: str = "http://langfuse:3000"
    langfuse_public_key: str = field(
        default_factory=lambda: os.environ.get(
            "LANGFUSE_PUBLIC_KEY", "pk-lf-3423e765-515a-46b1-a8ae-879ccac4d091"
        )
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.environ.get(
            "LANGFUSE_SECRET_KEY", "sk-lf-7fcdd8d3-9758-4ec7-a19b-7da6e1213c6d"
        )
    )
    langfuse_host: str = field(
        default_factory=lambda: os.environ.get(
            "LANGFUSE_HOST", "http://langfuse:3000"
        )
    )

    embedding_model_name: str = "nomic-embed-text"
    # Default dm is 768 for the nomic-embed-text model
    embedding_dimension: int = 768
    embedding_batch_size: int = 32
    embedding_local_files_only: bool = True
    
    ollama_model: str = "llama3.2:3b"
    # Chunking strategy: "fixed_size", "sentence_preserving", or "semantic"
    chunking_strategy: str = "sentence_preserving"
    chunk_size_words: int = 375
    chunk_overlap_words: int = 90
    # Semantic chunker specific settings
    min_semantic_similarity: float = 0.5
    max_chunk_words: int | None = None  # Auto: 1.5x chunk_size_words if None
    # Feature flags
    enable_semantic_chunking: bool = True  # Enable semantic chunker (requires embedding model)
    retrieval_top_k: int = 15
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5
    max_context_chars: int = 4000
    confidence_threshold: float = 0.18
    request_timeout_seconds: int = 15
    ollama_timeout_seconds: int = 300
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 512
    ollama_retry_context_ratio: float = 0.5
    ollama_retry_extra_num_predict: int = 512
    ollama_retry_max_num_predict: int = 1024
    crawl_delay_seconds: float = 0.2
    max_pages_per_source: int = 80
    ingestion_batch_chunk_size: int = 256
    logging_enabled: bool = True
    sources: tuple[DocumentationSource, ...] = load_documentation_sources(documentation_sources_path)


settings = AppSettings()
