from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DocumentationSource:
    name: str
    start_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    url_prefixes: tuple[str, ...] = ()
    priority: int = 1  # Crawl priority (higher = more concurrency slots)


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


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    project_root: Path = PROJECT_ROOT
    documentation_sources_path: Path = (
        PROJECT_ROOT / "data_engineering_copilot" / "config" / "documentation_sources.json"
    )
    collection_name: str = "data_engineering_docs"

    # URLs accessed from localhost
    qdrant_url: str = "http://localhost:6333"
    ollama_base_url: str = "http://localhost:11434"

    # URLs accessed within docker
    redis_url: str = "redis://redis:6379/0"
    langfuse_url: str = "http://langfuse:3000"
    langfuse_public_key: str = Field(
        default="pk-lf-ff6ebcae-7f5f-470a-92b9-cd78ed04a8be",
        validation_alias="LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: str = Field(
        default="sk-lf-30b5912b-5882-4fe3-acfd-ecf0e38d1bb1",
        validation_alias="LANGFUSE_SECRET_KEY",
    )
    langfuse_host: str = Field(
        default="http://langfuse:3000",
        validation_alias="LANGFUSE_HOST",
    )

    embedding_model_name: str = "nomic-embed-text"
    # Default dm is 768 for the nomic-embed-text model
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

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
    crawl_thread_pool_size: int = 4
    ingestion_batch_chunk_size: int = 256
    processing_concurrency: int = 4
    # Async crawler settings
    crawl_db_path: Path = PROJECT_ROOT / "data" / "crawl_frontier.db"
    crawl_async_concurrency: int = 20
    crawl_async_max_concurrency: int = 40
    crawl_async_per_domain_concurrency: int = 3
    crawl_async_conditional_get: bool = True
    crawl_async_cache_url: str = ""
    crawl_async_thread_pool_size: int = 4
    logging_enabled: bool = True
    sources: tuple[DocumentationSource, ...] = ()

    @model_validator(mode="after")
    def _load_sources_from_json(self) -> AppSettings:
        if not self.sources:
            object.__setattr__(self, "sources", load_documentation_sources(self.documentation_sources_path))
        return self


settings = AppSettings()
