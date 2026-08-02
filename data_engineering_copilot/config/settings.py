from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
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
    model_config = SettingsConfigDict(
        frozen=True,
        populate_by_name=True,
        env_file=(".env", ".env.secrets", ".env.local"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Module-load seam ONLY (see module-level `settings` below).  Never set by tests.
    skip_provider_check: bool = Field(default=False, exclude=True)

    project_root: Path = PROJECT_ROOT
    documentation_sources_path: Path = (
        PROJECT_ROOT / "data_engineering_copilot" / "config" / "documentation_sources.json"
    )
    collection_name: str = "data_engineering_docs"

    # URLs accessed from localhost
    qdrant_url: str = "http://localhost:6333"
    ollama_base_url: str = "http://localhost:11434"
    # Separate Ollama instances for embedding vs LLM to prevent resource contention.
    # Default to the same value; set to different ports for dedicated instances.
    embedding_ollama_base_url: str = ""
    llm_ollama_base_url: str = ""

    # URLs accessed within docker
    redis_url: str = "redis://:local_secure_password_123@localhost:6379/0"
    langfuse_url: str = "http://langfuse:3000"
    langfuse_public_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGFUSE_SECRET_KEY",
    )
    langfuse_host: str = Field(
        default="http://langfuse:3000",
        validation_alias="LANGFUSE_HOST",
    )

    embedding_batch_size: int = 128
    embed_concurrency: int = 1
    enrichment_batch_size: int = 20

    # LLM Provider selection: "ollama" | "openrouter"
    # Embedding provider selection: "ollama" | "openrouter"
    embedding_model_name: str = "nomic-embed-text"
    # Embedding dimension is model-dependent, not provider-dependent.
    # Map model names to their known output dimensions.
    embedding_model_dimensions: dict[str, int] = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "snowflake-arctic-embed2": 1024,
        "llama3.2:3b": 3072,
        "nvidia/nemotron-3-embed-1b": 2048,
        "nvidia/nemotron-3-embed-1b:free": 2048,
        "text-embedding-004": 768,
    }
    default_embedding_dimension: int = 768
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    embedding_provider: str = "ollama"
    ollama_model: str = "llama3.2:3b"

    # API key for the API auth middleware (pydantic-set, env-file aware).
    api_key: SecretStr = SecretStr("")

    # OpenRouter settings (LLM + Embeddings)
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_model: str = "openrouter/free"
    openrouter_embedding_model: str = "nvidia/nemotron-3-embed-1b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_rpm_limit: int = 18
    openrouter_rpd_limit: int = 900

    # NVIDIA settings (LLM + Embeddings)
    nvidia_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
    )
    nvidia_model: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_embedding_model: str = "nvidia/nemotron-3-embed-1b"
    nvidia_rpm_limit: int = 36
    nvidia_rpd_limit: int = 10000

    # Groq settings (LLM only)
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_rpm_limit: int = 27
    groq_rpd_limit: int = 13000

    # Cerebras settings (LLM only)
    cerebras_api_key: SecretStr = SecretStr("")
    cerebras_model: str = "gpt-oss-120b"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_rpm_limit: int = 4
    cerebras_rpd_limit: int = 2200

    # Gemini settings (LLM + Embeddings)
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_embedding_model: str = "text-embedding-004"
    gemini_rpm_limit: int = 13
    gemini_rpd_limit: int = 450

    # Cloudflare Workers AI settings (LLM only)
    cloudflare_api_key: SecretStr = SecretStr("")
    cloudflare_model: str = ""
    cloudflare_account_id: str = ""
    cloudflare_base_url: str = ""
    cloudflare_rpm_limit: int = 60
    cloudflare_rpd_limit: int = 100000

    # LLM fallback chain: ordered list of providers to try on failure
    llm_fallback_order: list[str] = Field(
        default_factory=lambda: ["cloudflare", "groq", "openrouter", "gemini", "cerebras", "ollama"]
    )
    llm_fallback_call_timeout: int = 120  # per-attempt timeout for non-primary fallback providers

    # Provider cooldown / routing
    provider_cooldown_seconds: int = 60
    health_success_rate_weight: float = 0.6
    health_latency_weight: float = 0.2
    health_recency_weight: float = 0.2
    health_consecutive_failure_penalty: float = 0.3

    # After this many consecutive failures the degraded Ollama fallback is
    # skipped (fail fast) instead of stalling the request on a broken local
    # model. A single success resets the counter.
    ollama_degraded_max_consecutive_failures: int = 3

    # Per-purpose LLM overrides (empty = use global llm_provider / llm_model)
    answer_llm_provider: str = "openrouter"
    answer_llm_model: str = "openrouter/free"
    rewrite_llm_provider: str = "groq"
    rewrite_llm_model: str = ""
    groundedness_llm_provider: str = "groq"
    groundedness_llm_model: str = ""
    intent_llm_provider: str = "groq"
    intent_llm_model: str = ""
    enrichment_llm_provider: str = ""
    enrichment_llm_model: str = ""
    evaluation_llm_provider: str = "groq"
    evaluation_llm_model: str = ""
    # Code-specific LLM override (optional, for code_example/api_lookup intents)
    code_llm_provider: str = ""
    code_llm_model: str = ""

    # Per-provider purpose model overrides (override model per-provider per-purpose)
    # When set, these take priority over the generic {purpose}_llm_model above when
    # the effective provider matches. Format: {provider}_{purpose}_llm_model.
    openrouter_answer_llm_model: str = ""
    openrouter_rewrite_llm_model: str = ""
    openrouter_groundedness_llm_model: str = ""
    openrouter_intent_llm_model: str = ""
    openrouter_enrichment_llm_model: str = ""
    openrouter_evaluation_llm_model: str = ""
    openrouter_code_llm_model: str = ""
    nvidia_answer_llm_model: str = ""
    nvidia_rewrite_llm_model: str = ""
    nvidia_groundedness_llm_model: str = ""
    nvidia_intent_llm_model: str = ""
    nvidia_enrichment_llm_model: str = ""
    nvidia_evaluation_llm_model: str = ""
    nvidia_code_llm_model: str = ""
    groq_answer_llm_model: str = ""
    groq_rewrite_llm_model: str = ""
    groq_groundedness_llm_model: str = ""
    groq_intent_llm_model: str = ""
    groq_enrichment_llm_model: str = ""
    groq_evaluation_llm_model: str = ""
    groq_code_llm_model: str = ""
    cerebras_answer_llm_model: str = ""
    cerebras_rewrite_llm_model: str = ""
    cerebras_groundedness_llm_model: str = ""
    cerebras_intent_llm_model: str = ""
    cerebras_enrichment_llm_model: str = ""
    cerebras_evaluation_llm_model: str = ""
    cerebras_code_llm_model: str = ""
    gemini_answer_llm_model: str = ""
    gemini_rewrite_llm_model: str = ""
    gemini_groundedness_llm_model: str = ""
    gemini_intent_llm_model: str = ""
    gemini_enrichment_llm_model: str = ""
    gemini_evaluation_llm_model: str = ""
    gemini_code_llm_model: str = ""
    cloudflare_answer_llm_model: str = ""
    cloudflare_rewrite_llm_model: str = ""
    cloudflare_groundedness_llm_model: str = ""
    cloudflare_intent_llm_model: str = ""
    cloudflare_enrichment_llm_model: str = ""
    cloudflare_evaluation_llm_model: str = ""
    cloudflare_code_llm_model: str = ""

    ollama_code_llm_model: str = "qwen2.5-coder:7b"

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
    max_expansion_queries: int = 2
    context_compression_ratio: float = 0.8
    groundedness_threshold: float = 0.6
    confidence_threshold: float = 0.18
    request_timeout_seconds: int = 15
    ollama_timeout_seconds: int = 180
    ollama_connect_timeout_seconds: int = 5
    ollama_pool_timeout_seconds: int = 5
    ollama_keep_alive: str | int = "10m"
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 512
    ollama_retry_context_ratio: float = 0.5
    ollama_retry_extra_num_predict: int = 512
    ollama_retry_max_num_predict: int = 1024
    crawl_delay_seconds: float = 0.5
    max_pages_per_source: int = 80
    crawl_thread_pool_size: int = 4
    ingestion_batch_chunk_size: int = 256
    processing_concurrency: int = 2
    enrichment_concurrency: int = 1
    # Multi-stage pipeline concurrency (isolated executor pools)
    parse_concurrency: int = 4
    chunk_concurrency: int = 4
    store_concurrency: int = 2
    # Async crawler settings
    crawl_db_url: str = ""
    crawl_async_concurrency: int = 5
    crawl_async_max_concurrency: int = 40
    crawl_async_per_domain_concurrency: int = 1
    crawl_async_conditional_get: bool = True
    crawl_async_cache_url: str = ""
    crawl_async_thread_pool_size: int = 4
    # Maximum number of times a FAILED frontier URL is re-discovered before it
    # becomes terminal (stops being retried).
    frontier_max_attempts: int = 3
    logging_enabled: bool = True
    # Hybrid search
    hybrid_search_enabled: bool = True
    hybrid_rrf_k: int = 60
    # Semantic cache
    semantic_cache_threshold: float = 0.95
    semantic_cache_ttl: int = 3600
    # Query rewriting / grounding
    query_rewrite_enabled: bool = True
    groundedness_enabled: bool = True
    intent_classification_llm_enabled: bool = False  # Enable LLM fallback for intent classification
    # Context management
    context_compression_enabled: bool = False
    max_context_tokens: int = 4096
    # Post-processing toggles
    contextual_enrichment_enabled: bool = True
    api_extraction_enabled: bool = True
    code_block_parsing_enabled: bool = True
    chunk_filtering_enabled: bool = True
    # PII redaction
    pii_redaction_enabled: bool = True
    pii_redaction_mode: str = "full"  # full | masked | none
    # Indirect prompt injection guard for retrieved documents
    input_guardrails_enabled: bool = True
    # RBAC (document-level access control)
    rbac_enabled: bool = False
    rbac_users_json: str = ""  # inline JSON string: {"key_prefix": {"allowed_sources": [...], "role": "reader"}}
    # Drift detection
    drift_detection_enabled: bool = True
    drift_eval_history_path: str = "data/eval_history.jsonl"
    drift_window_days: int = 7
    sources: tuple[DocumentationSource, ...] = ()

    @model_validator(mode="after")
    def _load_sources_from_json(self) -> AppSettings:
        if not self.sources:
            object.__setattr__(self, "sources", load_documentation_sources(self.documentation_sources_path))
        return self

    @model_validator(mode="after")
    def _resolve_cloudflare_base_url(self) -> AppSettings:
        if not self.cloudflare_base_url and self.cloudflare_account_id:
            object.__setattr__(
                self,
                "cloudflare_base_url",
                f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/v1",
            )
        return self

    @model_validator(mode="after")
    def _validate_provider_api_keys(self) -> AppSettings:
        if self.skip_provider_check:
            return self
        all_llm_providers = {
            p
            for p in [
                self.llm_provider,
                self.answer_llm_provider,
                self.rewrite_llm_provider,
                self.groundedness_llm_provider,
                self.intent_llm_provider,
                self.enrichment_llm_provider,
                self.evaluation_llm_provider,
                self.code_llm_provider,
            ]
            if p
        }
        # Each API-key-gated provider is checked: if referenced by any LLM purpose
        # or as the embedding provider, its API key must be set.
        provider_api_key_map: dict[str, tuple[str, str]] = {
            "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
            "nvidia": ("nvidia_api_key", "NVIDIA_API_KEY"),
            "groq": ("groq_api_key", "GROQ_API_KEY"),
            "cerebras": ("cerebras_api_key", "CEREBRAS_API_KEY"),
            "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
            "cloudflare": ("cloudflare_api_key", "CLOUDFLARE_API_KEY"),
        }
        for provider, (field_name, env_var) in provider_api_key_map.items():
            key_value = getattr(self, field_name).get_secret_value()
            if provider in all_llm_providers and not key_value:
                raise ValueError(f"{env_var} is required when any LLM provider is '{provider}'")
            if self.embedding_provider == provider and not key_value:
                raise ValueError(f"{env_var} is required when EMBEDDING_PROVIDER='{provider}'")
        return self

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension for the active model.

        Dimension is model-dependent — looks up the configured model in
        ``embedding_model_dimensions`` and falls back to
        ``default_embedding_dimension`` if the model is unrecognised.
        """
        provider = self.embedding_provider.lower()
        if provider == "openrouter":
            model_name = self.openrouter_embedding_model
        elif provider == "nvidia":
            model_name = self.nvidia_embedding_model
        elif provider == "gemini":
            model_name = self.gemini_embedding_model
        else:
            model_name = self.embedding_model_name
        return self.embedding_model_dimensions.get(model_name, self.default_embedding_dimension)

    def validate_all(self) -> None:
        """Cross-field consistency checks. Raises ``ValidationError`` on conflicts."""
        from pydantic import ValidationError

        errors: list[str] = []

        if self.reranker_enabled and self.reranker_top_k > self.retrieval_top_k:
            errors.append(
                f"reranker_top_k ({self.reranker_top_k}) must be <= retrieval_top_k ({self.retrieval_top_k}) "
                "— the reranker can only narrow the retrieved set."
            )
        if self.max_pages_per_source < 0:
            errors.append(f"max_pages_per_source ({self.max_pages_per_source}) must be >= 0")

        configured_providers = {
            p
            for p in [
                self.llm_provider,
                self.answer_llm_provider,
                self.rewrite_llm_provider,
                self.groundedness_llm_provider,
                self.intent_llm_provider,
                self.enrichment_llm_provider,
                self.evaluation_llm_provider,
                self.code_llm_provider,
                self.embedding_provider,
            ]
            if p
        }
        if not configured_providers:
            errors.append("At least one LLM or embedding provider must be configured")

        if errors:
            from typing import cast

            from pydantic_core import InitErrorDetails

            line_errors = [
                {
                    "type": "value_error",
                    "loc": ("settings",),
                    "msg": msg,
                    "input": None,
                    "ctx": {"error": ValueError(msg)},
                }
                for msg in errors
            ]
            raise ValidationError.from_exception_data(
                title="AppSettings.validate_all",
                line_errors=cast(list[InitErrorDetails], line_errors),
            )


# Module-load seam ONLY: skip the API-key validator so importing this module
# never hard-fails on missing provider API keys (e.g. in CI or a fresh clone
# where .env.secrets is absent).  Provider builders in factory.py still
# validate API keys and raise clear errors when a non-Ollama provider is
# built.  NOT a test escape hatch — tests use tests.conftest.make_settings()
# (which is hermetic and never reads .env) and must not set this flag.
settings = AppSettings(skip_provider_check=True)
