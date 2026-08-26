from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import redis.asyncio as aioredis
import redis.exceptions
import structlog

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.domain.models import RagConfig, RerankRequest, RerankResult

if TYPE_CHECKING:
    from data_engineering_copilot.services.conversation_rag import ConversationService
    from data_engineering_copilot.services.pipeline_lab import PipelineLab
from data_engineering_copilot.domain.protocols import EmbedderProtocol, LLMClientProtocol
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import OpenAICompatibleEmbeddings
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.infrastructure.cooldown_aware_router import CooldownAwareEmbeddingRouter
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache, NoOpCrawlCache
from data_engineering_copilot.infrastructure.crawl_db import PostgresCrawlFrontierDB
from data_engineering_copilot.infrastructure.error_categorization import categorize_provider_error
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.infrastructure.huggingface_serverless_embeddings import (
    HuggingFaceServerlessEmbeddings,
)
from data_engineering_copilot.infrastructure.llm_client import LLMClient
from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import (
    LocalSentenceTransformerEmbeddings,
)
from data_engineering_copilot.infrastructure.provider_fallback import (
    FallbackChainConfig,
    ProviderConfig,
    ProviderFallbackChain,
)
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.infrastructure.provider_selector import (
    InMemoryRouterState,
    ProviderRouterConfig,
    ProviderSelector,
    RedisRouterState,
    RouterStateBackend,
)
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter
from data_engineering_copilot.infrastructure.rst_parser import RstParser
from data_engineering_copilot.infrastructure.tokenizer_registry import declared_input_limit, token_counter_for
from data_engineering_copilot.observability.token_tracker import RetrievalTracker, TokenTracker
from data_engineering_copilot.services.api_extractor import ApiDocExtractor
from data_engineering_copilot.services.async_ingestion import AsyncIngestionService
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.chunker_router import ChunkerRouter
from data_engineering_copilot.services.code_block_parser import CodeBlockParser
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.semantic_chunker import SemanticChunker
from data_engineering_copilot.services.sentence_preserving_chunker import SentencePreservingChunker
from data_engineering_copilot.services.text_filter import ChunkFilter

logger = structlog.get_logger(__name__)

# Shared async Redis client so all components (ingestion URL registry, crawl
# cache, query cache) reuse one connection pool instead of each opening its own.
_shared_redis: aioredis.Redis | None = None


def get_shared_redis_client(redis_url: str | None = None) -> aioredis.Redis:
    """Return a process-wide shared async Redis client.

    All components that need Redis (ingestion URL registry, crawl cache, query
    cache) should call this instead of creating their own ``aioredis.from_url``
    so connection usage is bounded and pool sizing stays predictable.
    """
    global _shared_redis
    if _shared_redis is None:
        _shared_redis = aioredis.from_url(
            redis_url or settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
    return _shared_redis


def _build_provider_rate_limiters(app_settings: AppSettings = settings) -> dict[str, SlidingWindowRateLimiter]:
    """Create one shared rate limiter per unique API-key-gated provider.

    Collects all provider names referenced across global LLM, per-purpose
    LLM, and embedding config, then creates one ``SlidingWindowRateLimiter``
    per provider that needs rate limiting.  Providers without API limits
    (e.g. local Ollama) are excluded.
    """
    providers: set[str] = set()
    providers.add(app_settings.llm_provider.lower())
    for p in [
        app_settings.answer_llm_provider,
        app_settings.rewrite_llm_provider,
        app_settings.groundedness_llm_provider,
        app_settings.intent_llm_provider,
        app_settings.enrichment_llm_provider,
        app_settings.evaluation_llm_provider,
        app_settings.code_llm_provider,
    ]:
        if p:
            providers.add(p.lower())
    providers.add(app_settings.embedding_provider.lower())
    for p in app_settings.embedding_fallback_order:
        providers.add(p.lower())
    for p in app_settings.llm_fallback_order:
        providers.add(p.lower())
    for p in app_settings.rerank_fallback_order:
        providers.add(p.lower())

    rate_limiters: dict[str, SlidingWindowRateLimiter] = {}
    for p in providers:
        if p == "openrouter":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.openrouter_rpm_limit,
                rpd_limit=app_settings.openrouter_rpd_limit,
            )
        elif p == "nvidia":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.nvidia_rpm_limit,
                rpd_limit=app_settings.nvidia_rpd_limit,
            )
        elif p == "groq":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.groq_rpm_limit,
                rpd_limit=app_settings.groq_rpd_limit,
            )
        elif p == "cerebras":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.cerebras_rpm_limit,
                rpd_limit=app_settings.cerebras_rpd_limit,
            )
        elif p == "gemini":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.gemini_rpm_limit,
                rpd_limit=app_settings.gemini_rpd_limit,
            )
        elif p == "cloudflare":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.cloudflare_rpm_limit,
                rpd_limit=app_settings.cloudflare_rpd_limit,
            )
        elif p == "opencodezen":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.opencodezen_rpm_limit,
                rpd_limit=app_settings.opencodezen_rpd_limit,
            )
        elif p == "opencodego":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.opencodego_rpm_limit,
                rpd_limit=app_settings.opencodego_rpd_limit,
            )
        elif p == "sambanova":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.sambanova_rpm_limit,
                rpd_limit=app_settings.sambanova_rpd_limit,
            )
        elif p == "mistral":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.mistral_rpm_limit,
                rpd_limit=app_settings.mistral_rpd_limit,
            )
        elif p == "deepseek":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.deepseek_rpm_limit,
                rpd_limit=app_settings.deepseek_rpd_limit,
            )
        elif p == "zai":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.zai_rpm_limit,
                rpd_limit=app_settings.zai_rpd_limit,
            )
        elif p == "siliconflow":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.siliconflow_rpm_limit,
                rpd_limit=app_settings.siliconflow_rpd_limit,
            )
        elif p == "together":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.together_rpm_limit,
                rpd_limit=app_settings.together_rpd_limit,
            )
        elif p == "fireworks":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.fireworks_rpm_limit,
                rpd_limit=app_settings.fireworks_rpd_limit,
            )
        elif p == "llm7":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.llm7_rpm_limit,
                rpd_limit=app_settings.llm7_rpd_limit,
            )
        elif p == "agnes":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.agnes_rpm_limit,
                rpd_limit=app_settings.agnes_rpd_limit,
            )
        elif p == "ollama_cloud":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.ollama_cloud_rpm_limit,
                rpd_limit=app_settings.ollama_cloud_rpd_limit,
            )
        elif p == "helyx":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.helyx_rpm_limit,
                rpd_limit=app_settings.helyx_rpd_limit,
            )
        elif p == "anyapi":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.anyapi_rpm_limit,
                rpd_limit=app_settings.anyapi_rpd_limit,
            )
        elif p == "huggingface":
            rate_limiters[p] = SlidingWindowRateLimiter(
                rpm_limit=app_settings.huggingface_rpm_limit,
                rpd_limit=app_settings.huggingface_rpd_limit,
            )
    return rate_limiters


def _build_provider_health_registry(app_settings: AppSettings = settings) -> ProviderHealthRegistry:
    return ProviderHealthRegistry(
        success_rate_weight=app_settings.health_success_rate_weight,
        latency_weight=app_settings.health_latency_weight,
        recency_weight=app_settings.health_recency_weight,
        consecutive_failure_penalty=app_settings.health_consecutive_failure_penalty,
        default_cooldown_seconds=app_settings.provider_cooldown_seconds,
    )


def _is_model_not_supported_text(text: str) -> bool:
    """Heuristic for model-not-supported error bodies (e.g. opencodego's
    ``ModelError: "Model X is not supported"``), which some gateways report as
    HTTP 401. A 401 carrying this body is a wrong-model error, not an auth
    failure, so it must be categorised as ``INVALID_REQUEST``."""
    lowered = (text or "").lower()
    return (
        "not supported" in lowered
        or "modelerror" in lowered
        or "does not exist" in lowered
        or "unknown model" in lowered
        or ("model" in lowered and "not found" in lowered)
    )


_categorize_embedding_error = categorize_provider_error


def _build_purpose_llm_client(
    provider: str,
    model: str | None = None,
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
    timeout_seconds: int | None = None,
    purpose: str | None = None,
    apply_purpose_model_override: bool = True,
) -> LLMClient | None:
    """Build an LLM client for a specific purpose.

    When both *provider* and *model* are empty/blank the caller intends to
    fall back to the global ``llm_provider`` / ``llm_model``.  Returns
    ``None`` in that case so the factory can reuse a shared global client.

    Model resolution priority:
        1. *model* — explicit purpose-level override
        2. ``{provider}_{purpose}_llm_model`` — per-provider purpose override
           (only when *apply_purpose_model_override* is True, i.e. the primary
           provider in a fallback chain)
        3. ``{provider}_model`` — provider default model
        4. ``llm_model`` — global default model

    Fallback providers in a chain always resolve to their own ``{provider}_model``
    default — a purpose-specific model that happens to be pinned for the primary
    provider must never be forwarded to a provider that does not host it.
    """
    eff_provider = (provider or app_settings.llm_provider).lower()

    # Model resolution: explicit > per-provider purpose > provider default > global
    eff_model = model or ""
    if not eff_model and purpose and apply_purpose_model_override:
        purpose_override = getattr(app_settings, f"{eff_provider}_{purpose}_llm_model", "")
        if purpose_override:
            eff_model = purpose_override
    if not eff_model:
        provider_model = getattr(app_settings, f"{eff_provider}_model", "")
        eff_model = provider_model or app_settings.llm_model
    if not eff_provider or not eff_model:
        return None

    logger.info(
        "resolved_llm_client",
        purpose=purpose or "global",
        provider=eff_provider,
        model=eff_model,
    )

    # Per-purpose output cap (sent as max_tokens / max_completion_tokens).
    purpose_max_tokens = app_settings.purpose_max_tokens.get((purpose or "global").lower(), app_settings.llm_max_tokens)

    rate_limiter = (provider_rate_limiters or {}).get(eff_provider)

    _gen_kwargs: dict = {}
    if purpose == "answer":
        _gen_kwargs["temperature"] = app_settings.generation_temperature
    elif purpose == "code":
        _gen_kwargs["temperature"] = app_settings.code_generation_temperature
    elif purpose == "evaluation":
        _gen_kwargs["temperature"] = app_settings.evaluation_temperature
    if app_settings.generation_seed is not None:
        _gen_kwargs["seed"] = app_settings.generation_seed
    if app_settings.generation_frequency_penalty:
        _gen_kwargs["frequency_penalty"] = app_settings.generation_frequency_penalty
    if app_settings.generation_presence_penalty:
        _gen_kwargs["presence_penalty"] = app_settings.generation_presence_penalty
    if app_settings.generation_top_p != 1.0:
        _gen_kwargs["top_p"] = app_settings.generation_top_p
    if purpose in ("answer", "code"):
        from data_engineering_copilot.services.structured_output import STRUCTURED_RAG_ANSWER_SCHEMA

        _gen_kwargs["structured_schema"] = STRUCTURED_RAG_ANSWER_SCHEMA

    def _make(**base) -> LLMClient:
        merged = {**base, "provider": eff_provider, **_gen_kwargs}
        return LLMClient(**merged)

    if eff_provider == "ollama":
        llm_base = app_settings.llm_ollama_base_url or app_settings.ollama_base_url
        # Respect the per-purpose token budget (parity with other providers);
        # ollama_num_predict only applies to the purpose-less/global path
        # (legacy contract pinned by test_ollama_sends_max_tokens_not_ignored_options).
        purpose_budget = (
            app_settings.purpose_max_tokens.get(purpose.lower()) if purpose and purpose.lower() != "global" else None
        )
        return _make(
            base_url=f"{llm_base}/v1",
            model=eff_model,
            api_key="",
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_budget or app_settings.ollama_num_predict,
            connect_timeout_seconds=app_settings.ollama_connect_timeout_seconds,
            pool_timeout_seconds=app_settings.ollama_pool_timeout_seconds,
        )

    if eff_provider == "openrouter":
        api_key = app_settings.openrouter_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when provider='openrouter'")
        return _make(
            base_url=app_settings.openrouter_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_completion_tokens",
            extra_headers={"HTTP-Referer": "https://data-engineering-copilot.local"},
            rate_limiter=rate_limiter,
        )

    if eff_provider == "nvidia":
        api_key = app_settings.nvidia_api_key.get_secret_value()
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required when provider='nvidia'")
        return _make(
            base_url=app_settings.nvidia_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            rate_limiter=rate_limiter,
        )

    if eff_provider == "groq":
        api_key = app_settings.groq_api_key.get_secret_value()
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when provider='groq'")
        return _make(
            base_url=app_settings.groq_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_completion_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "cerebras":
        api_key = app_settings.cerebras_api_key.get_secret_value()
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY is required when provider='cerebras'")
        return _make(
            base_url=app_settings.cerebras_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_completion_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "gemini":
        api_key = app_settings.gemini_api_key.get_secret_value()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when provider='gemini'")
        return _make(
            base_url=app_settings.gemini_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            rate_limiter=rate_limiter,
        )

    if eff_provider == "cloudflare":
        api_key = app_settings.cloudflare_api_key.get_secret_value()
        if not api_key:
            raise ValueError("CLOUDFLARE_API_KEY is required when provider='cloudflare'")
        return _make(
            base_url=app_settings.cloudflare_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            rate_limiter=rate_limiter,
        )

    if eff_provider == "opencodezen":
        api_key = app_settings.opencodezen_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENCODEZEN_API_KEY is required when provider='opencodezen'")
        return _make(
            base_url=app_settings.opencodezen_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_completion_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "opencodego":
        api_key = app_settings.opencodego_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENCODEGO_API_KEY is required when provider='opencodego'")
        return _make(
            base_url=app_settings.opencodego_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_completion_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "sambanova":
        api_key = app_settings.sambanova_api_key.get_secret_value()
        if not api_key:
            raise ValueError("SAMBANOVA_API_KEY is required when provider='sambanova'")
        return _make(
            base_url=app_settings.sambanova_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_completion_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "mistral":
        api_key = app_settings.mistral_api_key.get_secret_value()
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required when provider='mistral'")
        return _make(
            base_url=app_settings.mistral_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_completion_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "deepseek":
        api_key = app_settings.deepseek_api_key.get_secret_value()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when provider='deepseek'")
        return _make(
            base_url=app_settings.deepseek_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "zai":
        api_key = app_settings.zai_api_key.get_secret_value()
        if not api_key:
            raise ValueError("ZAI_API_KEY is required when provider='zai'")
        return _make(
            base_url=app_settings.zai_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "siliconflow":
        api_key = app_settings.siliconflow_api_key.get_secret_value()
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is required when provider='siliconflow'")
        return _make(
            base_url=app_settings.siliconflow_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "together":
        api_key = app_settings.together_api_key.get_secret_value()
        if not api_key:
            raise ValueError("TOGETHER_API_KEY is required when provider='together'")
        return _make(
            base_url=app_settings.together_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "fireworks":
        api_key = app_settings.fireworks_api_key.get_secret_value()
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is required when provider='fireworks'")
        return _make(
            base_url=app_settings.fireworks_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "ollama_cloud":
        api_key = app_settings.ollama_cloud_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OLLAMA_API_KEY is required when provider='ollama_cloud'")
        return _make(
            base_url=f"{app_settings.ollama_cloud_base_url}/v1",
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=app_settings.ollama_num_predict,
            connect_timeout_seconds=app_settings.ollama_connect_timeout_seconds,
            pool_timeout_seconds=app_settings.ollama_pool_timeout_seconds,
            rate_limiter=rate_limiter,
        )

    if eff_provider == "llm7":
        api_key = app_settings.llm7_api_key.get_secret_value()
        if not api_key:
            raise ValueError("LLM7_API_KEY is required when provider='llm7'")
        return _make(
            base_url=app_settings.llm7_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "agnes":
        api_key = app_settings.agnes_api_key.get_secret_value()
        if not api_key:
            raise ValueError("AGNES_API_KEY is required when provider='agnes'")
        return _make(
            base_url=app_settings.agnes_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "helyx":
        api_key = app_settings.helyx_api_key.get_secret_value()
        if not api_key:
            raise ValueError("HELYX_API_KEY is required when provider='helyx'")
        return _make(
            base_url=app_settings.helyx_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    if eff_provider == "anyapi":
        api_key = app_settings.anyapi_api_key.get_secret_value()
        if not api_key:
            raise ValueError("ANYAPI_API_KEY is required when provider='anyapi'")
        return _make(
            base_url=app_settings.anyapi_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            max_tokens=purpose_max_tokens,
            max_tokens_field="max_tokens",
            rate_limiter=rate_limiter,
        )

    raise ValueError(
        f"Unsupported LLM provider: {eff_provider!r}. Supported: 'ollama', 'openrouter', 'nvidia', 'groq', 'cerebras', 'gemini', 'cloudflare', 'opencodezen', 'opencodego', 'sambanova', 'mistral', 'deepseek', 'zai', 'siliconflow', 'together', 'fireworks', 'llm7', 'agnes', 'ollama_cloud', 'helyx', 'anyapi'."
    )


def _build_chain_clients(
    ordered: list[str],
    app_settings: AppSettings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None,
    purpose: str | None,
    health_registry: ProviderHealthRegistry,
    purpose_model: str = "",
) -> list[tuple[str, LLMClient]]:
    """Build one ``LLMClient`` per provider in *ordered* (skipping unusable ones).

    Providers without a configured API key, or that cannot be resolved, are
    skipped. ``purpose_model`` is only applied to the first entry (the primary);
    every other provider falls back to its own default model (the per-provider
    purpose override is suppressed for fallbacks so a purpose model pinned for
    the primary is never forwarded to a provider that does not host it).
    """
    clients: list[tuple[str, LLMClient]] = []
    client_timeout = app_settings.llm_fallback_call_timeout

    for idx, provider in enumerate(ordered):
        try:
            model_arg = purpose_model if idx == 0 else ""
            timeout = (
                None if idx == 0 else (app_settings.ollama_timeout_seconds if provider == "ollama" else client_timeout)
            )
            client = _build_purpose_llm_client(
                provider=provider,
                model=model_arg,
                app_settings=app_settings,
                provider_rate_limiters=provider_rate_limiters,
                timeout_seconds=timeout,
                purpose=purpose,
                apply_purpose_model_override=(idx == 0),
            )
            if client is not None:
                clients.append((provider, client))
                health_registry.register_provider(provider, [client.model])
        except Exception as exc:
            logger.warning(
                "Skipping provider in fallback chain",
                provider=provider,
                error=str(exc),
            )

    return clients


def get_catalog_fallback_order(purpose: str, app_settings: AppSettings) -> list[str] | None:
    """Return catalog-driven order for *purpose* if catalog_auto_order is enabled.

    Fail-open: missing/stale/empty catalog returns ``None``.
    """
    if not getattr(app_settings, "catalog_auto_order", False):
        return None
    try:
        from data_engineering_copilot.services.provider_catalog import (  # noqa: PLC0415
            get_catalog_fallback_order as _get_order,
        )
        from data_engineering_copilot.services.provider_catalog import (
            is_catalog_stale,
            load_provider_catalog,
        )

        catalog = load_provider_catalog(app_settings.provider_catalog_path)
        if catalog is None:
            logger.info("catalog_auto_order_no_catalog", purpose=purpose)
            return None
        if is_catalog_stale(catalog, stale_days=getattr(app_settings, "catalog_stale_days", 7)):
            logger.warning("catalog_auto_order_stale", purpose=purpose, generated_at=catalog.generated_at)
            return None
        order = _get_order(purpose, catalog)
        if order:
            logger.info("catalog_auto_order_used", purpose=purpose, order=order)
            return order
    except Exception as exc:  # noqa: BLE001
        logger.warning("catalog_auto_order_failed", purpose=purpose, error=str(exc))
    return None


def get_catalog_embedding_order(app_settings: AppSettings) -> list[str] | None:
    if not getattr(app_settings, "catalog_auto_order", False):
        return None
    try:
        from data_engineering_copilot.services.provider_catalog import (  # noqa: PLC0415
            is_catalog_stale,
            load_provider_catalog,
        )

        catalog = load_provider_catalog(app_settings.provider_catalog_path)
        if catalog is None or is_catalog_stale(catalog, stale_days=getattr(app_settings, "catalog_stale_days", 7)):
            return None
        if catalog.embedding_fallback_order:
            logger.info("catalog_embedding_order_used", order=catalog.embedding_fallback_order)
            return list(catalog.embedding_fallback_order)
    except Exception as exc:  # noqa: BLE001
        logger.warning("catalog_embedding_order_failed", error=str(exc))
    return None


def get_catalog_rerank_order(app_settings: AppSettings) -> list[str] | None:
    if not getattr(app_settings, "catalog_auto_order", False):
        return None
    try:
        from data_engineering_copilot.services.provider_catalog import (  # noqa: PLC0415
            is_catalog_stale,
            load_provider_catalog,
        )

        catalog = load_provider_catalog(app_settings.provider_catalog_path)
        if catalog is None or is_catalog_stale(catalog, stale_days=getattr(app_settings, "catalog_stale_days", 7)):
            return None
        if catalog.rerank_fallback_order:
            logger.info("catalog_rerank_order_used", order=catalog.rerank_fallback_order)
            return list(catalog.rerank_fallback_order)
    except Exception as exc:  # noqa: BLE001
        logger.warning("catalog_rerank_order_failed", error=str(exc))
    return None


def _build_llm_chain_config(
    purpose: str,
    app_settings: AppSettings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None,
    health_registry: ProviderHealthRegistry,
    purpose_provider: str | None = None,
    purpose_model: str | None = None,
) -> FallbackChainConfig:
    """Build FallbackChainConfig for LLM providers based on purpose."""
    from data_engineering_copilot.infrastructure.provider_fallback import ProviderConfig

    limiters = provider_rate_limiters or {}

    # Catalog auto-order: when enabled and a fresh probe catalog exists, use
    # fastest-OK-per-provider order; otherwise fall back to settings order.
    # Explicit purpose_provider pin wins over catalog. When the caller omits
    # the pin, derive it from settings.{purpose}_llm_provider so the documented
    # per-purpose env pins stay authoritative everywhere (a missing arg must
    # not silently demote a pinned provider behind catalog auto-order).
    if not (purpose_provider and purpose_provider.strip()):
        purpose_provider = getattr(app_settings, f"{purpose}_llm_provider", "") or ""
    if purpose_provider and purpose_provider.strip():
        primary = purpose_provider.strip().lower()
        ordered = [primary]
        for p in app_settings.llm_fallback_order:
            p_lower = p.lower()
            if p_lower not in ordered:
                ordered.append(p_lower)
    else:
        catalog_order = get_catalog_fallback_order(purpose, app_settings)
        if catalog_order is not None:
            ordered = [p.lower() for p in catalog_order]
        else:
            ordered = [p.lower() for p in app_settings.llm_fallback_order]

    # Build ProviderConfig for each
    providers_config: list[ProviderConfig] = []
    for idx, provider_name in enumerate(ordered):
        try:
            model_arg = purpose_model if idx == 0 else ""
            timeout = (
                None
                if idx == 0
                else (
                    app_settings.ollama_timeout_seconds
                    if provider_name == "ollama"
                    else app_settings.llm_fallback_call_timeout
                )
            )
            client = _build_purpose_llm_client(
                provider=provider_name,
                model=model_arg,
                app_settings=app_settings,
                provider_rate_limiters=limiters,
                timeout_seconds=timeout,
                purpose=purpose,
            )
            if client is not None:
                providers_config.append(
                    ProviderConfig(
                        name=provider_name,
                        client=client,
                        rate_limiter=limiters.get(provider_name),
                    )
                )
                health_registry.register_provider(provider_name, [client.model])
        except Exception as exc:
            logger.warning(
                "Skipping provider in LLM fallback chain",
                provider=provider_name,
                error=str(exc),
            )

    # Separate Ollama as degraded fallback
    main_providers = [p for p in providers_config if p.name.lower() != "ollama"]
    degraded = next((p for p in providers_config if p.name.lower() == "ollama"), None)

    return FallbackChainConfig(
        providers=main_providers,
        degraded_fallback=degraded,
        max_degraded_consecutive_failures=app_settings.ollama_degraded_max_consecutive_failures,
        router_deadline_seconds=app_settings.router_deadline_seconds,
    )


def _build_llm_router(
    config: FallbackChainConfig,
    health: ProviderHealthRegistry,
    app_settings: AppSettings,
    purpose: str,
    preference_provider: str | None = None,
) -> ProviderSelector | None:
    """Build the availability-aware router for an LLM chain.

    The selector uses a Redis-backed state when ``router_redis_sharing`` is
    enabled (cooldowns + cached best provider shared across processes) and an
    in-memory state otherwise.  Returns ``None`` when there is nothing to route
    (no external providers), in which case the chain falls back to its legacy
    serial behaviour.
    """
    from data_engineering_copilot.infrastructure.provider_selector import ProviderSelector

    external = [p for p in config.providers if p.name.lower() != "ollama"]
    if not external:
        return None

    state: RouterStateBackend
    if app_settings.router_redis_sharing:
        try:
            from data_engineering_copilot.factory import get_shared_redis_client

            state = RedisRouterState(redis_client=get_shared_redis_client())
        except Exception:
            state = InMemoryRouterState()
    else:
        state = InMemoryRouterState()

    return ProviderSelector(
        providers=external,
        health=health,
        state=state,
        config=ProviderRouterConfig(
            purpose=purpose,
            preference_provider=preference_provider,
            preference_weight=app_settings.router_purpose_preference_weight,
            best_cache_ttl_seconds=app_settings.router_best_cache_ttl_seconds,
            wait_min_available_fraction=app_settings.router_wait_min_available_fraction,
            wait_max_seconds=app_settings.router_wait_max_seconds,
        ),
    )


def build_llm_fallback_chain(
    purpose: str,
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
    health_registry: ProviderHealthRegistry | None = None,
    purpose_provider: str | None = None,
    purpose_model: str | None = None,
) -> ProviderFallbackChain[str, str] | LLMClient:
    """Build unified LLM fallback chain for a purpose.

    Args:
        purpose: Purpose name (e.g., "answer", "rewrite", "evaluation", "enrichment", "global")
        app_settings: Application settings
        provider_rate_limiters: Pre-built rate limiters (optional)
        health_registry: Shared health registry (optional)
        purpose_provider: Optional pinned primary provider (empty = adaptive from llm_fallback_order)
        purpose_model: Optional pinned primary model (empty = resolve via priority)

    Returns:
        ProviderFallbackChain when ≥2 providers, or single LLMClient when only 1 available.
    """
    health = health_registry or _build_provider_health_registry(app_settings)
    limiters = provider_rate_limiters or _build_provider_rate_limiters(app_settings)

    config = _build_llm_chain_config(
        purpose=purpose,
        app_settings=app_settings,
        provider_rate_limiters=limiters,
        health_registry=health,
        purpose_provider=purpose_provider,
        purpose_model=purpose_model,
    )

    if not config.providers and not config.degraded_fallback:
        raise ValueError(
            f"No LLM client could be built for purpose '{purpose}'. Check API keys and LLM_FALLBACK_ORDER configuration."
        )

    chain = ProviderFallbackChain(
        config,
        health,
        router=_build_llm_router(
            config,
            health,
            app_settings,
            purpose=purpose,
            preference_provider=purpose_provider,
        ),
    )
    logger.info(
        "llm_fallback_chain_built",
        purpose=purpose,
        chain=str([(p.name, p.client.model) for p in config.providers]),
        degraded_fallback=config.degraded_fallback.name if config.degraded_fallback else None,
    )
    return chain


def build_global_llm_client(
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
    health_registry: ProviderHealthRegistry | None = None,
) -> LLMClient | ProviderFallbackChain[str, str]:
    """Build the global LLM client with adaptive fallback chain (backward-compatible wrapper).

    Uses the global ``llm_provider`` as the primary if set, otherwise builds an
    adaptive chain from ``llm_fallback_order``.
    """
    return build_llm_fallback_chain(
        purpose="global",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
        purpose_provider=app_settings.llm_provider,
        purpose_model="",
    )


def _build_embedding_chain_config(
    purpose: str,
    app_settings: AppSettings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None,
    health_registry: ProviderHealthRegistry,
) -> FallbackChainConfig:
    """Build FallbackChainConfig for embedding providers."""
    from data_engineering_copilot.infrastructure.provider_fallback import ProviderConfig

    limiters = provider_rate_limiters or {}

    # Determine fallback order from purpose-specific, catalog, or global
    if purpose == "enrichment" and app_settings.enrichment_embedding_provider:
        ordered = [app_settings.enrichment_embedding_provider.lower()]
    elif purpose == "evaluation" and app_settings.evaluation_embedding_provider:
        ordered = [app_settings.evaluation_embedding_provider.lower()]
    else:
        catalog_order = get_catalog_embedding_order(app_settings)
        if catalog_order is not None:
            ordered = catalog_order
        else:
            ordered = [p.lower() for p in app_settings.embedding_fallback_order]

    providers_config: list[ProviderConfig] = []
    for provider_name in ordered:
        try:
            client = None
            if provider_name == "nvidia":
                api_key = app_settings.nvidia_api_key.get_secret_value()
                if api_key:
                    dimension = app_settings.embedding_model_dimensions.get(
                        app_settings.nvidia_embedding_model, app_settings.default_embedding_dimension
                    )
                    batch_size = app_settings.embedding_batch_sizes.get("nvidia", app_settings.embedding_batch_size)
                    client = OpenAICompatibleEmbeddings(
                        api_key=api_key,
                        model_name=app_settings.nvidia_embedding_model,
                        base_url=app_settings.nvidia_base_url,
                        embedding_dimension=dimension,
                        batch_size=batch_size,
                        rate_limiter=limiters.get("nvidia"),
                        include_provider_param=False,
                        token_counter=token_counter_for(app_settings.nvidia_embedding_model),
                        declared_input_limit=declared_input_limit(app_settings.nvidia_embedding_model),
                    )
            elif provider_name == "openrouter":
                api_key = app_settings.openrouter_api_key.get_secret_value()
                if api_key:
                    dimension = app_settings.embedding_model_dimensions.get(
                        app_settings.openrouter_embedding_model, app_settings.default_embedding_dimension
                    )
                    batch_size = app_settings.embedding_batch_sizes.get("openrouter", app_settings.embedding_batch_size)
                    client = OpenAICompatibleEmbeddings(
                        api_key=api_key,
                        model_name=app_settings.openrouter_embedding_model,
                        base_url=app_settings.openrouter_base_url,
                        embedding_dimension=dimension,
                        batch_size=batch_size,
                        rate_limiter=limiters.get("openrouter"),
                        include_provider_param=True,
                        token_counter=token_counter_for(app_settings.openrouter_embedding_model),
                        declared_input_limit=declared_input_limit(app_settings.openrouter_embedding_model),
                    )
            elif provider_name == "gemini":
                api_key = app_settings.gemini_api_key.get_secret_value()
                if api_key:
                    dimension = app_settings.embedding_model_dimensions.get(
                        app_settings.gemini_embedding_model, app_settings.default_embedding_dimension
                    )
                    batch_size = app_settings.embedding_batch_sizes.get("gemini", app_settings.embedding_batch_size)
                    client = OpenAICompatibleEmbeddings(
                        api_key=api_key,
                        model_name=app_settings.gemini_embedding_model,
                        base_url=app_settings.gemini_base_url,
                        embedding_dimension=dimension,
                        batch_size=batch_size,
                        rate_limiter=limiters.get("gemini"),
                        include_provider_param=False,
                        token_counter=token_counter_for(app_settings.gemini_embedding_model),
                        declared_input_limit=declared_input_limit(app_settings.gemini_embedding_model),
                    )
            elif provider_name == "huggingface":
                api_key = app_settings.huggingface_api_key.get_secret_value()
                if api_key:
                    dimension = app_settings.embedding_model_dimensions.get(
                        app_settings.huggingface_embedding_model, app_settings.default_embedding_dimension
                    )
                    batch_size = app_settings.embedding_batch_sizes.get(
                        "huggingface", app_settings.embedding_batch_size
                    )
                    client = HuggingFaceServerlessEmbeddings(
                        api_key=api_key,
                        model_name=app_settings.huggingface_embedding_model,
                        base_url=app_settings.huggingface_base_url,
                        embedding_dimension=dimension,
                        batch_size=batch_size,
                        rate_limiter=limiters.get("huggingface"),
                        token_counter=token_counter_for(app_settings.huggingface_embedding_model),
                    )
            elif provider_name == "local-hf":
                client = LocalSentenceTransformerEmbeddings(
                    model_name=app_settings.local_hf_embedding_model,
                    embedding_dimension=app_settings.embedding_model_dimensions.get(
                        app_settings.local_hf_embedding_model, app_settings.default_embedding_dimension
                    ),
                    batch_size=app_settings.embedding_batch_sizes.get("local-hf", app_settings.embedding_batch_size),
                )

            if client is not None:
                providers_config.append(
                    ProviderConfig(
                        name=provider_name,
                        client=client,
                        rate_limiter=limiters.get(provider_name),
                    )
                )
                health_registry.register_provider(
                    provider_name, [getattr(client, "model_name", getattr(client, "model", "unknown"))]
                )
        except Exception as exc:
            logger.warning(
                "Skipping provider in embedding fallback chain",
                provider=provider_name,
                error=str(exc),
            )

    # Separate Ollama as degraded fallback
    main_providers = [p for p in providers_config if p.name.lower() != "ollama"]
    degraded = next((p for p in providers_config if p.name.lower() == "ollama"), None)

    return FallbackChainConfig(
        providers=main_providers,
        degraded_fallback=degraded,
        max_degraded_consecutive_failures=app_settings.ollama_degraded_max_consecutive_failures,
        error_categorizer=_categorize_embedding_error,
    )


def build_embedding_fallback_chain(
    purpose: str = "global",
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
    health_registry: ProviderHealthRegistry | None = None,
) -> ProviderFallbackChain[list[str], list[list[float]]] | EmbedderProtocol:
    """Build unified embedding fallback chain.

    Args:
        purpose: Purpose name (e.g., "global", "evaluation", "enrichment")
        app_settings: Application settings
        provider_rate_limiters: Pre-built rate limiters (optional)
        health_registry: Shared health registry (optional)

    Returns:
        ProviderFallbackChain when ≥2 providers, or single EmbedderProtocol when only 1 available.
    """
    health = health_registry or _build_provider_health_registry(app_settings)
    limiters = provider_rate_limiters or _build_provider_rate_limiters(app_settings)

    config = _build_embedding_chain_config(
        purpose=purpose,
        app_settings=app_settings,
        provider_rate_limiters=limiters,
        health_registry=health,
    )

    if not config.providers and not config.degraded_fallback:
        raise ValueError(
            f"No embedding client could be built for purpose '{purpose}'. Check API keys and EMBEDDING_FALLBACK_ORDER configuration."
        )

    if len(config.providers) + (1 if config.degraded_fallback else 0) == 1:
        if config.providers:
            client = config.providers[0].client
            name = config.providers[0].name
        else:
            assert config.degraded_fallback is not None
            client = config.degraded_fallback.client
            name = config.degraded_fallback.name
        logger.info(
            "embedding_fallback_chain_built",
            purpose=purpose,
            chain="single",
            provider=name,
        )
        return client  # type: ignore[return-value]

    chain = ProviderFallbackChain(config, health)

    if config.degraded_fallback is not None:
        chain = CooldownAwareEmbeddingRouter(
            chain=chain,
            health=health,
            max_cooldown_wait_s=app_settings.max_cooldown_wait_s,
        )

    logger.info(
        "embedding_fallback_chain_built",
        purpose=purpose,
        chain=str(
            [(p.name, getattr(p.client, "model_name", getattr(p.client, "model", "unknown"))) for p in config.providers]
        ),
        degraded_fallback=config.degraded_fallback.name if config.degraded_fallback else None,
        cooldown_aware=True,
    )
    return chain


_categorize_rerank_error = categorize_provider_error


def _build_rerank_chain_config(
    app_settings: AppSettings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None,
    health_registry: ProviderHealthRegistry,
    local_reranker=None,
) -> FallbackChainConfig:
    """Build ``FallbackChainConfig`` for cloud rerank providers.

    Cloud providers come from ``rerank_fallback_order`` (providers without an
    API key are skipped). The local ``CrossEncoderReranker`` is attached as the
    ``degraded_fallback`` — the last resort tried only after every cloud
    provider is skipped or fails.
    """
    from data_engineering_copilot.infrastructure.provider_fallback import ProviderConfig
    from data_engineering_copilot.infrastructure.rerank_clients import (
        HuggingFaceRerankClient,
        LocalRerankerClient,
        NvidiaRerankClient,
        OpenRouterRerankClient,
    )

    limiters = provider_rate_limiters or {}
    timeout = app_settings.rerank_cloud_timeout_seconds

    catalog_rerank = get_catalog_rerank_order(app_settings)
    rerank_order = catalog_rerank if catalog_rerank is not None else app_settings.rerank_fallback_order

    providers_config: list[ProviderConfig] = []
    for provider_name in rerank_order:
        provider_name = provider_name.lower()
        try:
            client = None
            if provider_name == "openrouter":
                api_key = app_settings.openrouter_api_key.get_secret_value()
                if api_key:
                    client = OpenRouterRerankClient(
                        api_key=api_key,
                        model_name=app_settings.openrouter_rerank_model,
                        base_url=app_settings.openrouter_rerank_url,
                        timeout_seconds=timeout,
                        rate_limiter=limiters.get("openrouter"),
                    )
            elif provider_name == "nvidia":
                api_key = app_settings.nvidia_api_key.get_secret_value()
                if api_key:
                    client = NvidiaRerankClient(
                        api_key=api_key,
                        model_name=app_settings.nvidia_rerank_model,
                        base_url=app_settings.nvidia_rerank_url,
                        timeout_seconds=timeout,
                        rate_limiter=limiters.get("nvidia"),
                    )
            elif provider_name == "huggingface":
                api_key = app_settings.huggingface_api_key.get_secret_value()
                if api_key:
                    client = HuggingFaceRerankClient(
                        api_key=api_key,
                        model_name=app_settings.huggingface_rerank_model,
                        base_url=app_settings.huggingface_base_url,
                        timeout_seconds=timeout,
                        rate_limiter=limiters.get("huggingface"),
                    )

            if client is not None:
                providers_config.append(
                    ProviderConfig(
                        name=provider_name,
                        client=client,
                        rate_limiter=limiters.get(provider_name),
                    )
                )
                health_registry.register_provider(provider_name, [client.model])
        except Exception as exc:
            logger.warning(
                "Skipping provider in rerank fallback chain",
                provider=provider_name,
                error=str(exc),
            )

    degraded = None
    if local_reranker is not None:
        local_client = LocalRerankerClient(local_reranker)
        degraded = ProviderConfig(
            name="local-crossencoder",
            client=local_client,
            rate_limiter=None,
        )
        health_registry.register_provider("local-crossencoder", [local_client.model])

    return FallbackChainConfig(
        providers=providers_config,
        degraded_fallback=degraded,
        max_degraded_consecutive_failures=3,
        error_categorizer=_categorize_rerank_error,
    )


def build_rerank_fallback_chain(
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
    health_registry: ProviderHealthRegistry | None = None,
    local_reranker=None,
) -> ProviderFallbackChain[RerankRequest, RerankResult] | None:
    """Build the cloud rerank fallback chain.

    Returns ``None`` when no cloud rerank provider has an API key configured
    (the caller keeps local-only reranking). Cloud providers run in
    ``rerank_fallback_order``; the local cross-encoder is the degraded
    fallback.
    """
    health = health_registry or _build_provider_health_registry(app_settings)
    limiters = provider_rate_limiters or _build_provider_rate_limiters(app_settings)

    config = _build_rerank_chain_config(
        app_settings=app_settings,
        provider_rate_limiters=limiters,
        health_registry=health,
        local_reranker=local_reranker,
    )

    if not config.providers:
        logger.info("rerank_fallback_chain_built", chain="local-only", degraded="local-crossencoder")
        return None

    chain = ProviderFallbackChain(config, health)
    logger.info(
        "rerank_fallback_chain_built",
        chain=str([(p.name, p.client.model) for p in config.providers]),
        degraded_fallback=config.degraded_fallback.name if config.degraded_fallback else None,
    )
    return chain


def build_embedder(
    app_settings: AppSettings = settings,
    rate_limiter: SlidingWindowRateLimiter | None = None,
):
    """Build embedding provider based on configured provider."""
    provider = app_settings.embedding_provider.lower()
    if provider == "openrouter":
        api_key = app_settings.openrouter_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when embedding_provider='openrouter'")
        return OpenAICompatibleEmbeddings(
            api_key=api_key,
            model_name=app_settings.openrouter_embedding_model,
            base_url=app_settings.openrouter_base_url,
            embedding_dimension=app_settings.get_embedding_dimension(),
            batch_size=app_settings.embedding_batch_size,
            rate_limiter=rate_limiter,
            include_provider_param=True,
            token_counter=token_counter_for(app_settings.openrouter_embedding_model),
            declared_input_limit=declared_input_limit(app_settings.openrouter_embedding_model),
        )
    elif provider == "nvidia":
        api_key = app_settings.nvidia_api_key.get_secret_value()
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required when embedding_provider='nvidia'")
        return OpenAICompatibleEmbeddings(
            api_key=api_key,
            model_name=app_settings.nvidia_embedding_model,
            base_url=app_settings.nvidia_base_url,
            embedding_dimension=app_settings.get_embedding_dimension(),
            batch_size=app_settings.embedding_batch_size,
            rate_limiter=rate_limiter,
            include_provider_param=False,
            token_counter=token_counter_for(app_settings.nvidia_embedding_model),
            declared_input_limit=declared_input_limit(app_settings.nvidia_embedding_model),
        )
    elif provider == "gemini":
        api_key = app_settings.gemini_api_key.get_secret_value()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when embedding_provider='gemini'")
        return OpenAICompatibleEmbeddings(
            api_key=api_key,
            model_name=app_settings.gemini_embedding_model,
            base_url=app_settings.gemini_base_url,
            embedding_dimension=app_settings.get_embedding_dimension(),
            batch_size=app_settings.embedding_batch_size,
            rate_limiter=rate_limiter,
            include_provider_param=False,
            token_counter=token_counter_for(app_settings.gemini_embedding_model),
            declared_input_limit=declared_input_limit(app_settings.gemini_embedding_model),
        )
    elif provider == "huggingface":
        api_key = app_settings.huggingface_api_key.get_secret_value()
        if not api_key:
            raise ValueError("HF_TOKEN is required when embedding_provider='huggingface'")
        return HuggingFaceServerlessEmbeddings(
            api_key=api_key,
            model_name=app_settings.huggingface_embedding_model,
            base_url=app_settings.huggingface_base_url,
            embedding_dimension=app_settings.get_embedding_dimension(),
            batch_size=app_settings.embedding_batch_size,
            token_counter=token_counter_for(app_settings.huggingface_embedding_model),
        )
    elif provider == "local-hf":
        return LocalSentenceTransformerEmbeddings(
            model_name=app_settings.local_hf_embedding_model,
            embedding_dimension=app_settings.embedding_model_dimensions.get(
                app_settings.local_hf_embedding_model, app_settings.default_embedding_dimension
            ),
            batch_size=app_settings.embedding_batch_sizes.get("local-hf", app_settings.embedding_batch_size),
        )
    elif provider == "groq":
        raise ValueError(
            "Groq does not support embeddings. Set embedding_provider to 'nvidia', 'openrouter', 'gemini', 'huggingface', or 'local-hf'."
        )
    elif provider == "cerebras":
        raise ValueError(
            "Cerebras does not support embeddings. Set embedding_provider to 'nvidia', 'openrouter', 'gemini', 'huggingface', or 'local-hf'."
        )
    else:
        raise ValueError(
            f"Unsupported embedding_provider: {provider!r}. Choose 'nvidia', 'openrouter', 'gemini', 'huggingface', or 'local-hf'."
        )


def build_evaluation_embeddings(
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
) -> list[tuple[str, EmbedderProtocol]]:
    """Build the adaptive embedding chain used by RAGAS evaluation (legacy wrapper).

    DEPRECATED: Use build_embedding_fallback_chain("evaluation", ...) instead.
    This function maintains backward compatibility by returning the old tuple format.
    """
    from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain

    chain = build_embedding_fallback_chain(
        purpose="evaluation",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
    )

    # Convert unified chain back to legacy tuple format for backward compatibility
    inner = chain.inner if isinstance(chain, CooldownAwareEmbeddingRouter) else chain
    if isinstance(inner, ProviderFallbackChain):
        return [(p.name, p.client) for p in inner._config.providers] + (  # type: ignore[return-value]
            [(inner._config.degraded_fallback.name, inner._config.degraded_fallback.client)]
            if inner._config.degraded_fallback
            else []
        )
    else:
        # Single embedder
        return [("primary", chain)]  # type: ignore[reportReturnType]


def build_chunker(app_settings: AppSettings = settings):
    from data_engineering_copilot.observability.telemetry import build_telemetry_tracer

    strategy = app_settings.chunking_strategy.lower()

    if strategy == "semantic":
        if not app_settings.enable_semantic_chunking:
            logger.warning(
                "semantic_chunking_disabled",
                strategy=strategy,
                fallback="sentence_preserving",
            )
            strategy = "sentence_preserving"
        else:
            logger.info(
                "building_semantic_chunker",
                strategy=strategy,
                similarity=app_settings.min_semantic_similarity,
            )
            from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder

            embedding_chain = build_embedding_fallback_chain(
                purpose="global",
                app_settings=app_settings,
                provider_rate_limiters=_build_provider_rate_limiters(app_settings),
                health_registry=_build_provider_health_registry(app_settings),
            )
            return SemanticChunker(
                chunk_size_words=app_settings.chunk_size_words,
                overlap_words=app_settings.chunk_overlap_words,
                embedding_model=FallbackEmbedder(embedding_chain),
                min_semantic_similarity=app_settings.min_semantic_similarity,
                min_chunk_words=int(app_settings.chunk_size_words * 0.1),
                max_chunk_words=app_settings.max_chunk_words or int(app_settings.chunk_size_words * 1.5),
                telemetry=build_telemetry_tracer(),
            )

    if strategy == "header_aware":
        logger.info(
            "building_header_aware_chunker",
            strategy=strategy,
            chunk_size=app_settings.chunk_size_words,
            overlap=app_settings.chunk_overlap_words,
        )
        return HeaderAwareChunker(
            chunk_size_words=app_settings.chunk_size_words,
            overlap_words=app_settings.chunk_overlap_words,
            min_chunk_words=int(app_settings.chunk_size_words * 0.1),
        )

    if strategy == "sentence_preserving":
        logger.info(
            "building_sentence_preserving_chunker",
            strategy=strategy,
            chunk_size=app_settings.chunk_size_words,
            overlap=app_settings.chunk_overlap_words,
        )
        return SentencePreservingChunker(max_chars=app_settings.chunk_size_words * 5)

    if strategy not in ["fixed_size"]:
        logger.warning(
            "unknown_chunking_strategy",
            strategy=strategy,
            fallback="sentence_preserving",
        )
        strategy = "sentence_preserving"
        return SentencePreservingChunker(max_chars=app_settings.chunk_size_words * 5)

    logger.info(
        "building_document_chunker",
        strategy=strategy,
        chunk_size=app_settings.chunk_size_words * 5,
        overlap=app_settings.chunk_overlap_words * 5,
    )

    return DocumentChunker(
        chunk_size_chars=app_settings.chunk_size_words * 5,
        chunk_overlap_chars=app_settings.chunk_overlap_words * 5,
    )


def build_chunker_router(app_settings: AppSettings = settings) -> ChunkerRouter:
    """Build the metadata-based chunker router with all strategy instances.

    The configured generic strategy is the fallback for unknown metadata; the
    structured, code, and guide strategies are always-available alternatives.
    No SparkChunker is wired here — the Spark path is handled by the dedicated
    Spark pipelines, mirroring the previous behavior.
    """
    from data_engineering_copilot.services.structured_data_chunker import StructuredDataChunker

    logger.info(
        "building_chunker_router",
        generic_strategy=app_settings.chunking_strategy,
        chunk_size=app_settings.chunk_size_words,
        overlap=app_settings.chunk_overlap_words,
    )

    return ChunkerRouter(
        generic_strategy=build_chunker(app_settings),
        structured_strategy=StructuredDataChunker(),
        code_strategy=DocumentChunker(
            chunk_size_chars=app_settings.chunk_size_words * 5,
            chunk_overlap_chars=app_settings.chunk_overlap_words * 5,
        ),
        guide_strategy=HeaderAwareChunker(
            chunk_size_words=app_settings.chunk_size_words,
            overlap_words=app_settings.chunk_overlap_words,
            min_chunk_words=int(app_settings.chunk_size_words * 0.1),
        ),
    )


def _validate_redis(redis_url: str, component: str) -> None:
    """Synchronous Redis connectivity check. Fails fast with clear message."""
    import redis as sync_redis

    try:
        client = sync_redis.from_url(redis_url, decode_responses=True)
        client.ping()
        client.close()
    except redis.exceptions.RedisError as exc:
        raise ConnectionError(
            f"Redis connection failed for {component}: {exc}. "
            f"Check REDIS_URL in .env (password required if Redis has requirepass)."
        ) from exc
    except ConnectionError:
        raise
    except Exception as exc:
        raise ConnectionError(
            f"Redis connection failed for {component}: {exc}. "
            f"Check REDIS_URL in .env (password required if Redis has requirepass)."
        ) from exc


def _validate_qdrant(qdrant_url: str) -> None:
    """Synchronous Qdrant health check. Fails fast with clear message."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(f"{qdrant_url}/collections")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                raise ConnectionError(f"Qdrant health check returned status {resp.status}")
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Qdrant connection failed: {exc}. Check QDRANT_URL and ensure Qdrant is running."
        ) from exc
    except ConnectionError:
        raise
    except Exception as exc:
        raise ConnectionError(
            f"Qdrant connection failed: {exc}. Check QDRANT_URL and ensure Qdrant is running."
        ) from exc


def build_async_crawler(app_settings: AppSettings = settings) -> AsyncDocumentationCrawler:
    db_url = app_settings.crawl_db_url
    if not db_url:
        raise ValueError(
            "CRAWL_DB_URL is required. Set it to a PostgreSQL connection string "
            "(e.g. postgresql://user:pass@host:5432/crawl_frontier)."
        )
    logger.info(
        "building_async_crawler",
        db_url=db_url,
        concurrency=app_settings.crawl_async_concurrency,
        max_concurrency=app_settings.crawl_async_max_concurrency,
    )
    frontier = PostgresCrawlFrontierDB(db_url)
    cache_url = app_settings.crawl_async_cache_url or app_settings.redis_url
    if app_settings.crawl_cache_enabled:
        cache = CrawlCache(cache_url, redis_client=get_shared_redis_client(app_settings.redis_url))
        _validate_redis(app_settings.redis_url, "CrawlCache")
    else:
        logger.info("crawl_cache_disabled caching=off")
        cache = NoOpCrawlCache()
    return AsyncDocumentationCrawler(
        frontier=frontier,
        cache=cache,
        timeout_seconds=app_settings.request_timeout_seconds,
        delay_seconds=app_settings.crawl_delay_seconds,
        concurrency=app_settings.crawl_async_concurrency,
        max_concurrency=app_settings.crawl_async_max_concurrency,
        thread_pool_size=app_settings.crawl_async_thread_pool_size,
        per_domain_concurrency=app_settings.crawl_async_per_domain_concurrency,
        user_agent="DataEngineeringCopilot/1.0",
    )


_RST_URL_SUFFIXES = (".rst", ".rst.txt")


def _build_content_aware_parser() -> MarkdownParser:
    _rst_parser = RstParser()
    _html_parser = MarkdownParser()

    class _ContentAwareParser(MarkdownParser):
        def parse(self, raw):
            if raw.content_type != "text/html" or any(raw.url.lower().endswith(s) for s in _RST_URL_SUFFIXES):
                try:
                    result = _rst_parser.parse(raw)
                    if result is not None:
                        return result
                except ValueError:
                    pass
            return _html_parser.parse(raw)

    return _ContentAwareParser()


def build_async_ingestion_service(app_settings: AppSettings = settings) -> AsyncIngestionService:
    from data_engineering_copilot.infrastructure.graph_store import GraphStore
    from data_engineering_copilot.observability.telemetry import build_telemetry_tracer
    from data_engineering_copilot.services.contextual_chunk_enricher import (
        ContextualChunkEnricher,
        LLMContextSummarizer,
    )
    from data_engineering_copilot.services.graph_extractor import GraphExtractor

    logger.info(
        "building_async_ingestion_service",
        sources=len(app_settings.sources),
        qdrant_url=app_settings.qdrant_url,
        collection=app_settings.collection_name,
    )
    _validate_redis(app_settings.redis_url, "IngestionService")
    _validate_qdrant(app_settings.qdrant_url)
    redis_client = get_shared_redis_client(app_settings.redis_url)

    provider_rate_limiters = _build_provider_rate_limiters(app_settings)
    health_registry = _build_provider_health_registry(app_settings)

    enrichment_client = build_llm_fallback_chain(
        purpose="enrichment",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
        purpose_provider=app_settings.enrichment_llm_provider or "ollama",
        purpose_model=app_settings.enrichment_llm_model,
    )

    # Knowledge-graph extraction: populate the shared GraphStore during ingestion
    # so the RAG service can later traverse it for topological context.
    graph_store = GraphStore()
    graph_extractor = GraphExtractor(llm_client=enrichment_client, graph_store=graph_store)

    async def _record_enrichment_failure(document) -> None:
        key = f"ingest:enrichment_failed:{document.source_name}"
        try:
            await redis_client.sadd(key, document.url)
        except Exception as exc:
            logger.warning(
                "enrichment_failure_tracking_failed",
                source=document.source_name,
                url=document.url,
                error_type=type(exc).__name__,
            )

    contextual_enricher = ContextualChunkEnricher(
        summarizer=LLMContextSummarizer(
            llm_client=enrichment_client,
            failure_recorder=_record_enrichment_failure,
            telemetry=build_telemetry_tracer(),
        ),
        enabled=app_settings.contextual_enrichment_enabled,
        batch_size=app_settings.enrichment_batch_size,
    )

    parser = _build_content_aware_parser()

    from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder

    ingestion_chain = build_embedding_fallback_chain(
        purpose="global",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
    )
    return AsyncIngestionService(
        settings=app_settings,
        crawler=build_async_crawler(app_settings),
        parser=parser,
        chunker=build_chunker(app_settings),
        embeddings=FallbackEmbedder(ingestion_chain),
        vector_store=AsyncQdrantVectorStore(
            url=app_settings.qdrant_url,
            collection_name=app_settings.collection_name,
            hybrid_search=app_settings.hybrid_search_enabled,
            hybrid_rrf_k=app_settings.hybrid_rrf_k,
            embedding_dimension=app_settings.get_embedding_dimension(),
            bm25_namespace=app_settings.namespace_bm25_enabled,
        ),
        redis_client=redis_client,  # type: ignore[arg-type]  # aioredis stubs return Awaitable not Coroutine; runtime-conformant
        contextual_enricher=contextual_enricher,
        api_extractor=ApiDocExtractor(enabled=getattr(app_settings, "api_extraction_enabled", True)),
        code_block_parser=CodeBlockParser(enabled=getattr(app_settings, "code_block_parsing_enabled", True)),
        chunk_filter=ChunkFilter(enabled=getattr(app_settings, "chunk_filtering_enabled", True)),
        telemetry=build_telemetry_tracer(),
        chunker_router=build_chunker_router(app_settings),
        graph_extractor=graph_extractor,
    )


def build_rag_service(
    app_settings: AppSettings = settings,
    token_tracker: TokenTracker | None = None,
    retrieval_tracker: RetrievalTracker | None = None,
) -> AsyncRagService:
    from data_engineering_copilot.infrastructure.graph_store import GraphStore
    from data_engineering_copilot.observability.telemetry import build_telemetry_tracer
    from data_engineering_copilot.services.context_compression import ContextCompressor
    from data_engineering_copilot.services.feedback_telemetry import FeedbackTelemetryService
    from data_engineering_copilot.services.graph_traversal import GraphTraversalService
    from data_engineering_copilot.services.groundedness import GroundednessVerifier
    from data_engineering_copilot.services.multi_hop_decomposer import MultiHopDecomposer
    from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
    from data_engineering_copilot.services.query_rewriting import QueryRewriter, default_hyde_policy
    from data_engineering_copilot.services.relevance_grader import RelevanceGrader
    from data_engineering_copilot.services.reranker import CrossEncoderReranker
    from data_engineering_copilot.services.scope_verifier import ScopeVerifier

    logger.info(
        "building_async_rag_service",
        llm_provider=app_settings.llm_provider,
        embedding_provider=app_settings.embedding_provider,
        top_k=app_settings.retrieval_top_k,
        max_context_chars=app_settings.max_context_chars,
        hybrid=app_settings.hybrid_search_enabled,
    )
    rag_config = RagConfig(
        retrieval_top_k=app_settings.retrieval_top_k,
        confidence_threshold=app_settings.confidence_threshold,
        reranker_enabled=app_settings.reranker_enabled,
        reranker_model=app_settings.reranker_model,
        reranker_top_k=app_settings.reranker_top_k,
        reranker_confidence_threshold=app_settings.reranker_confidence_threshold,
        reranker_type=app_settings.reranker_type,
        reranker_pool_size=app_settings.reranker_pool_size,
        reranker_doc_truncation_chars=app_settings.reranker_doc_truncation_chars,
        reranker_selective_threshold=app_settings.reranker_selective_threshold,
        colbert_rerank_model=app_settings.colbert_rerank_model,
        colbert_max_query_tokens=app_settings.colbert_max_query_tokens,
        colbert_max_doc_tokens=app_settings.colbert_max_doc_tokens,
        max_context_chars=app_settings.max_context_chars,
        max_chunks_per_source=app_settings.max_chunks_per_source,
        self_consistency_enabled=app_settings.self_consistency_enabled,
        self_consistency_samples=app_settings.self_consistency_samples,
        assembly_mmr_enabled=app_settings.assembly_mmr_enabled,
        assembly_mmr_lambda=app_settings.assembly_mmr_lambda,
        assembly_enable_sibling_merge=app_settings.assembly_enable_sibling_merge,
        assembly_breadcrumb_format=app_settings.assembly_breadcrumb_format,
        assembly_content_hash_dedup=app_settings.assembly_content_hash_dedup,
        prompt_salted_xml_tags=app_settings.prompt_salted_xml_tags,
        prompt_xml_content_escape=app_settings.prompt_xml_content_escape,
        prompt_trailing_instructions=app_settings.prompt_trailing_instructions,
        prompt_citation_enforcement=app_settings.prompt_citation_enforcement,
        max_expansion_queries=app_settings.max_expansion_queries,
        cache_enabled=app_settings.query_cache_enabled,
        identifier_sparse_rrf_enabled=app_settings.identifier_sparse_rrf_enabled,
        chat_cache_recall_enabled=app_settings.chat_cache_recall_enabled,
        chat_cache_top_k=app_settings.chat_cache_top_k,
        chat_cache_recall_threshold=app_settings.chat_cache_recall_threshold,
        chat_cache_max_age_seconds=app_settings.chat_cache_max_age_seconds,
        chat_suggestions_enabled=app_settings.chat_suggestions_enabled,
        chat_suggestions_count=app_settings.chat_suggestions_count,
        chat_suggestions_mode=app_settings.chat_suggestions_mode,
    )

    provider_rate_limiters = _build_provider_rate_limiters(app_settings)
    health_registry = _build_provider_health_registry(app_settings)

    llm_client = build_llm_fallback_chain(
        purpose="global",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
    )

    answer_client = build_llm_fallback_chain(
        purpose="answer",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
        purpose_provider=app_settings.answer_llm_provider,
        purpose_model=app_settings.answer_llm_model,
    )

    code_llm_client = build_llm_fallback_chain(
        purpose="code",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
        purpose_provider=app_settings.code_llm_provider,
        purpose_model=app_settings.code_llm_model,
    )

    rewrite_client = build_llm_fallback_chain(
        purpose="rewrite",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
        purpose_provider=app_settings.rewrite_llm_provider,
        purpose_model=app_settings.rewrite_llm_model,
    )
    groundedness_client = build_llm_fallback_chain(
        purpose="groundedness",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
        purpose_provider=app_settings.groundedness_llm_provider,
        purpose_model=app_settings.groundedness_llm_model,
    )
    intent_client = build_llm_fallback_chain(
        purpose="intent",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
        purpose_provider=app_settings.intent_llm_provider,
        purpose_model=app_settings.intent_llm_model,
    )
    evaluation_client = build_llm_fallback_chain(
        purpose="evaluation",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
        purpose_provider=app_settings.evaluation_llm_provider,
        purpose_model=app_settings.evaluation_llm_model,
    )

    logger.info(
        "llm_assignments",
        answer_provider=app_settings.answer_llm_provider or app_settings.llm_provider,
        answer_model=app_settings.answer_llm_model or "(from provider default)",
        code_provider=app_settings.code_llm_provider or app_settings.llm_provider,
        code_model=app_settings.code_llm_model or "(from provider default)",
        rewrite_provider=app_settings.rewrite_llm_provider or app_settings.llm_provider,
        rewrite_model=app_settings.rewrite_llm_model or "(from provider default)",
        groundedness_provider=app_settings.groundedness_llm_provider or app_settings.llm_provider,
        groundedness_model=app_settings.groundedness_llm_model or "(from provider default)",
        intent_provider=app_settings.intent_llm_provider or app_settings.llm_provider,
        intent_model=app_settings.intent_llm_model or "(from provider default)",
        evaluation_provider=app_settings.evaluation_llm_provider or app_settings.llm_provider,
        evaluation_model=app_settings.evaluation_llm_model or "(from provider default)",
    )

    vector_store = AsyncQdrantVectorStore(
        url=app_settings.qdrant_url,
        collection_name=app_settings.collection_name,
        hybrid_search=app_settings.hybrid_search_enabled,
        hybrid_rrf_k=app_settings.hybrid_rrf_k,
        embedding_dimension=app_settings.get_embedding_dimension(),
        bm25_namespace=app_settings.namespace_bm25_enabled,
        mrl_multistage_enabled=app_settings.mrl_multistage_enabled,
        mrl_small_dim=app_settings.mrl_small_dim,
        mrl_oversample_factor=app_settings.mrl_oversample_factor,
    )
    from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder as _FallbackEmbedder

    _rag_chain = build_embedding_fallback_chain(
        purpose="global",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
    )
    embedder: Any = _FallbackEmbedder(_rag_chain)  # type: ignore[assignment]
    if app_settings.embedding_cache_enabled:
        from data_engineering_copilot.infrastructure.embedding_cache import CachedEmbedder

        embedder = CachedEmbedder(
            embedder,
            redis_client=get_shared_redis_client(app_settings.redis_url),
            embedding_dimension=app_settings.get_embedding_dimension(),
        )
    reranker = None
    if app_settings.reranker_enabled:
        if app_settings.reranker_type == "colbert":
            # "colbert" is a lexical char-trigram MaxSim proxy (see
            # colbert_reranker.py), not neural late-interaction.
            from data_engineering_copilot.services.colbert_reranker import LexicalNgramReranker

            reranker = LexicalNgramReranker(
                model_name=app_settings.colbert_rerank_model,
                max_query_tokens=app_settings.colbert_max_query_tokens,
                max_doc_tokens=app_settings.colbert_max_doc_tokens,
            )
        elif app_settings.reranker_type == "pylate_colbert":
            # True neural late-interaction via PyLate; optional [colbert] extra.
            from data_engineering_copilot.services.pylate_colbert_reranker import PyLateColBERTReranker

            reranker = PyLateColBERTReranker(model_name=app_settings.pylate_colbert_model)
        else:
            from data_engineering_copilot.services.llm_reranker import LLMReranker

            local_reranker = CrossEncoderReranker(
                model_name=app_settings.reranker_model,
                doc_truncation_chars=app_settings.reranker_doc_truncation_chars,
            )
            if app_settings.llm_rerank_enabled:
                rerank_chain = build_rerank_fallback_chain(
                    app_settings=app_settings,
                    provider_rate_limiters=provider_rate_limiters,
                    health_registry=health_registry,
                    local_reranker=local_reranker,
                )
                reranker = LLMReranker(chain=rerank_chain, local=local_reranker)
            else:
                reranker = local_reranker

    telemetry = build_telemetry_tracer()
    if token_tracker is None:
        token_tracker = TokenTracker()
    if retrieval_tracker is None:
        retrieval_tracker = RetrievalTracker()

    # Phase 2 modules — each gets its purpose-specific client or falls back to global
    query_rewriter = QueryRewriter(
        llm_client=rewrite_client or llm_client,
        enabled=app_settings.query_rewrite_enabled,
        hyde_policy=default_hyde_policy() if app_settings.hyde_policy_enabled else None,
        intent_llm_enabled=app_settings.intent_classification_llm_enabled,
        intent_llm_client=intent_client,
    )
    groundedness = GroundednessVerifier(
        llm_client=groundedness_client or llm_client,
        enabled=app_settings.groundedness_enabled,
        groundedness_threshold=app_settings.groundedness_threshold,
    )
    # Topic-scope gate rides the strong answer-purpose chain (not the cheap
    # budget chain): the gate is a nuanced topical judgment and needs the most
    # capable available model, not a cost-optimized verifier.
    scope_verifier = ScopeVerifier(
        llm_client=answer_client or llm_client,
        enabled=app_settings.scope_check_enabled,
    )
    context_compressor = ContextCompressor(
        enabled=app_settings.context_compression_enabled,
        max_chunks=app_settings.retrieval_top_k,
        compression_ratio=app_settings.context_compression_ratio,
    )

    # PII redaction
    pii_redactor = None
    if app_settings.pii_redaction_enabled:
        from data_engineering_copilot.infrastructure.pii_redactor import PiiRedactor, RedactionMode

        pii_redactor = PiiRedactor(mode=RedactionMode(app_settings.pii_redaction_mode))

    # Indirect prompt injection guard for retrieved documents
    from data_engineering_copilot.services.input_guardrails import InputGuardrails

    input_guardrails = InputGuardrails(enabled=app_settings.input_guardrails_enabled)

    # GraphRAG / Multi-hop / CRAG / Feedback modules (Plan 1). A single shared
    # GraphStore is used so graph context at query time reflects the entities
    # extracted during ingestion (both processes default to data/graph_store.db).
    graph_store = GraphStore()
    graph_traversal_service = GraphTraversalService(llm_client=answer_client or llm_client, graph_store=graph_store)
    multi_hop_decomposer = MultiHopDecomposer(llm_client=answer_client or llm_client)
    relevance_grader = RelevanceGrader(llm_client=answer_client or llm_client)
    feedback_telemetry = FeedbackTelemetryService()

    # Phase 6 (Task 6.3): low-confidence answers → review dataset (fail-open).
    from data_engineering_copilot.evaluation.langfuse_datasets import create_review_item

    return AsyncRagService(
        config=rag_config,
        vector_store=vector_store,
        llm_client=answer_client or llm_client,
        code_llm_client=code_llm_client,
        evaluation_llm_client=cast("LLMClientProtocol | None", evaluation_client),
        embedder=embedder,
        reranker=reranker,
        telemetry=telemetry,
        cache=TwoTierCache(
            exact_enabled=app_settings.query_cache_exact_enabled,
            semantic_enabled=app_settings.query_cache_semantic_enabled,
            similarity_threshold=app_settings.semantic_cache_threshold,
            ttl_seconds=app_settings.semantic_cache_ttl,
            redis_url=app_settings.redis_url,
            redis_client=get_shared_redis_client(app_settings.redis_url),
        ),
        query_rewriter=query_rewriter,
        groundedness_verifier=groundedness,
        scope_verifier=scope_verifier,
        context_compressor=context_compressor,
        token_tracker=token_tracker,
        retrieval_tracker=retrieval_tracker,
        pii_redactor=pii_redactor,
        input_guardrails=input_guardrails,
        multi_hop_decomposer=multi_hop_decomposer,
        graph_traversal_service=graph_traversal_service,
        relevance_grader=relevance_grader,
        feedback_telemetry_service=feedback_telemetry,
        review_dataset_hook=create_review_item,
    )


def build_pipeline_lab(app_settings: AppSettings = settings, *, dry_run: bool = True) -> PipelineLab:
    """Build a :class:`PipelineLab` wired to the production ingestion pieces.

    The enrichment step uses the same contextual-enricher construction as
    ``build_async_ingestion_service`` but degrades to a no-op if the LLM chain
    cannot be built (offline). Embedding and the Qdrant store are always built;
    ``dry_run`` keeps the run read-only (payload preview, no upsert).
    """
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.observability.telemetry import build_telemetry_tracer
    from data_engineering_copilot.services.contextual_chunk_enricher import (
        ContextualChunkEnricher,
        LLMContextSummarizer,
    )
    from data_engineering_copilot.services.pipeline_lab import PipelineLab

    enricher: ContextualChunkEnricher | None = None
    if getattr(app_settings, "contextual_enrichment_enabled", True):
        try:
            provider_rate_limiters = _build_provider_rate_limiters(app_settings)
            health_registry = _build_provider_health_registry(app_settings)
            enrichment_client = build_llm_fallback_chain(
                purpose="enrichment",
                app_settings=app_settings,
                provider_rate_limiters=provider_rate_limiters,
                health_registry=health_registry,
                purpose_provider=app_settings.enrichment_llm_provider or "ollama",
                purpose_model=app_settings.enrichment_llm_model,
            )
            enricher = ContextualChunkEnricher(
                summarizer=LLMContextSummarizer(
                    llm_client=enrichment_client,
                    failure_recorder=None,
                    telemetry=build_telemetry_tracer(),
                ),
                enabled=app_settings.contextual_enrichment_enabled,
                batch_size=app_settings.enrichment_batch_size,
            )
        except Exception as exc:  # noqa: BLE001 - lab degrades to no enrichment
            logger.warning("pipeline_lab_enricher_unavailable", error=repr(exc))
            enricher = None

    embedder = None
    try:
        from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder as _FallbackEmbedderLab

        _lab_rate = _build_provider_rate_limiters(app_settings)
        _lab_health = _build_provider_health_registry(app_settings)
        _lab_chain = build_embedding_fallback_chain(
            purpose="global",
            app_settings=app_settings,
            provider_rate_limiters=_lab_rate,
            health_registry=_lab_health,
        )
        embedder = _FallbackEmbedderLab(_lab_chain)
    except Exception as exc:  # noqa: BLE001 - lab degrades to no embedding step
        logger.warning("pipeline_lab_embedder_unavailable", error=repr(exc))
        embedder = None

    return PipelineLab(
        parser=_build_content_aware_parser(),
        chunk_filter=ChunkFilter(enabled=getattr(app_settings, "chunk_filtering_enabled", True)),
        chunker=build_chunker(app_settings),
        api_extractor=ApiDocExtractor(enabled=getattr(app_settings, "api_extraction_enabled", True)),
        enricher=enricher,
        embedder=embedder,
        vector_store=AsyncQdrantVectorStore(
            url=app_settings.qdrant_url,
            collection_name=app_settings.collection_name,
            hybrid_search=app_settings.hybrid_search_enabled,
            hybrid_rrf_k=app_settings.hybrid_rrf_k,
            embedding_dimension=app_settings.get_embedding_dimension(),
            bm25_namespace=app_settings.namespace_bm25_enabled,
        ),
        dry_run=dry_run,
    )


def build_conversation_service(app_settings: AppSettings = settings) -> ConversationService:
    """Build a :class:`ConversationService` wired to the RAG service + chat store.

    Uses the shared Redis client and the existing ``build_rag_service``
    singletons so chat turns reuse the same vector store/embedder/LLM chains.
    The Postgres pool is created lazily by ``ChatSessionPostgresStore`` (its
    DSN defaults to ``crawl_db_url`` when ``chat_db_url`` is unset).
    """
    from data_engineering_copilot.infrastructure.chat_session_store import (
        ChatSessionPostgresStore,
        ChatSessionRedisStore,
        ChatSessionStore,
    )
    from data_engineering_copilot.services.conversation_rag import ConversationService

    if not app_settings.chat_enabled:
        raise RuntimeError("Conversational chat is disabled (chat_enabled=false)")

    dsn = app_settings.chat_db_url or app_settings.crawl_db_url
    pg_store = ChatSessionPostgresStore(dsn)
    redis_store = ChatSessionRedisStore(
        get_shared_redis_client(app_settings.redis_url),
        ttl_seconds=app_settings.chat_session_ttl_seconds,
    )
    store = ChatSessionStore(redis_store, pg_store)

    from data_engineering_copilot.services.rag_service_singleton import get_rag_service_if_initialized

    rag_service = get_rag_service_if_initialized()
    if rag_service is None:
        rag_service = build_rag_service(app_settings=app_settings)

    # Phase C: build the local Ollama components used for the cheap short steps
    # (rewrite/hyde/expand/scope-verify). Only when the settings opt in.
    local_rewriter = None
    local_scope = None
    local_llm_client = None
    if app_settings.chat_rewrite_local or app_settings.chat_scope_local or app_settings.chat_answer_local:
        local_llm_client = LLMClient(
            base_url=f"{app_settings.ollama_base_url}/v1",
            model=app_settings.ollama_model,
            api_key="",
            timeout_seconds=app_settings.ollama_timeout_seconds,
            max_tokens=app_settings.ollama_num_predict,
            connect_timeout_seconds=app_settings.ollama_connect_timeout_seconds,
            pool_timeout_seconds=app_settings.ollama_pool_timeout_seconds,
        )
        if app_settings.chat_rewrite_local:
            from data_engineering_copilot.services.query_rewriting import QueryRewriter, default_hyde_policy

            local_rewriter = QueryRewriter(
                llm_client=local_llm_client,
                enabled=app_settings.query_rewrite_enabled,
                hyde_policy=default_hyde_policy() if app_settings.hyde_policy_enabled else None,
            )
        if app_settings.chat_scope_local:
            from data_engineering_copilot.services.scope_verifier import ScopeVerifier

            local_scope = ScopeVerifier(
                llm_client=local_llm_client,
                enabled=app_settings.scope_check_enabled,
            )

    # Chat reranking: prefer the local cross-encoder over the cloud LLM rerank
    # chain (which costs ~5s/turn). Off unless both reranking is enabled and the
    # local path is opted in.
    local_reranker = None
    if app_settings.chat_rerank_local and app_settings.reranker_enabled:
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        local_reranker = CrossEncoderReranker(
            model_name=app_settings.reranker_model,
            doc_truncation_chars=app_settings.reranker_doc_truncation_chars,
        )

    return ConversationService(
        rag_service=rag_service,
        store=store,
        title_max_chars=app_settings.chat_title_max_chars,
        local_query_rewriter=local_rewriter,
        local_scope_verifier=local_scope,
        local_llm_client=local_llm_client,
        answer_local=app_settings.chat_answer_local,
        local_reranker=local_reranker,
        blocked_url_substrings=app_settings.chat_blocked_url_substrings,
        domain_sources=app_settings.chat_domain_sources,
    )
