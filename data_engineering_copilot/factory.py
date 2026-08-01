from __future__ import annotations

from typing import cast

import redis.asyncio as aioredis
import redis.exceptions

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.domain.models import RagConfig
from data_engineering_copilot.domain.protocols import LLMClientProtocol
from data_engineering_copilot.infrastructure.adaptive_llm_router import AdaptiveLLMRouter
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import OpenAICompatibleEmbeddings
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache
from data_engineering_copilot.infrastructure.crawl_db import PostgresCrawlFrontierDB
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.infrastructure.llm_client import LLMClient
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter
from data_engineering_copilot.infrastructure.rst_parser import RstParser
from data_engineering_copilot.observability.structured_logging import StructuredLogger
from data_engineering_copilot.observability.token_tracker import RetrievalTracker, TokenTracker
from data_engineering_copilot.services.api_extractor import ApiDocExtractor
from data_engineering_copilot.services.async_ingestion import AsyncIngestionService
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.code_block_parser import CodeBlockParser
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.semantic_chunker import SemanticChunker
from data_engineering_copilot.services.text_filter import ChunkFilter

logger = StructuredLogger(__name__)

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
    for p in app_settings.llm_fallback_order:
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
    return rate_limiters


def _build_provider_health_registry(app_settings: AppSettings = settings) -> ProviderHealthRegistry:
    return ProviderHealthRegistry(
        success_rate_weight=app_settings.health_success_rate_weight,
        latency_weight=app_settings.health_latency_weight,
        recency_weight=app_settings.health_recency_weight,
        consecutive_failure_penalty=app_settings.health_consecutive_failure_penalty,
        default_cooldown_seconds=app_settings.provider_cooldown_seconds,
    )


def _build_purpose_llm_client(
    provider: str,
    model: str,
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
    timeout_seconds: int | None = None,
    purpose: str | None = None,
) -> LLMClient | None:
    """Build an LLM client for a specific purpose.

    When both *provider* and *model* are empty/blank the caller intends to
    fall back to the global ``llm_provider`` / ``llm_model``.  Returns
    ``None`` in that case so the factory can reuse a shared global client.

    Model resolution priority:
        1. *model* — explicit purpose-level override
        2. ``{provider}_{purpose}_llm_model`` — per-provider purpose override
        3. ``{provider}_model`` — provider default model
        4. ``llm_model`` — global default model
    """
    eff_provider = (provider or app_settings.llm_provider).lower()

    # Model resolution: explicit > per-provider purpose > provider default > global
    eff_model = model or ""
    if not eff_model and purpose:
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

    rate_limiter = (provider_rate_limiters or {}).get(eff_provider)

    if eff_provider == "ollama":
        llm_base = app_settings.llm_ollama_base_url or app_settings.ollama_base_url
        return LLMClient(
            base_url=f"{llm_base}/v1",
            model=eff_model,
            api_key="",
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            keep_alive=app_settings.ollama_keep_alive,
            connect_timeout_seconds=app_settings.ollama_connect_timeout_seconds,
            pool_timeout_seconds=app_settings.ollama_pool_timeout_seconds,
            extra_body={
                "options": {
                    "num_ctx": app_settings.ollama_num_ctx,
                    "num_predict": app_settings.ollama_num_predict,
                }
            },
        )

    if eff_provider == "openrouter":
        api_key = app_settings.openrouter_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when provider='openrouter'")
        return LLMClient(
            base_url=app_settings.openrouter_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            extra_headers={"HTTP-Referer": "https://data-engineering-copilot.local"},
            rate_limiter=rate_limiter,
        )

    if eff_provider == "nvidia":
        api_key = app_settings.nvidia_api_key.get_secret_value()
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required when provider='nvidia'")
        return LLMClient(
            base_url=app_settings.nvidia_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            rate_limiter=rate_limiter,
        )

    if eff_provider == "groq":
        api_key = app_settings.groq_api_key.get_secret_value()
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when provider='groq'")
        return LLMClient(
            base_url=app_settings.groq_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            rate_limiter=rate_limiter,
        )

    if eff_provider == "cerebras":
        api_key = app_settings.cerebras_api_key.get_secret_value()
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY is required when provider='cerebras'")
        return LLMClient(
            base_url=app_settings.cerebras_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            rate_limiter=rate_limiter,
        )

    if eff_provider == "gemini":
        api_key = app_settings.gemini_api_key.get_secret_value()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when provider='gemini'")
        return LLMClient(
            base_url=app_settings.gemini_base_url,
            model=eff_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds or app_settings.ollama_timeout_seconds,
            rate_limiter=rate_limiter,
        )

    raise ValueError(
        f"Unsupported LLM provider: {eff_provider!r}. Supported: 'ollama', 'openrouter', 'nvidia', 'groq', 'cerebras', 'gemini'."
    )


def _build_fallback_chain(
    purpose_provider: str,
    purpose_model: str,
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
    purpose: str | None = None,
    health_registry: ProviderHealthRegistry | None = None,
) -> AdaptiveLLMRouter | LLMClient | None:
    """Build a health-aware adaptive fallback chain of LLM providers.

    The *purpose_provider* (or the global ``llm_provider`` if empty) is
    tried first.  If it fails, remaining providers from
    ``llm_fallback_order`` are tried in sequence, with Ollama as the last
    resort.

    Returns ``None`` when *purpose_provider* is empty (caller intends to
    reuse the global chain).  Returns a bare ``LLMClient`` when only one
    provider is available.  Returns ``AdaptiveLLMRouter`` when ≥2 providers
    are configured.
    """
    eff_provider = (purpose_provider or "").strip().lower()
    if not eff_provider:
        return None

    fallback_order = [p.lower() for p in app_settings.llm_fallback_order]

    # Deduplicated ordered list: primary first, then remaining from fallback order
    ordered: list[str] = [eff_provider]
    for p in fallback_order:
        if p not in ordered:
            ordered.append(p)

    clients: list[tuple[str, LLMClient]] = []
    client_timeout = app_settings.llm_fallback_call_timeout
    health = health_registry or _build_provider_health_registry(app_settings)

    for idx, provider in enumerate(ordered):
        if provider == "nvidia":
            logger.warning(
                "Skipping nvidia in LLM fallback chain — reserved for embeddings only. "
                "Set code_llm_provider / enrichment_llm_provider to another provider.",
            )
            continue
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
            )
            if client is not None:
                clients.append((provider, client))
                health.register_provider(provider, [client.model])
        except Exception as exc:
            logger.warning(
                "Skipping provider in fallback chain",
                provider=provider,
                error=str(exc),
            )

    if not clients:
        raise ValueError(
            f"No LLM client could be built for provider {eff_provider!r}. Check API keys and provider configuration."
        )

    chain_info = [(p, c.model) for p, c in clients]
    logger.info(
        "fallback_chain_built",
        purpose=purpose or "global",
        primary=eff_provider,
        chain=str(chain_info),
    )
    if len(clients) == 1:
        return clients[0][1]
    return AdaptiveLLMRouter(
        clients=clients,
        health=health,
        rate_limiters=provider_rate_limiters,
        ollama_max_consecutive_failures=app_settings.ollama_degraded_max_consecutive_failures,
    )


def build_global_llm_client(
    app_settings: AppSettings = settings,
    provider_rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
    health_registry: ProviderHealthRegistry | None = None,
) -> LLMClient | AdaptiveLLMRouter:
    """Build the global LLM client with adaptive fallback chain.

    The global ``llm_model`` is NOT passed as ``purpose_model`` here — doing so
    would short-circuit model resolution at priority 1 (explicit override) and
    force that model onto whatever provider is primary, even if the provider
    has its own default model (e.g. ``openrouter_model = "openrouter/free"``).
    Instead, ``llm_model`` stays as the fallback at priority 4 in
    ``_build_purpose_llm_client``.
    """
    client = _build_fallback_chain(
        purpose_provider=app_settings.llm_provider,
        purpose_model="",
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        health_registry=health_registry,
    )
    if client is None:
        raise ValueError("Global llm_provider and llm_model must be set")
    return client


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
        )
    elif provider in ("ollama", "local"):
        embed_base = app_settings.embedding_ollama_base_url or app_settings.ollama_base_url
        return AsyncOllamaEmbeddings(
            model_name=app_settings.embedding_model_name,
            base_url=embed_base,
            batch_size=app_settings.embedding_batch_size,
            timeout_seconds=app_settings.ollama_timeout_seconds,
            max_concurrency=app_settings.embed_concurrency,
            keep_alive=app_settings.ollama_keep_alive,
            connect_timeout_seconds=app_settings.ollama_connect_timeout_seconds,
            pool_timeout_seconds=app_settings.ollama_pool_timeout_seconds,
        )
    elif provider == "groq":
        raise ValueError(
            "Groq does not support embeddings. Set embedding_provider to 'ollama', 'openrouter', 'nvidia', or 'gemini'."
        )
    elif provider == "cerebras":
        raise ValueError(
            "Cerebras does not support embeddings. Set embedding_provider to 'ollama', 'openrouter', 'nvidia', or 'gemini'."
        )
    else:
        raise ValueError(
            f"Unsupported embedding_provider: {provider!r}. Choose 'ollama', 'openrouter', 'nvidia', 'gemini'."
        )


def build_chunker(app_settings: AppSettings = settings):
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
            return SemanticChunker(
                chunk_size_words=app_settings.chunk_size_words,
                overlap_words=app_settings.chunk_overlap_words,
                embedding_model=build_embedder(app_settings),
                min_semantic_similarity=app_settings.min_semantic_similarity,
                min_chunk_words=int(app_settings.chunk_size_words * 0.1),
                max_chunk_words=app_settings.max_chunk_words or int(app_settings.chunk_size_words * 1.5),
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

    if strategy not in ["fixed_size", "sentence_preserving"]:
        logger.warning(
            "unknown_chunking_strategy",
            strategy=strategy,
            fallback="sentence_preserving",
        )
        strategy = "sentence_preserving"

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
    cache = CrawlCache(cache_url, redis_client=get_shared_redis_client(app_settings.redis_url))
    _validate_redis(app_settings.redis_url, "CrawlCache")
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
    from data_engineering_copilot.services.contextual_chunk_enricher import (
        ContextualChunkEnricher,
        LLMContextSummarizer,
    )

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

    enrichment_client = _build_fallback_chain(
        purpose_provider=app_settings.enrichment_llm_provider or "ollama",
        purpose_model=app_settings.enrichment_llm_model,
        app_settings=app_settings,
        health_registry=health_registry,
        provider_rate_limiters=provider_rate_limiters,
        purpose="enrichment",
    )

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
        summarizer=LLMContextSummarizer(llm_client=enrichment_client, failure_recorder=_record_enrichment_failure),
        enabled=app_settings.contextual_enrichment_enabled,
        batch_size=app_settings.enrichment_batch_size,
    )

    parser = _build_content_aware_parser()

    return AsyncIngestionService(
        settings=app_settings,
        crawler=build_async_crawler(app_settings),
        parser=parser,
        chunker=build_chunker(app_settings),
        embeddings=build_embedder(app_settings, provider_rate_limiters.get(app_settings.embedding_provider.lower())),
        vector_store=AsyncQdrantVectorStore(
            url=app_settings.qdrant_url,
            collection_name=app_settings.collection_name,
            hybrid_search=app_settings.hybrid_search_enabled,
            hybrid_rrf_k=app_settings.hybrid_rrf_k,
            embedding_dimension=app_settings.get_embedding_dimension(),
        ),
        redis_client=redis_client,
        contextual_enricher=contextual_enricher,
        api_extractor=ApiDocExtractor(enabled=getattr(app_settings, "api_extraction_enabled", True)),
        code_block_parser=CodeBlockParser(enabled=getattr(app_settings, "code_block_parsing_enabled", True)),
        chunk_filter=ChunkFilter(enabled=getattr(app_settings, "chunk_filtering_enabled", True)),
    )


def build_rag_service(
    app_settings: AppSettings = settings,
    token_tracker: TokenTracker | None = None,
    retrieval_tracker: RetrievalTracker | None = None,
) -> AsyncRagService:
    from data_engineering_copilot.observability.telemetry import build_telemetry_tracer
    from data_engineering_copilot.services.context_compression import ContextCompressor
    from data_engineering_copilot.services.groundedness import GroundednessVerifier
    from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
    from data_engineering_copilot.services.query_rewriting import QueryRewriter
    from data_engineering_copilot.services.reranker import CrossEncoderReranker

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
        max_context_chars=app_settings.max_context_chars,
    )

    provider_rate_limiters = _build_provider_rate_limiters(app_settings)
    health_registry = _build_provider_health_registry(app_settings)

    llm_client = build_global_llm_client(app_settings, provider_rate_limiters, health_registry)

    answer_client = _build_fallback_chain(
        purpose_provider=app_settings.answer_llm_provider,
        purpose_model=app_settings.answer_llm_model,
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        purpose="answer",
        health_registry=health_registry,
    )

    code_llm_client = _build_fallback_chain(
        purpose_provider=app_settings.code_llm_provider,
        purpose_model=app_settings.code_llm_model,
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        purpose="code",
        health_registry=health_registry,
    )

    rewrite_client = _build_fallback_chain(
        purpose_provider=app_settings.rewrite_llm_provider,
        purpose_model=app_settings.rewrite_llm_model,
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        purpose="rewrite",
        health_registry=health_registry,
    )
    groundedness_client = _build_fallback_chain(
        purpose_provider=app_settings.groundedness_llm_provider,
        purpose_model=app_settings.groundedness_llm_model,
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        purpose="groundedness",
        health_registry=health_registry,
    )
    intent_client = _build_fallback_chain(
        purpose_provider=app_settings.intent_llm_provider,
        purpose_model=app_settings.intent_llm_model,
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        purpose="intent",
        health_registry=health_registry,
    )
    evaluation_client = _build_fallback_chain(
        purpose_provider=app_settings.evaluation_llm_provider,
        purpose_model=app_settings.evaluation_llm_model,
        app_settings=app_settings,
        provider_rate_limiters=provider_rate_limiters,
        purpose="evaluation",
        health_registry=health_registry,
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
    )
    embedder = build_embedder(app_settings, provider_rate_limiters.get(app_settings.embedding_provider.lower()))
    from data_engineering_copilot.infrastructure.embedding_cache import CachedEmbedder

    embedder = CachedEmbedder(embedder)
    reranker = None
    if app_settings.reranker_enabled:
        reranker = CrossEncoderReranker(model_name=app_settings.reranker_model)

    telemetry = build_telemetry_tracer()
    if token_tracker is None:
        token_tracker = TokenTracker()
    if retrieval_tracker is None:
        retrieval_tracker = RetrievalTracker()

    # Phase 2 modules — each gets its purpose-specific client or falls back to global
    query_rewriter = QueryRewriter(
        llm_client=rewrite_client or llm_client,
        enabled=app_settings.query_rewrite_enabled,
        intent_llm_enabled=app_settings.intent_classification_llm_enabled,
        intent_llm_client=intent_client,
    )
    groundedness = GroundednessVerifier(
        llm_client=groundedness_client or llm_client,
        enabled=app_settings.groundedness_enabled,
    )
    context_compressor = ContextCompressor(
        enabled=app_settings.context_compression_enabled,
        max_chunks=app_settings.retrieval_top_k,
    )

    # PII redaction
    pii_redactor = None
    if app_settings.pii_redaction_enabled:
        from data_engineering_copilot.infrastructure.pii_redactor import PiiRedactor, RedactionMode

        pii_redactor = PiiRedactor(mode=RedactionMode(app_settings.pii_redaction_mode))

    # Indirect prompt injection guard for retrieved documents
    from data_engineering_copilot.services.input_guardrails import InputGuardrails

    input_guardrails = InputGuardrails(enabled=app_settings.input_guardrails_enabled)

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
            exact_enabled=True,
            semantic_enabled=True,
            similarity_threshold=app_settings.semantic_cache_threshold,
            ttl_seconds=app_settings.semantic_cache_ttl,
            redis_url=app_settings.redis_url,
            redis_client=get_shared_redis_client(app_settings.redis_url),
        ),
        query_rewriter=query_rewriter,
        groundedness_verifier=groundedness,
        context_compressor=context_compressor,
        token_tracker=token_tracker,
        retrieval_tracker=retrieval_tracker,
        pii_redactor=pii_redactor,
        input_guardrails=input_guardrails,
    )
