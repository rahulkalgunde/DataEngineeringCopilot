from __future__ import annotations

import redis.asyncio as aioredis
import redis.exceptions

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.domain.models import RagConfig
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient
from data_engineering_copilot.infrastructure.async_openai_compatible_client import OpenAICompatibleLLMClient
from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import OpenAICompatibleEmbeddings
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache
from data_engineering_copilot.infrastructure.crawl_db import CrawlFrontierDB, PostgresCrawlFrontierDB
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter
from data_engineering_copilot.observability.structured_logging import StructuredLogger
from data_engineering_copilot.observability.token_tracker import RetrievalTracker, TokenTracker
from data_engineering_copilot.services.api_extractor import ApiDocExtractor
from data_engineering_copilot.services.async_ingestion import AsyncIngestionService
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.code_block_parser import CodeBlockParser
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.semantic_chunker import SemanticChunker

logger = StructuredLogger(__name__)


def build_rate_limiter(app_settings: AppSettings = settings) -> SlidingWindowRateLimiter | None:
    """Build a shared rate limiter when OpenRouter is the active provider for either LLM or embeddings."""
    if app_settings.llm_provider.lower() != "openrouter" and app_settings.embedding_provider.lower() != "openrouter":
        return None
    return SlidingWindowRateLimiter(
        rpm_limit=app_settings.openrouter_rpm_limit,
        rpd_limit=app_settings.openrouter_rpd_limit,
    )


def build_llm_client(
    app_settings: AppSettings = settings,
    rate_limiter: SlidingWindowRateLimiter | None = None,
):
    """Build LLM client based on configured provider."""
    provider = app_settings.llm_provider.lower()
    if provider == "openrouter":
        api_key = app_settings.openrouter_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when llm_provider='openrouter'")
        return OpenAICompatibleLLMClient(
            api_key=api_key,
            model=app_settings.openrouter_model,
            timeout_seconds=app_settings.ollama_timeout_seconds,
            rate_limiter=rate_limiter,
        )
    elif provider == "ollama":
        return AsyncOllamaClient(
            base_url=app_settings.ollama_base_url,
            model=app_settings.ollama_model,
            timeout_seconds=app_settings.ollama_timeout_seconds,
            num_ctx=app_settings.ollama_num_ctx,
            num_predict=app_settings.ollama_num_predict,
        )

    else:
        raise ValueError(f"Unsupported llm_provider: {provider!r}. Choose 'ollama' or 'openrouter'.")


def build_code_llm_client(
    app_settings: AppSettings = settings,
    shared_rate_limiter: SlidingWindowRateLimiter | None = None,
):
    """Build optional code-specific LLM client. Returns None if not configured.

    When the code provider differs from the primary provider (e.g. primary=ollama,
    code=nvidia), a separate rate limiter is created. When they match (e.g. both
    openrouter), the shared rate limiter is reused.
    """
    provider = app_settings.code_llm_provider.lower()
    if not provider:
        return None

    if provider == "nvidia":
        api_key = app_settings.nvidia_nim_api_key.get_secret_value()
        if not api_key:
            raise ValueError("NVIDIA_NIM_API_KEY is required when code_llm_provider='nvidia'")
        nvidia_limiter = SlidingWindowRateLimiter(
            rpm_limit=app_settings.nvidia_nim_rpm_limit,
            rpd_limit=app_settings.nvidia_nim_rpd_limit,
        )
        return OpenAICompatibleLLMClient(
            api_key=api_key,
            model=app_settings.code_llm_model,
            base_url=app_settings.nvidia_nim_base_url,
            timeout_seconds=app_settings.ollama_timeout_seconds,
            rate_limiter=nvidia_limiter,
        )

    if provider in ("ollama", "openrouter"):
        return build_llm_client(app_settings, shared_rate_limiter)

    raise ValueError(f"Unsupported code_llm_provider: {provider!r}. Choose 'ollama', 'openrouter', or 'nvidia'.")


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
            embedding_dimension=app_settings.openrouter_embedding_dimension,
            batch_size=app_settings.embedding_batch_size,
            rate_limiter=rate_limiter,
        )
    elif provider == "nvidia":
        api_key = app_settings.nvidia_nim_api_key.get_secret_value()
        if not api_key:
            raise ValueError("NVIDIA_NIM_API_KEY is required when embedding_provider='nvidia'")
        nvidia_embedding_limiter = SlidingWindowRateLimiter(
            rpm_limit=app_settings.nvidia_nim_rpm_limit,
            rpd_limit=app_settings.nvidia_nim_rpd_limit,
        )
        return OpenAICompatibleEmbeddings(
            api_key=api_key,
            model_name=app_settings.nvidia_embedding_model,
            base_url=app_settings.nvidia_nim_base_url,
            embedding_dimension=app_settings.nvidia_embedding_dimension,
            batch_size=app_settings.embedding_batch_size,
            rate_limiter=nvidia_embedding_limiter,
            include_provider_param=False,
        )
    elif provider in ("ollama", "local"):
        return AsyncOllamaEmbeddings(model_name=app_settings.embedding_model_name)
    else:
        raise ValueError(f"Unsupported embedding_provider: {provider!r}. Choose 'ollama', 'openrouter', 'nvidia'.")


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
        chunk_size=app_settings.chunk_size_words * 5,
        chunk_overlap=app_settings.chunk_overlap_words * 5,
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
    if db_url:
        logger.info(
            "building_async_crawler",
            db_url=db_url,
            concurrency=app_settings.crawl_async_concurrency,
            max_concurrency=app_settings.crawl_async_max_concurrency,
        )
        frontier = PostgresCrawlFrontierDB(db_url)
    else:
        logger.info(
            "building_async_crawler",
            db=str(app_settings.crawl_db_path),
            concurrency=app_settings.crawl_async_concurrency,
            max_concurrency=app_settings.crawl_async_max_concurrency,
        )
        db_path = str(app_settings.crawl_db_path)
        frontier = CrawlFrontierDB(db_path)
    cache_url = app_settings.crawl_async_cache_url or app_settings.redis_url
    cache = CrawlCache(cache_url)
    _validate_redis(app_settings.redis_url, "CrawlCache")
    return AsyncDocumentationCrawler(
        frontier=frontier,
        cache=cache,
        timeout_seconds=app_settings.request_timeout_seconds,
        delay_seconds=app_settings.crawl_delay_seconds,
        concurrency=app_settings.crawl_async_concurrency,
        max_concurrency=app_settings.crawl_async_max_concurrency,
        conditional_get=app_settings.crawl_async_conditional_get,
        thread_pool_size=app_settings.crawl_async_thread_pool_size,
        per_domain_concurrency=app_settings.crawl_async_per_domain_concurrency,
        user_agent="DataEngineeringCopilot/1.0",
    )


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
    redis_client = aioredis.from_url(
        app_settings.redis_url,
        decode_responses=True,
        max_connections=20,
    )

    rate_limiter = build_rate_limiter(app_settings)

    contextual_enricher = ContextualChunkEnricher(
        summarizer=LLMContextSummarizer(llm_client=build_llm_client(app_settings, rate_limiter)),
        enabled=app_settings.contextual_enrichment_enabled,
        batch_size=app_settings.enrichment_batch_size,
    )

    return AsyncIngestionService(
        settings=app_settings,
        crawler=build_async_crawler(app_settings),
        parser=MarkdownParser(),
        chunker=build_chunker(app_settings),
        embeddings=build_embedder(app_settings, rate_limiter),
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
    rate_limiter = build_rate_limiter(app_settings)
    llm_client = build_llm_client(app_settings, rate_limiter)
    code_llm_client = build_code_llm_client(app_settings, shared_rate_limiter=rate_limiter)
    vector_store = AsyncQdrantVectorStore(
        url=app_settings.qdrant_url,
        collection_name=app_settings.collection_name,
        hybrid_search=app_settings.hybrid_search_enabled,
        hybrid_rrf_k=app_settings.hybrid_rrf_k,
        embedding_dimension=app_settings.get_embedding_dimension(),
    )
    embedder = build_embedder(app_settings, rate_limiter)
    reranker = None
    if app_settings.reranker_enabled:
        reranker = CrossEncoderReranker(model_name=app_settings.reranker_model)

    telemetry = build_telemetry_tracer()
    if token_tracker is None:
        token_tracker = TokenTracker()
    if retrieval_tracker is None:
        retrieval_tracker = RetrievalTracker()

    # Phase 2 modules
    query_rewriter = QueryRewriter(
        llm_client=llm_client,
        enabled=app_settings.query_rewrite_enabled,
    )
    groundedness = GroundednessVerifier(
        llm_client=llm_client,
        enabled=app_settings.groundedness_enabled,
    )
    context_compressor = ContextCompressor(
        enabled=app_settings.context_compression_enabled,
        max_chunks=app_settings.retrieval_top_k,
    )
    from data_engineering_copilot.services.ragas_evaluation import RagasEvaluator

    ragas_evaluator = RagasEvaluator()

    return AsyncRagService(
        config=rag_config,
        vector_store=vector_store,
        llm_client=llm_client,
        code_llm_client=code_llm_client,
        embedder=embedder,
        reranker=reranker,
        telemetry=telemetry,
        cache=TwoTierCache(
            exact_enabled=True,
            semantic_enabled=True,
            similarity_threshold=app_settings.semantic_cache_threshold,
            ttl_seconds=app_settings.semantic_cache_ttl,
        ),
        query_rewriter=query_rewriter,
        groundedness_verifier=groundedness,
        context_compressor=context_compressor,
        token_tracker=token_tracker,
        retrieval_tracker=retrieval_tracker,
        ragas_evaluator=ragas_evaluator,
    )
