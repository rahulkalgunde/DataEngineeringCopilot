from __future__ import annotations

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.domain.models import RagConfig
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient
from data_engineering_copilot.infrastructure.async_openai_embeddings import OpenAIEmbeddings
from data_engineering_copilot.infrastructure.async_openrouter_client import OpenRouterLLMClient
from data_engineering_copilot.infrastructure.async_openrouter_embeddings import OpenRouterEmbeddings
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache
from data_engineering_copilot.infrastructure.crawl_db import CrawlFrontierDB
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.observability.structured_logging import StructuredLogger
from data_engineering_copilot.observability.token_tracker import RetrievalTracker, TokenTracker
from data_engineering_copilot.services.api_extractor import ApiDocExtractor
from data_engineering_copilot.services.async_ingestion import AsyncIngestionService
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.chunker import ChunkingStrategy, DocumentChunker
from data_engineering_copilot.services.code_block_parser import CodeBlockParser
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.semantic_chunker import SemanticChunker
from data_engineering_copilot.workers.progress import get_redis_client

logger = StructuredLogger(__name__)


def build_llm_client(app_settings: AppSettings = settings):
    """Build LLM client based on configured provider."""
    provider = app_settings.llm_provider.lower()
    if provider == "openrouter":
        api_key = app_settings.openrouter_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when llm_provider='openrouter'")
        return OpenRouterLLMClient(
            api_key=api_key,
            model=app_settings.openrouter_model,
            timeout_seconds=app_settings.ollama_timeout_seconds,
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


def build_embedder(app_settings: AppSettings = settings):
    """Build embedding provider based on configured provider."""
    provider = app_settings.embedding_provider.lower()
    if provider == "openai":
        api_key = app_settings.openai_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when embedding_provider='openai'")
        return OpenAIEmbeddings(
            api_key=api_key,
            model_name=app_settings.openai_embedding_model,
            base_url=app_settings.openai_embedding_base_url,
            embedding_dimension=app_settings.openai_embedding_dimension,
            batch_size=app_settings.embedding_batch_size,
        )
    elif provider == "openrouter":
        api_key = app_settings.openrouter_api_key.get_secret_value()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when embedding_provider='openrouter'")
        return OpenRouterEmbeddings(
            api_key=api_key,
            model_name=app_settings.openrouter_embedding_model,
            embedding_dimension=app_settings.openrouter_embedding_dimension,
            batch_size=app_settings.embedding_batch_size,
        )
    elif provider == "ollama":
        return AsyncOllamaEmbeddings(model_name=app_settings.embedding_model_name)
    else:
        raise ValueError(f"Unsupported embedding_provider: {provider!r}. Choose 'ollama', 'openai', or 'openrouter'.")


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
        chunk_size=app_settings.chunk_size_words,
        overlap=app_settings.chunk_overlap_words,
    )

    return DocumentChunker(
        chunk_size_words=app_settings.chunk_size_words,
        overlap_words=app_settings.chunk_overlap_words,
        strategy=ChunkingStrategy(strategy),
        min_chunk_words=int(app_settings.chunk_size_words * 0.1),
    )


def build_async_crawler(app_settings: AppSettings = settings) -> AsyncDocumentationCrawler:
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
    try:
        redis_client = get_redis_client()
    except Exception:
        redis_client = None

    contextual_enricher = ContextualChunkEnricher(
        summarizer=LLMContextSummarizer(llm_client=build_llm_client(app_settings)),
        enabled=app_settings.contextual_enrichment_enabled,
    )

    return AsyncIngestionService(
        settings=app_settings,
        crawler=build_async_crawler(app_settings),
        parser=MarkdownParser(),
        chunker=build_chunker(app_settings),
        embeddings=build_embedder(app_settings),
        vector_store=AsyncQdrantVectorStore(
            url=app_settings.qdrant_url,
            collection_name=app_settings.collection_name,
            hybrid_search=app_settings.hybrid_search_enabled,
            hybrid_rrf_k=app_settings.hybrid_rrf_k,
        ),
        redis_client=redis_client,
        contextual_enricher=contextual_enricher,
        api_extractor=ApiDocExtractor(enabled=getattr(app_settings, "api_extraction_enabled", True)),
        code_block_parser=CodeBlockParser(enabled=getattr(app_settings, "code_block_parsing_enabled", True)),
    )


def build_rag_service(app_settings: AppSettings = settings) -> AsyncRagService:
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
    llm_client = build_llm_client(app_settings)
    vector_store = AsyncQdrantVectorStore(
        url=app_settings.qdrant_url,
        collection_name=app_settings.collection_name,
        hybrid_search=app_settings.hybrid_search_enabled,
        hybrid_rrf_k=app_settings.hybrid_rrf_k,
    )
    embedder = build_embedder(app_settings)
    reranker = None
    if app_settings.reranker_enabled:
        reranker = CrossEncoderReranker(model_name=app_settings.reranker_model)

    telemetry = build_telemetry_tracer()
    token_tracker = TokenTracker()
    retrieval_tracker = RetrievalTracker()

    # Wire trackers to API metrics endpoint
    from data_engineering_copilot.api.app import set_trackers

    set_trackers(retrieval_tracker=retrieval_tracker, token_tracker=token_tracker)

    # New Phase 2 modules
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
