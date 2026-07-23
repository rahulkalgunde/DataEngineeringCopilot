from __future__ import annotations

import logging

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.domain.models import RagConfig
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache
from data_engineering_copilot.infrastructure.crawl_db import CrawlFrontierDB
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.services.async_ingestion import AsyncIngestionService
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.chunker import ChunkingStrategy, DocumentChunker
from data_engineering_copilot.services.semantic_chunker import SemanticChunker
from data_engineering_copilot.workers.progress import get_redis_client

logger = logging.getLogger(__name__)


def build_chunker(app_settings: AppSettings = settings):
    strategy = app_settings.chunking_strategy.lower()

    if strategy == "semantic":
        if not app_settings.enable_semantic_chunking:
            logger.warning(
                "Semantic chunking requested but disabled in settings. "
                "Falling back to sentence_preserving. "
                "Set enable_semantic_chunking=True to use semantic chunking."
            )
            strategy = "sentence_preserving"
        else:
            logger.info(
                "Building semantic chunker strategy=%s similarity=%s",
                strategy,
                app_settings.min_semantic_similarity,
            )
            return SemanticChunker(
                chunk_size_words=app_settings.chunk_size_words,
                overlap_words=app_settings.chunk_overlap_words,
                embedding_model=AsyncOllamaEmbeddings(
                    model_name=app_settings.embedding_model_name,
                ),
                min_semantic_similarity=app_settings.min_semantic_similarity,
                min_chunk_words=int(app_settings.chunk_size_words * 0.1),
                max_chunk_words=app_settings.max_chunk_words or int(app_settings.chunk_size_words * 1.5),
            )

    if strategy not in ["fixed_size", "sentence_preserving"]:
        logger.warning(
            "Unknown chunking strategy '%s'. Defaulting to sentence_preserving.",
            strategy,
        )
        strategy = "sentence_preserving"

    logger.info(
        "Building document chunker strategy=%s size=%s overlap=%s",
        strategy,
        app_settings.chunk_size_words,
        app_settings.chunk_overlap_words,
    )

    return DocumentChunker(
        chunk_size_words=app_settings.chunk_size_words,
        overlap_words=app_settings.chunk_overlap_words,
        strategy=ChunkingStrategy(strategy),
        min_chunk_words=int(app_settings.chunk_size_words * 0.1),
    )


def build_async_crawler(app_settings: AppSettings = settings) -> AsyncDocumentationCrawler:
    logger.info(
        "Building async crawler db=%s concurrency=%s max_concurrency=%s",
        app_settings.crawl_db_path,
        app_settings.crawl_async_concurrency,
        app_settings.crawl_async_max_concurrency,
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
    logger.info(
        "Building async ingestion service sources=%s qdrant_url=%s collection=%s",
        len(app_settings.sources),
        app_settings.qdrant_url,
        app_settings.collection_name,
    )
    try:
        redis_client = get_redis_client()
    except Exception:
        redis_client = None

    return AsyncIngestionService(
        settings=app_settings,
        crawler=build_async_crawler(app_settings),
        parser=MarkdownParser(),
        chunker=build_chunker(app_settings),
        embeddings=AsyncOllamaEmbeddings(model_name=app_settings.embedding_model_name),
        vector_store=AsyncQdrantVectorStore(
            url=app_settings.qdrant_url,
            collection_name=app_settings.collection_name,
            hybrid_search=app_settings.hybrid_search_enabled,
        ),
        redis_client=redis_client,
    )


def build_rag_service(app_settings: AppSettings = settings) -> AsyncRagService:
    from data_engineering_copilot.observability.telemetry import build_telemetry_tracer
    from data_engineering_copilot.observability.token_tracker import TokenTracker
    from data_engineering_copilot.services.context_compression import ContextCompressor
    from data_engineering_copilot.services.groundedness import GroundednessVerifier
    from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
    from data_engineering_copilot.services.query_rewriting import QueryRewriter
    from data_engineering_copilot.services.reranker import CrossEncoderReranker

    logger.info(
        "Building async RAG service model=%s top_k=%s max_context_chars=%s hybrid=%s",
        app_settings.ollama_model,
        app_settings.retrieval_top_k,
        app_settings.max_context_chars,
        app_settings.hybrid_search_enabled,
    )
    rag_config = RagConfig(
        retrieval_top_k=app_settings.retrieval_top_k,
        confidence_threshold=app_settings.confidence_threshold,
        reranker_enabled=app_settings.reranker_enabled,
        reranker_model=app_settings.reranker_model,
        reranker_top_k=app_settings.reranker_top_k,
        max_context_chars=app_settings.max_context_chars,
    )
    llm_client = AsyncOllamaClient(
        base_url=app_settings.ollama_base_url,
        model=app_settings.ollama_model,
        timeout_seconds=app_settings.ollama_timeout_seconds,
        num_ctx=app_settings.ollama_num_ctx,
        num_predict=app_settings.ollama_num_predict,
    )
    vector_store = AsyncQdrantVectorStore(
        url=app_settings.qdrant_url,
        collection_name=app_settings.collection_name,
        hybrid_search=app_settings.hybrid_search_enabled,
    )
    embedder = AsyncOllamaEmbeddings(
        model_name=app_settings.embedding_model_name,
    )
    reranker = None
    if app_settings.reranker_enabled:
        reranker = CrossEncoderReranker(model_name=app_settings.reranker_model)

    telemetry = build_telemetry_tracer()
    token_tracker = TokenTracker()

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

    return AsyncRagService(
        config=rag_config,
        vector_store=vector_store,
        llm_client=llm_client,
        embedder=embedder,
        reranker=reranker,
        telemetry=telemetry,
        cache=TwoTierCache(exact_enabled=True, semantic_enabled=True),
        query_rewriter=query_rewriter,
        groundedness_verifier=groundedness,
        context_compressor=context_compressor,
        token_tracker=token_tracker,
    )
