from __future__ import annotations

import logging

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser

from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker, ChunkingStrategy
from data_engineering_copilot.services.semantic_chunker import SemanticChunker
from data_engineering_copilot.services.ingestion import IngestionService
from data_engineering_copilot.services.rag import ProductionRagService

logger = logging.getLogger(__name__)


def build_chunker(app_settings: AppSettings = settings):
    """
    Build a document chunker based on configured strategy.

    Supports multiple strategies:
    - "fixed_size": Legacy word-based chunking
    - "sentence_preserving": Sentence-boundary aware (default)
    - "semantic": Embedding-based semantic clustering (requires enable_semantic_chunking=True)

    Args:
        app_settings: Application settings containing chunking configuration

    Returns:
        DocumentChunker or SemanticChunker instance

    Raises:
        ValueError: If semantic chunking requested but disabled, or invalid strategy
    """
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
            embedding_model = SentenceTransformerEmbeddings(
                model_name=app_settings.embedding_model_name,
                cache_dir=app_settings.embedding_cache_dir,
                local_files_only=app_settings.embedding_local_files_only,
            )
            return SemanticChunker(
                chunk_size_words=app_settings.chunk_size_words,
                overlap_words=app_settings.chunk_overlap_words,
                embedding_model=embedding_model,
                min_semantic_similarity=app_settings.min_semantic_similarity,
                min_chunk_words=int(app_settings.chunk_size_words * 0.1),  # 10% of target
                max_chunk_words=app_settings.max_chunk_words
                or int(app_settings.chunk_size_words * 1.5),
            )

    # For fixed_size or sentence_preserving
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
        min_chunk_words=int(app_settings.chunk_size_words * 0.1),  # 10% of target
    )


def build_ingestion_service(app_settings: AppSettings = settings) -> IngestionService:
    logger.info(
        "Building ingestion service sources=%s strategy=%s qdrant_url=%s collection=%s",
        len(app_settings.sources),
        app_settings.chunking_strategy,
        app_settings.qdrant_url,
        app_settings.collection_name,
    )
    return IngestionService(
        settings=app_settings,
        crawler=DocumentationCrawler(
            timeout_seconds=app_settings.request_timeout_seconds,
            delay_seconds=app_settings.crawl_delay_seconds,
        ),
        parser=DocumentationHtmlParser(),
        chunker=build_chunker(app_settings),
        embeddings=SentenceTransformerEmbeddings(
            model_name=app_settings.embedding_model_name,
            cache_dir=app_settings.embedding_cache_dir,
            local_files_only=app_settings.embedding_local_files_only,
        ),
        vector_store=QdrantVectorStore(
            url=app_settings.qdrant_url,
            collection_name=app_settings.collection_name,
        ),
    )


def build_rag_service(app_settings: AppSettings = settings) -> ProductionRagService:
    logger.info(
        "Building Production RAG service model=%s top_k=%s max_context_chars=%s",
        app_settings.ollama_model,
        app_settings.retrieval_top_k,
        app_settings.max_context_chars,
    )
    return ProductionRagService()
