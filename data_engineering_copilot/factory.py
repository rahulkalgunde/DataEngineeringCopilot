from __future__ import annotations

import logging

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser

from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.ingestion import IngestionService
from data_engineering_copilot.services.rag import ProductionRagService

logger = logging.getLogger(__name__)


def build_ingestion_service(app_settings: AppSettings = settings) -> IngestionService:
    logger.info(
        "Building ingestion service sources=%s qdrant_url=%s collection=%s",
        len(app_settings.sources),
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
        chunker=DocumentChunker(
            chunk_size_words=app_settings.chunk_size_words,
            overlap_words=app_settings.chunk_overlap_words,
        ),
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
