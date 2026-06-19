from __future__ import annotations

import logging

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
from data_engineering_copilot.infrastructure.vector_store import ChromaVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.ingestion import IngestionService
from data_engineering_copilot.services.rag import RagAnswerService


logger = logging.getLogger(__name__)


def build_ingestion_service(app_settings: AppSettings = settings) -> IngestionService:
    logger.info(
        "Building ingestion service sources=%s chroma_dir=%s collection=%s",
        len(app_settings.sources),
        app_settings.chroma_dir,
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
        vector_store=ChromaVectorStore(
            persist_directory=str(app_settings.chroma_dir),
            collection_name=app_settings.collection_name,
        ),
    )


def build_rag_service(app_settings: AppSettings = settings) -> RagAnswerService:
    logger.info(
        "Building RAG service model=%s top_k=%s max_context_chars=%s threshold=%.4f",
        app_settings.ollama_model,
        app_settings.retrieval_top_k,
        app_settings.max_context_chars,
        app_settings.confidence_threshold,
    )
    return RagAnswerService(
        embeddings=SentenceTransformerEmbeddings(
            model_name=app_settings.embedding_model_name,
            cache_dir=app_settings.embedding_cache_dir,
            local_files_only=app_settings.embedding_local_files_only,
        ),
        vector_store=ChromaVectorStore(
            persist_directory=str(app_settings.chroma_dir),
            collection_name=app_settings.collection_name,
        ),
        ollama_client=OllamaClient(
            base_url=app_settings.ollama_base_url,
            model=app_settings.ollama_model,
            timeout_seconds=app_settings.ollama_timeout_seconds,
            num_ctx=app_settings.ollama_num_ctx,
            num_predict=app_settings.ollama_num_predict,
        ),
        top_k=app_settings.retrieval_top_k,
        max_context_chars=app_settings.max_context_chars,
        confidence_threshold=app_settings.confidence_threshold,
    )
