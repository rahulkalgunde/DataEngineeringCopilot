"""Celery tasks for background ingestion.

``execute_background_ingestion`` is the legacy Crawl4AI-based task.
``async_ingest_task`` is the production task that uses the full
IngestionService pipeline with Redis progress tracking.
"""

from __future__ import annotations

import asyncio
import logging

from crawl4ai import AsyncWebCrawler

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.factory import build_ingestion_service
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.workers.celery_app import celery_app
from data_engineering_copilot.workers.progress import IngestionProgressTracker, get_redis_client

logger = logging.getLogger(__name__)

# Alias for ``celery -A data_engineering_copilot.workers.tasks worker``
app = celery_app


async def _run_async_crawl(urls: list[str]):
    """Crawl a list of URLs concurrently and return the raw Crawl4AI results."""
    async with AsyncWebCrawler(verbose=True) as crawler:
        tasks = [crawler.arun(url=url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results


@celery_app.task
def execute_background_ingestion(urls: list[str]):
    """Celery entry point.

    The function runs the async crawler, parses the markdown, chunks the
    text, embeds the chunks and finally upserts them into Qdrant.
    """
    loop = asyncio.get_event_loop()
    raw_docs = loop.run_until_complete(_run_async_crawl(urls))

    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model_name,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    chunker = DocumentChunker(
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
    )

    processed = 0
    for doc in raw_docs:
        if not getattr(doc, "success", False):
            continue

        text = doc.markdown
        chunks = chunker.chunk(
            type(
                "TmpDoc",
                (),
                {
                    "source_name": "crawl4ai",
                    "title": doc.title or doc.url,
                    "url": doc.url,
                    "text": text,
                },
            )()
        )
        embeddings = embedder.embed_texts([c.text for c in chunks])
        vector_store.upsert_chunks(chunks, embeddings)
        processed += 1

    return {"status": "INGESTION_COMPLETED", "processed_count": processed}


@celery_app.task(bind=True)
def async_ingest_task(self, source_names: list[str], max_pages: int):
    """Production ingestion task using the full IngestionService pipeline.

    Progress is persisted to Redis via ``IngestionProgressTracker`` so that
    the Streamlit UI and API endpoints can poll for real-time updates.
    """
    task_id = self.request.id
    logger.info(
        "async_ingest_task started task_id=%s source_names=%s max_pages=%s",
        task_id, source_names, max_pages,
    )
    tracker = IngestionProgressTracker(task_id, redis_client=get_redis_client(), source_names=source_names)

    try:
        service = build_ingestion_service()
        service.ingest(
            source_names=source_names,
            max_pages_per_source=max_pages,
            on_event=tracker.on_event,
        )
        tracker.mark_completed()
        logger.info("async_ingest_task completed task_id=%s", task_id)
    except Exception as e:
        logger.exception("async_ingest_task failed task_id=%s error=%s", task_id, e)
        tracker.mark_failed(str(e))
