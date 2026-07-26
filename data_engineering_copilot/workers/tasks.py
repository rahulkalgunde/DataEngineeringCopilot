"""Celery tasks for background ingestion.

``execute_background_ingestion`` is the legacy Crawl4AI-based task.
``async_ingest_task`` is the production task that uses the full
IngestionService pipeline with Redis progress tracking.
"""

from __future__ import annotations

import asyncio
import json

import structlog
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure
from crawl4ai import AsyncWebCrawler

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.workers.celery_app import celery_app
from data_engineering_copilot.workers.progress import (
    _STATUS_KEY_TTL_SECONDS,
    IngestionProgressTracker,
    get_redis_client,
)

log = structlog.get_logger(__name__)

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
    """Legacy Celery entry point that crawls URLs directly using Crawl4AI."""

    async def _pipeline():
        raw_docs = await _run_async_crawl(urls)

        embedder = AsyncOllamaEmbeddings(
            model_name=settings.embedding_model_name,
        )
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size_words * 5,
            chunk_overlap=settings.chunk_overlap_words * 5,
        )
        vector_store = AsyncQdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )

        await vector_store.initialize()

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
            embeddings = await embedder.embed_texts([c.text for c in chunks])
            await vector_store.upsert_chunks(chunks, embeddings)
            processed += 1
        return processed

    processed_count = asyncio.run(_pipeline())
    return {"status": "INGESTION_COMPLETED", "processed_count": processed_count}


@celery_app.task(
    bind=True,
    queue="ingestion",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_kwargs={"max_retries": 3, "countdown": 10},
    retry_backoff=True,
)
def async_ingest_task(self, source_names: list[str], max_pages: int):
    """Production ingestion task using the full AsyncIngestionService pipeline.

    Progress is persisted to Redis via ``IngestionProgressTracker`` so that
    the Streamlit UI and API endpoints can poll for real-time updates.
    """
    task_id = self.request.id
    log.info(
        "async_ingest_task.started",
        task_id=task_id,
        source_names=source_names,
        max_pages=max_pages,
    )
    tracker = IngestionProgressTracker(task_id, redis_client=get_redis_client(), source_names=source_names)

    try:
        from data_engineering_copilot.factory import build_async_ingestion_service

        service = build_async_ingestion_service()
        asyncio.run(
            service.ingest(
                source_names=source_names,
                max_pages_per_source=max_pages,
                on_event=tracker.on_event,
            )
        )
        tracker.mark_completed()
        log.info("async_ingest_task.completed", task_id=task_id)
    except SoftTimeLimitExceeded:
        err_msg = "Task exceeded soft time limit. Execution cancelled."
        log.error("async_ingest_task.timeout", task_id=task_id)
        tracker.mark_failed(err_msg)
        raise
    except Exception as e:
        log.exception("async_ingest_task.failed", task_id=task_id, error=str(e))
        tracker.mark_failed(str(e))
        raise


@task_failure.connect
def _on_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    """Update the Redis progress tracker when a task fails unexpectedly.

    This catches failures that bypass the try/except block in the task body,
    most notably the hard time limit (``TimeLimitExceeded``) which kills
    the worker process before Python exception handling can execute.
    """
    if not task_id:
        return
    try:
        client = get_redis_client()
        redis_key = f"ingestion:status:{task_id}"
        raw = client.get(redis_key)
        if raw is None:
            return
        state = json.loads(raw) if isinstance(raw, bytes) else json.loads(raw.decode("utf-8"))
        if state.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            return
        state["status"] = "FAILED"
        state["error"] = str(exception or "Task failed unexpectedly (hard time limit?).")
        client.set(redis_key, json.dumps(state), ex=_STATUS_KEY_TTL_SECONDS)
        log.info("task_failure.updated_redis", task_id=task_id, error=state["error"])
    except Exception as exc:
        log.warning("task_failure.update_failed", task_id=task_id, error=str(exc))
