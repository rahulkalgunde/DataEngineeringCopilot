"""Celery tasks for background ingestion.

``execute_background_ingestion`` is the legacy Crawl4AI-based task.
``async_ingest_task`` is the production task that uses the full
IngestionService pipeline with Redis progress tracking.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Coroutine
from typing import Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure, task_revoked

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.factory import build_embedder
from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.workers.celery_app import celery_app
from data_engineering_copilot.workers.progress import (
    _LEASE_KEY_PREFIX,
    _STATUS_KEY_TTL_SECONDS,
    IngestionProgressTracker,
    get_redis_client,
)

log = structlog.get_logger(__name__)

# Alias for ``celery -A data_engineering_copilot.workers.tasks worker``
app = celery_app

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_loop_lock = threading.Lock()
_INGESTION_HEARTBEAT_INTERVAL_SECONDS = 30


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    """Return the single process-wide event loop used by all Celery tasks.

    ``asyncio.run()`` creates and then closes a fresh event loop per call.
    Loop-bound clients (async Redis, httpx/AsyncQdrantClient keep-alive
    connections) that outlive a task then get reused against a closed loop,
    surfacing as ``RuntimeError: Event loop is closed`` in later tasks.  A
    single long-lived loop keeps every lazy client bound to one loop that is
    never closed.
    """
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        with _worker_loop_lock:
            if _worker_loop is None or _worker_loop.is_closed():
                loop = asyncio.new_event_loop()

                def _run() -> None:
                    asyncio.set_event_loop(loop)
                    loop.run_forever()

                thread = threading.Thread(target=_run, name="dec-worker-loop", daemon=True)
                thread.start()
                _worker_loop = loop
    return _worker_loop


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* on the process-wide worker event loop and block for the result."""
    loop = _get_worker_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


async def _run_async_crawl(urls: list[str]):
    """Crawl a list of URLs concurrently and return the raw Crawl4AI results."""
    # Lazy import: crawl4ai calls load_dotenv() at import time, polluting
    # os.environ for the whole process. Importing this module (e.g. for Celery
    # task registration or FastAPI route wiring) must stay side-effect free.
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=True) as crawler:
        tasks = [crawler.arun(url=url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results


@celery_app.task
def execute_background_ingestion(urls: list[str]):
    """Legacy Celery entry point that crawls URLs directly using Crawl4AI."""

    async def _pipeline():
        raw_docs = await _run_async_crawl(urls)

        embedder = build_embedder(settings)
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

    processed_count = run_async(_pipeline())
    return {"status": "INGESTION_COMPLETED", "processed_count": processed_count}


def _validate_ingest_inputs(source_names: list[str] | None, max_pages: int) -> None:
    """Validate task inputs that arrive from the Celery broker.

    The API route validates requests via Pydantic, but the broker is a separate
    trust boundary — a compromised broker could inject arbitrary arguments that
    bypass API validation. This guard rejects malformed inputs up-front with a
    clear error instead of failing deep inside the pipeline.
    """
    from pydantic import BaseModel, Field, ValidationError

    class _IngestTaskInput(BaseModel):
        source_names: list[str] = Field(min_length=1, max_length=20)
        max_pages: int = Field(default=0, ge=0, le=20000)

    try:
        _IngestTaskInput(source_names=source_names or [], max_pages=max_pages)
    except ValidationError as exc:
        raise ValueError(f"Invalid ingestion task inputs: {exc}") from exc


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
    _validate_ingest_inputs(source_names, max_pages)
    log.info(
        "async_ingest_task.started",
        task_id=task_id,
        source_names=source_names,
        max_pages=max_pages,
    )
    tracker = IngestionProgressTracker(task_id, redis_client=get_redis_client(), source_names=source_names)
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(_INGESTION_HEARTBEAT_INTERVAL_SECONDS):
            try:
                tracker.heartbeat()
            except Exception as exc:
                log.warning("async_ingest_task.heartbeat_failed", task_id=task_id, error=str(exc))

    heartbeat_thread = threading.Thread(
        target=_heartbeat,
        name=f"ingestion-heartbeat-{(task_id or 'unknown')[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        from data_engineering_copilot.factory import build_async_ingestion_service

        async def _run_ingest() -> None:
            service = build_async_ingestion_service()
            await service.ingest(
                source_names=source_names,
                max_pages_per_source=max_pages,
                on_event=tracker.on_event,
            )

        run_async(_run_ingest())
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
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)
        tracker.clear_lease()


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
        client.delete(f"{_LEASE_KEY_PREFIX}:{task_id}")
        log.info("task_failure.updated_redis", extra={"task_id": task_id, "error": state["error"]})
    except Exception as exc:
        log.warning("task_failure.update_failed", extra={"task_id": task_id, "error": str(exc)})


@task_revoked.connect
def _on_task_revoked(sender=None, request=None, terminated=None, signum=None, expired=None, **kwargs):
    """Update the Redis progress tracker when a task is revoked.

    This serves as a backup for the cancel route: if the SIGTERM kills the
    worker process before the API route finishes writing ``CANCELLED`` to
    Redis, this handler ensures the status is still set.
    """
    task_id = getattr(request, "id", None) if request else None
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
        state["status"] = "CANCELLED"
        state["error"] = "Task revoked by user"
        client.set(redis_key, json.dumps(state), ex=_STATUS_KEY_TTL_SECONDS)
        client.delete(f"{_LEASE_KEY_PREFIX}:{task_id}")
        log.info("task_revoked.updated_redis", extra={"task_id": task_id})
    except Exception as exc:
        log.warning("task_revoked.update_failed", extra={"task_id": task_id, "error": str(exc)})
