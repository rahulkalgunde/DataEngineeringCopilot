"""Celery tasks for background ingestion.

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

from data_engineering_copilot.factory import get_shared_redis_client
from data_engineering_copilot.workers.celery_app import celery_app
from data_engineering_copilot.workers.progress import (
    _LEASE_KEY_PREFIX,
    _STATUS_KEY_TTL_SECONDS,
    IngestionProgressTracker,
)

log = structlog.get_logger(__name__)

# Alias for ``celery -A data_engineering_copilot.workers.tasks worker``
app = celery_app

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_loop_lock = threading.Lock()


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


def _validate_ingest_inputs(source_names: list[str] | None, max_pages: int | None) -> None:
    """Validate task inputs that arrive from the Celery broker.

    The API route validates requests via Pydantic, but the broker is a separate
    trust boundary — a compromised broker could inject arbitrary arguments that
    bypass API validation. This guard rejects malformed inputs up-front with a
    clear error instead of failing deep inside the pipeline.
    """
    from pydantic import ValidationError

    from data_engineering_copilot.domain.models import IngestRequest

    try:
        IngestRequest(source_names=source_names, max_pages=max_pages)
    except ValidationError as exc:
        raise ValueError(f"Invalid ingestion task inputs: {exc}") from exc
    if not source_names:
        raise ValueError("Invalid ingestion task inputs: source_names must be non-empty")


@celery_app.task(
    bind=True,
    queue="ingestion",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_kwargs={"max_retries": 3, "countdown": 10},
    retry_backoff=True,
)
def async_ingest_task(self, source_names: list[str], max_pages: int | None):
    """Production ingestion task using the full AsyncIngestionService pipeline.

    Progress is persisted to Redis via ``IngestionProgressTracker`` so that
    the Streamlit UI and API endpoints can poll for real-time updates.
    """
    # Propagate W3C trace context from Celery task headers
    try:
        from data_engineering_copilot.observability.otel_telemetry import extract_w3c_context

        trace_ctx = extract_w3c_context(dict(self.request.headers or {}))
        if trace_ctx is not None:
            from opentelemetry import context as otel_context

            otel_context.attach(trace_ctx)
    except Exception:
        pass

    task_id = self.request.id
    _validate_ingest_inputs(source_names, max_pages)
    log.info(
        "async_ingest_task.started",
        task_id=task_id,
        source_names=source_names,
        max_pages=max_pages,
    )
    tracker = IngestionProgressTracker(
        task_id,
        redis_client=get_shared_redis_client(),
        source_names=source_names,
    )

    try:
        from data_engineering_copilot.factory import build_async_ingestion_service

        async def _run_ingest() -> None:
            await tracker.start()
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
        try:
            run_async(tracker.aclose())
        except Exception as exc:
            log.warning("async_ingest_task.aclose_failed", task_id=task_id, error=str(exc))


async def _set_status_one_shot(task_id: str, status: str, error: str) -> None:
    """One-shot async write used by the ``task_failure``/``task_revoked`` handlers.

    Runs on the worker event loop via :func:`run_async`.  The shared async
    Redis client decodes responses as ``str``, but bytes are handled too for
    safety.  Terminal states are never overwritten.
    """
    redis_client = get_shared_redis_client()
    redis_key = f"ingestion:status:{task_id}"
    raw = await redis_client.get(redis_key)
    if raw is None:
        return
    state = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
    if state.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
        return
    state["status"] = status
    state["error"] = error
    await redis_client.set(redis_key, json.dumps(state), ex=_STATUS_KEY_TTL_SECONDS)
    await redis_client.delete(f"{_LEASE_KEY_PREFIX}:{task_id}")


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
        run_async(
            _set_status_one_shot(task_id, "FAILED", str(exception or "Task failed unexpectedly (hard time limit?)."))
        )
        log.info("task_failure.updated_redis", extra={"task_id": task_id})
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
        run_async(_set_status_one_shot(task_id, "CANCELLED", "Task revoked by user"))
        log.info("task_revoked.updated_redis", extra={"task_id": task_id})
    except Exception as exc:
        log.warning("task_revoked.update_failed", extra={"task_id": task_id, "error": str(exc)})
