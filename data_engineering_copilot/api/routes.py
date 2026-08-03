"""FastAPI routes for ingestion dispatch, status polling, control, and RAG ask."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import redis.asyncio as aioredis
import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import CacheScope
from data_engineering_copilot.workers.celery_app import celery_app
from data_engineering_copilot.workers.progress import (
    _LEASE_KEY_PREFIX,
    _STATUS_KEY_TTL_SECONDS,
)
from data_engineering_copilot.workers.tasks import async_ingest_task

log = structlog.get_logger(__name__)
logger = logging.getLogger(__name__)

router = APIRouter()

REDIS_KEY_PREFIX = "ingestion:status"

_async_redis: aioredis.Redis | None = None


async def _get_async_redis() -> aioredis.Redis:
    """Return a lazily-created async Redis client (decode_responses=True).

    Shared across route handlers so connection pooling is centralized.
    """
    global _async_redis
    if _async_redis is None:
        _async_redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _async_redis


def _resolve_source_filter(
    request,
    client_filter: list[str] | None,
    *,
    rbac_enabled: bool = False,
) -> list[str] | None:
    """Apply RBAC source filtering, failing closed when RBAC is enabled.

    - ``rbac_enabled=False``: pass the client filter through unchanged.
    - ``rbac_enabled=True``: admin roles bypass; readers are restricted to
      their ``allowed_sources``. Any of the following raises
      ``AuthorizationError`` (HTTP 403) instead of silently widening scope:
      - no ``UserPermissions`` resolved for the caller;
      - a reader with an empty ``allowed_sources`` (they can see nothing);
      - an empty intersection between the client filter and the reader's
        allowed sources.
    An empty filter is never returned to signal "all sources".
    """
    from data_engineering_copilot.domain.exceptions import AuthorizationError

    if not rbac_enabled:
        return client_filter

    perms = getattr(getattr(request, "state", None), "user_permissions", None)
    if perms is None:
        raise AuthorizationError("Caller has no resolved permissions for this RBAC-enabled endpoint")
    if perms.role == "admin":
        return client_filter  # admin sees everything
    if not perms.allowed_sources:
        raise AuthorizationError("Caller has no permitted sources")
    if client_filter:
        intersection = [s for s in client_filter if s in perms.allowed_sources]
        if not intersection:
            raise AuthorizationError("Requested sources are not permitted for this caller")
        return intersection
    return list(perms.allowed_sources)


def _build_cache_scope(request, source_filter: list[str] | None) -> CacheScope:
    """Build the cache isolation scope for the current request.

    Combines tenant context, role, the resolved source filter, and the active
    embedding model / collection so cached answers are never served across
    tenants or scopes.
    """
    from data_engineering_copilot.domain.models import CacheScope

    perms = getattr(getattr(request, "state", None), "user_permissions", None)
    tenant_id = request.headers.get("X-Tenant-ID", "default")
    role = perms.role if perms is not None else "anonymous"
    return CacheScope(
        tenant_id=tenant_id,
        role=role,
        source_filter=tuple(source_filter or ()),
        embedding_model=settings.embedding_model_name,
        collection_name=settings.collection_name,
    )


async def _reconcile_ingestion_status(client, task_id: str, state: dict) -> dict:
    """Repair progress state when Celery or the worker disappeared.

    Hot path is async-only: for a PROCESSING task a live worker lease means the
    task is healthy, so we return immediately without touching the Celery
    backend. The synchronous ``AsyncResult`` query (which blocks the event
    loop) is only run as a backstop — via ``asyncio.to_thread`` — when the
    lease is missing.
    """
    if state.get("status") not in ("PROCESSING", "DISPATCHED"):
        return state

    # A live lease means the worker is healthy; trust Redis progress and skip
    # the (blocking) Celery backend round-trip entirely.
    if state.get("status") == "PROCESSING" and await client.exists(f"{_LEASE_KEY_PREFIX}:{task_id}"):
        return state

    celery_state = await asyncio.to_thread(lambda: AsyncResult(task_id).state)
    terminal_state = {
        "SUCCESS": ("COMPLETED", None),
        "FAILURE": ("FAILED", "Celery reported task failure."),
        "REVOKED": ("CANCELLED", "Task was revoked."),
    }.get(celery_state)
    if terminal_state is not None:
        state["status"], default_error = terminal_state
        if default_error and not state.get("error"):
            state["error"] = default_error
        await client.set(
            f"{REDIS_KEY_PREFIX}:{task_id}",
            json.dumps(state),
            ex=_STATUS_KEY_TTL_SECONDS,
        )
        await client.delete(f"{_LEASE_KEY_PREFIX}:{task_id}")
        return state

    if state.get("status") == "DISPATCHED":
        dispatched_at = state.get("dispatched_at", 0)
        if dispatched_at and time.time() - dispatched_at > 300:
            state["status"] = "FAILED"
            state["error"] = "Task orphaned (dispatched but not running for > 5 min)."
            await client.set(
                f"{REDIS_KEY_PREFIX}:{task_id}",
                json.dumps(state),
                ex=_STATUS_KEY_TTL_SECONDS,
            )
        return state

    # PENDING is ambiguous while a task is queued, but a PROCESSING task must
    # have a live worker lease. Its absence means the worker was lost.
    if not await client.exists(f"{_LEASE_KEY_PREFIX}:{task_id}"):
        state["status"] = "FAILED"
        state["error"] = "Task heartbeat expired; worker likely lost."
        await client.set(
            f"{REDIS_KEY_PREFIX}:{task_id}",
            json.dumps(state),
            ex=_STATUS_KEY_TTL_SECONDS,
        )
    return state


class IngestRequest(BaseModel):
    source_names: list[str] | None = Field(default=None, max_length=20)
    max_pages: int | None = Field(default=None, ge=1, le=20000)


class TaskStatus(BaseModel):
    task_id: str
    state: str
    result: dict | None = None


@router.post("/api/v1/ingest", response_model=TaskStatus)
async def ingest_documents(request: IngestRequest, fastapi_request: Request):
    # Gate: refuse to dispatch if the Docker image is stale (deps changed since build)
    from data_engineering_copilot.api.app import _deps_fingerprint_ok

    if _deps_fingerprint_ok is False:
        from data_engineering_copilot.infrastructure.dep_check import STALE_MESSAGE

        raise HTTPException(status_code=503, detail=STALE_MESSAGE)

    # RBAC: restrict ingest sources to the caller's allowed_sources
    effective_sources = _resolve_source_filter(
        fastapi_request, request.source_names, rbac_enabled=settings.rbac_enabled
    )
    log.info(
        "ingest.dispatch",
        source_names=effective_sources,
        max_pages=request.max_pages,
    )

    client = await _get_async_redis()

    # Atomic SETNX lock to prevent concurrent dispatch (TOCTOU race).
    dispatch_id = str(uuid.uuid4())
    acquired = await client.set("ingestion:dispatch_lock", dispatch_id, nx=True, ex=60)
    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="Ingestion dispatch already in progress. Please wait.",
        )

    try:
        raw_task_id = await client.get("ingestion:latest_task_id")
        if raw_task_id:
            try:
                latest_task_id = str(raw_task_id)
                raw = await client.get(f"{REDIS_KEY_PREFIX}:{latest_task_id}")
                if raw:
                    existing_data = json.loads(raw)
                    existing_data = await _reconcile_ingestion_status(client, latest_task_id, existing_data)
                    existing_status = existing_data.get("status")
                    if existing_status in ("PROCESSING", "DISPATCHED"):
                        raise HTTPException(
                            status_code=409,
                            detail=f"Ingestion is already running (task {latest_task_id}). Cancel it or wait for completion.",
                        )
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # Propagate W3C trace context to Celery worker for distributed tracing
        trace_headers = {}
        try:
            from data_engineering_copilot.observability.otel_telemetry import inject_w3c_context

            inject_w3c_context(trace_headers)
        except Exception:
            pass

        task = async_ingest_task.apply_async(
            args=(effective_sources, request.max_pages),
            headers=trace_headers,
        )

        # Write an initial status so the polling endpoint has something to
        # return immediately, before the worker picks up the task.
        initial_status = json.dumps(
            {
                "task_id": task.id,
                "status": "DISPATCHED",
                "dispatched_at": time.time(),
                "source_names": effective_sources or [],
                "pages_fetched": 0,
                "chunks_indexed": 0,
                "current_url": "",
                "error": None,
            }
        )
        await client.set(f"{REDIS_KEY_PREFIX}:{task.id}", initial_status, ex=86400)
        await client.set("ingestion:latest_task_id", task.id, ex=86400)

        return TaskStatus(task_id=task.id, state=task.state)
    finally:
        await client.delete("ingestion:dispatch_lock")


@router.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id)
    return TaskStatus(
        task_id=task_id,
        state=task_result.state,
        result=task_result.result if task_result.ready() else None,
    )


@router.get("/api/v1/ingest/status/{task_id}")
async def get_ingestion_status(task_id: str) -> dict:
    """Return the latest progress snapshot for a background ingestion task."""
    client = await _get_async_redis()
    raw = await client.get(f"{REDIS_KEY_PREFIX}:{task_id}")

    if not raw:
        raise HTTPException(
            status_code=404,
            detail="Ingestion task status tracking record not found.",
        )

    try:
        state = json.loads(raw)
        return await _reconcile_ingestion_status(client, task_id, state)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Ingestion status record is corrupted.",
        ) from exc


@router.get("/api/v1/ingest/latest")
async def get_latest_ingestion() -> dict:
    """Return the status of the most recently dispatched ingestion task."""
    client = await _get_async_redis()
    raw_task_id = await client.get("ingestion:latest_task_id")
    if not raw_task_id:
        raise HTTPException(status_code=404, detail="No ingestion task found.")
    task_id = str(raw_task_id)
    raw = await client.get(f"{REDIS_KEY_PREFIX}:{task_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Task status expired.")
    return await _reconcile_ingestion_status(client, task_id, json.loads(raw))


@router.post("/api/v1/ingest/{task_id}/cancel")
async def cancel_ingestion(task_id: str) -> dict:
    """Cancel a running Celery ingestion task and update its Redis status."""
    log.info("ingest.cancel", task_id=task_id)
    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")

    client = await _get_async_redis()
    redis_key = f"{REDIS_KEY_PREFIX}:{task_id}"
    raw = await client.get(redis_key)

    if raw:
        data = json.loads(raw)
        data["status"] = "CANCELLED"
        data["error"] = data.get("error") or "Task revoked by user"
        await client.set(redis_key, json.dumps(data), ex=_STATUS_KEY_TTL_SECONDS)
    else:
        await client.set(
            redis_key,
            json.dumps(
                {
                    "task_id": task_id,
                    "status": "CANCELLED",
                    "source_names": [],
                    "pages_fetched": 0,
                    "chunks_indexed": 0,
                    "current_url": "",
                    "error": None,
                }
            ),
            ex=_STATUS_KEY_TTL_SECONDS,
        )
    await client.delete(f"{_LEASE_KEY_PREFIX}:{task_id}")

    return {"task_id": task_id, "status": "CANCELLED"}


# --- RAG Ask endpoint ---


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    source_filter: list[str] | None = None
    rerank: bool = True


class SourceRef(BaseModel):
    source_name: str
    title: str
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence: float
    groundedness_score: float = 1.0
    citations: list[dict[str, str]] = []
    metrics: dict[str, float] = {}


@router.post("/api/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest, fastapi_request: Request):
    """Answer a question using the RAG pipeline."""
    from data_engineering_copilot.services.rag_service_singleton import get_rag_service
    from data_engineering_copilot.services.structured_output import parse_rag_response, verify_citations

    try:
        service = await get_rag_service()
        effective_source_filter = _resolve_source_filter(
            fastapi_request, request.source_filter, rbac_enabled=settings.rbac_enabled
        )
        cache_scope = _build_cache_scope(fastapi_request, effective_source_filter)

        # Extract user/session from request for Langfuse tracking
        user_id = fastapi_request.headers.get("X-User-ID") or fastapi_request.query_params.get("user_id")
        session_id = fastapi_request.headers.get("X-Session-ID") or fastapi_request.query_params.get("session_id")

        answer_obj = await asyncio.wait_for(
            service.answer(
                request.question,
                source_filter=effective_source_filter,
                cache_scope=cache_scope,
                user_id=user_id,
                session_id=session_id,
            ),
            timeout=max(120.0, float(settings.ollama_timeout_seconds)),
        )
        parsed = parse_rag_response(answer_obj.text)

        # Cross-reference citations against retrieved sources
        source_names = [src.source_name for src in answer_obj.sources]
        parsed.citations = verify_citations(parsed.citations, source_names)

        sources = [
            SourceRef(
                source_name=src.source_name,
                title=src.title,
                url=src.url,
                snippet=src.text[:200],
            )
            for src in answer_obj.sources
        ]

        return AskResponse(
            answer=parsed.answer,
            sources=sources,
            confidence=answer_obj.confidence,
            groundedness_score=answer_obj.groundedness_score,
            citations=parsed.citations,
            metrics={
                "chunks_retrieved": len(answer_obj.sources),
                "confidence": answer_obj.confidence,
                **{f"time_{k}": v for k, v in answer_obj.stage_times.items()},
            },
        )
    except TimeoutError:
        logger.warning(
            "RAG ask timed out after %ss question=%r",
            max(120, settings.ollama_timeout_seconds),
            request.question[:100],
        )
        raise HTTPException(status_code=504, detail="Request timed out. Try a simpler question.") from None
    except Exception as exc:
        logger.exception("RAG ask failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/api/v1/ask/stream")
async def ask_stream(request: AskRequest, fastapi_request: Request):
    """Streaming RAG answer with Server-Sent Events."""
    from fastapi.responses import StreamingResponse

    from data_engineering_copilot.services.rag_service_singleton import get_rag_service

    effective_source_filter = _resolve_source_filter(
        fastapi_request, request.source_filter, rbac_enabled=settings.rbac_enabled
    )

    async def event_stream():
        try:
            service = await get_rag_service()
            async for event in service.answer_stream(
                request.question,
                source_filter=effective_source_filter,
            ):
                yield event
        except TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Request timed out'})}\n\n"
        except Exception as exc:
            logger.exception("RAG streaming ask failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal server error'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
