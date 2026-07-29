"""FastAPI routes for ingestion dispatch, status polling, control, and RAG ask."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from data_engineering_copilot.workers.celery_app import celery_app
from data_engineering_copilot.workers.progress import get_redis_client
from data_engineering_copilot.workers.tasks import async_ingest_task

log = structlog.get_logger(__name__)
logger = logging.getLogger(__name__)

router = APIRouter()

REDIS_KEY_PREFIX = "ingestion:status"


def _resolve_source_filter(request, client_filter: list[str] | None) -> list[str] | None:
    """Apply RBAC source filtering if permissions are present.

    Admin roles bypass the filter. Reader roles intersect the client filter
    (if any) with the user's allowed sources.
    """
    perms = getattr(getattr(request, "state", None), "user_permissions", None)
    if perms is None:
        return client_filter
    if perms.role == "admin":
        return client_filter  # admin sees everything
    if not perms.allowed_sources:
        return client_filter  # empty = all sources
    if client_filter:
        # Intersect: user can only see sources they're allowed AND the client requested
        return [s for s in client_filter if s in perms.allowed_sources]
    return list(perms.allowed_sources)


class IngestRequest(BaseModel):
    source_names: list[str] | None = Field(default=None, max_length=20)
    max_pages: int | None = Field(default=None, ge=1, le=10000)


class TaskStatus(BaseModel):
    task_id: str
    state: str
    result: dict | None = None


@router.post("/api/v1/ingest", response_model=TaskStatus)
async def ingest_documents(request: IngestRequest):
    log.info(
        "ingest.dispatch",
        source_names=request.source_names,
        max_pages=request.max_pages,
    )

    client = get_redis_client()

    # Atomic SETNX lock to prevent concurrent dispatch (TOCTOU race).
    dispatch_id = str(uuid.uuid4())
    acquired = client.set("ingestion:dispatch_lock", dispatch_id, nx=True, ex=60)
    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="Ingestion dispatch already in progress. Please wait.",
        )

    try:
        raw_task_id = client.get("ingestion:latest_task_id")
        if raw_task_id:
            try:
                latest_task_id = raw_task_id.decode() if isinstance(raw_task_id, bytes) else str(raw_task_id)
                raw = client.get(f"{REDIS_KEY_PREFIX}:{latest_task_id}")
                if raw:
                    raw = raw.decode() if isinstance(raw, bytes) else raw
                    existing_status = json.loads(raw).get("status")
                    if existing_status in ("PROCESSING", "DISPATCHED"):
                        task_res = AsyncResult(latest_task_id)
                        if task_res.state in ("FAILURE", "REVOKED"):
                            existing_status = "FAILED"
                    if existing_status in ("PROCESSING", "DISPATCHED"):
                        raise HTTPException(
                            status_code=409,
                            detail=f"Ingestion is already running (task {latest_task_id}). Cancel it or wait for completion.",
                        )
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        task = async_ingest_task.delay(request.source_names, request.max_pages or 0)

        # Write an initial status so the polling endpoint has something to
        # return immediately, before the worker picks up the task.
        initial_status = json.dumps(
            {
                "task_id": task.id,
                "status": "DISPATCHED",
                "source_names": request.source_names or [],
                "pages_fetched": 0,
                "chunks_indexed": 0,
                "current_url": "",
                "error": None,
            }
        )
        client.set(f"{REDIS_KEY_PREFIX}:{task.id}", initial_status, ex=86400)
        client.set("ingestion:latest_task_id", task.id, ex=86400)

        return TaskStatus(task_id=task.id, state=task.state)
    finally:
        client.delete("ingestion:dispatch_lock")


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
    client = get_redis_client()
    raw = client.get(f"{REDIS_KEY_PREFIX}:{task_id}")

    if not raw:
        raise HTTPException(
            status_code=404,
            detail="Ingestion task status tracking record not found.",
        )

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Ingestion status record is corrupted.",
        ) from exc


@router.get("/api/v1/ingest/latest")
async def get_latest_ingestion() -> dict:
    """Return the status of the most recently dispatched ingestion task."""
    client = get_redis_client()
    raw_task_id = client.get("ingestion:latest_task_id")
    if not raw_task_id:
        raise HTTPException(status_code=404, detail="No ingestion task found.")
    task_id = raw_task_id.decode() if isinstance(raw_task_id, bytes) else raw_task_id
    raw = client.get(f"{REDIS_KEY_PREFIX}:{task_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Task status expired.")
    raw = raw.decode() if isinstance(raw, bytes) else raw
    return json.loads(raw)


@router.post("/api/v1/ingest/{task_id}/cancel")
async def cancel_ingestion(task_id: str) -> dict:
    """Cancel a running Celery ingestion task and update its Redis status."""
    log.info("ingest.cancel", task_id=task_id)
    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")

    client = get_redis_client()
    redis_key = f"{REDIS_KEY_PREFIX}:{task_id}"
    raw = client.get(redis_key)

    if raw:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        data["status"] = "CANCELLED"
        client.set(redis_key, json.dumps(data))
    else:
        client.set(
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
        )

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
        effective_source_filter = _resolve_source_filter(fastapi_request, request.source_filter)
        answer_obj = await asyncio.wait_for(
            service.answer(
                request.question,
                source_filter=effective_source_filter,
            ),
            timeout=120.0,
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
        logger.warning("RAG ask timed out after 120s question=%r", request.question[:100])
        raise HTTPException(status_code=504, detail="Request timed out. Try a simpler question.") from None
    except Exception as exc:
        logger.exception("RAG ask failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/api/v1/ask/stream")
async def ask_stream(request: AskRequest, fastapi_request: Request):
    """Streaming RAG answer with Server-Sent Events."""
    from fastapi.responses import StreamingResponse

    from data_engineering_copilot.services.rag_service_singleton import get_rag_service

    effective_source_filter = _resolve_source_filter(fastapi_request, request.source_filter)

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
