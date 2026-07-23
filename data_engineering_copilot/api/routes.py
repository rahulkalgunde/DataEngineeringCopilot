"""FastAPI routes for ingestion dispatch, status polling, control, and RAG ask."""

from __future__ import annotations

import json
import logging

import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from data_engineering_copilot.workers.celery_app import celery_app
from data_engineering_copilot.workers.progress import get_redis_client
from data_engineering_copilot.workers.tasks import async_ingest_task

log = structlog.get_logger(__name__)
logger = logging.getLogger(__name__)

router = APIRouter()

REDIS_KEY_PREFIX = "ingestion:status"


class IngestRequest(BaseModel):
    source_names: list[str] | None = None
    max_pages: int | None = None


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
    raw_task_id = client.get("ingestion:latest_task_id")
    if raw_task_id:
        try:
            latest_task_id = raw_task_id.decode() if isinstance(raw_task_id, bytes) else str(raw_task_id)
            raw = client.get(f"{REDIS_KEY_PREFIX}:{latest_task_id}")
            if raw:
                raw = raw.decode() if isinstance(raw, bytes) else raw
                existing_status = json.loads(raw).get("status")
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

    return json.loads(raw)


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
    citations: list[dict[str, str]] = []
    metrics: dict[str, float] = {}


@router.post("/api/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Answer a question using the RAG pipeline."""
    from data_engineering_copilot.factory import build_rag_service
    from data_engineering_copilot.services.structured_output import parse_rag_response, verify_citations

    try:
        service = build_rag_service()
        answer_obj = await service.answer(request.question)
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
            citations=parsed.citations,
            metrics={"chunks_retrieved": len(answer_obj.sources), "confidence": answer_obj.confidence},
        )
    except Exception as exc:
        logger.exception("RAG ask failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc
