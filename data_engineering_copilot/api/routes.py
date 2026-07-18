"""FastAPI routes for ingestion dispatch, status polling and control."""

from __future__ import annotations

import json

import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from data_engineering_copilot.workers.celery_app import celery_app
from data_engineering_copilot.workers.progress import get_redis_client
from data_engineering_copilot.workers.tasks import async_ingest_task

log = structlog.get_logger(__name__)

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
    log.info("ingest.dispatch", source_names=request.source_names, max_pages=request.max_pages)
    task = async_ingest_task.delay(request.source_names or [], request.max_pages or 0)

    # Write an initial status so the polling endpoint has something to
    # return immediately, before the worker picks up the task.
    client = get_redis_client()
    initial_status = json.dumps({
        "task_id": task.id,
        "status": "DISPATCHED",
        "source_names": request.source_names or [],
        "pages_fetched": 0,
        "chunks_indexed": 0,
        "current_url": "",
        "error": None,
    })
    client.set(f"{REDIS_KEY_PREFIX}:{task.id}", initial_status)

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
        client.set(redis_key, json.dumps({
            "task_id": task_id,
            "status": "CANCELLED",
            "source_names": [],
            "pages_fetched": 0,
            "chunks_indexed": 0,
            "current_url": "",
            "error": None,
        }))

    return {"task_id": task_id, "status": "CANCELLED"}
