from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from celery.result import AsyncResult
import sys
sys.path.insert(0, '/app')

from data_engineering_copilot.workers.tasks import execute_background_ingestion as ingestion_task

router = APIRouter()

class IngestRequest(BaseModel):
    source_names: Optional[list[str]] = None
    max_pages: Optional[int] = None

class TaskStatus(BaseModel):
    task_id: str
    state: str
    result: Optional[dict] = None

@router.post("/api/v1/ingest", response_model=TaskStatus)
async def ingest_documents(request: IngestRequest):
    task = ingestion_task.delay(request.source_names, request.max_pages)
    return TaskStatus(task_id=task.id, state=task.state)

@router.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id)
    return TaskStatus(
        task_id=task_id,
        state=task_result.state,
        result=task_result.result if task_result.ready() else None
    )
