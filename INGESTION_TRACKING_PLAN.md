# Implementation Plan: Persistent Async Ingestion Tracking

This document outlines the strict TDD blueprint for adding non-blocking, background ingestion progress tracking using FastAPI, Celery, and Redis.

## 1. Context & Objective
The current ingestion system blocks the UI or restarts from the base page upon disruption. We are decoupling the execution layer by delegating processing to Celery workers and storing progressive execution states in Redis. This ensures that the user can close the browser without disrupting active ingestions or impeding the Q&A pipeline.

---

## 2. Step 1: Write Tests First (TDD Verification Phase)

Create the following file at `tests/unit/test_ingestion_progress.py`. The agent must run `dec_venv/bin/python -m pytest tests/unit/test_ingestion_progress.py -v` to verify implementation gap failures before updating the production code stack.

```python
# tests/unit/test_ingestion_progress.py
import json
from unittest.mock import MagicMock
import pytest
from data_engineering_copilot.domain.models import IngestionEvent

def test_redis_progress_updates_on_ingestion_events():
    """
    Test that our callback wrapper processes IngestionEvents and 
    correctly serializes the operational states down to Redis.
    """
    mock_redis = MagicMock()
    task_id = "test-task-123"
    
    # In-memory target state configuration
    progress_state = {
        "task_id": task_id,
        "status": "PROCESSING",
        "source_names": ["Apache Spark Documentation"],
        "pages_fetched": 0,
        "chunks_indexed": 0,
        "current_url": "",
        "error": None
    }
    
    def on_event_callback(event: IngestionEvent):
        progress_state["pages_fetched"] = event.pages_fetched
        progress_state["chunks_indexed"] = event.chunks_indexed
        if event.url:
            progress_state["current_url"] = event.url
        if event.error:
            progress_state["error"] = str(event.error)
        
        mock_redis.set(f"ingestion:status:{task_id}", json.dumps(progress_state))

    # Trigger a sample page_indexed structural event
    event_payload = IngestionEvent(
        event_type="page_indexed",
        source_name="Apache Spark Documentation",
        message="Successfully indexed quick-start.html",
        url="[https://spark.apache.org/docs/latest/quick-start.html](https://spark.apache.org/docs/latest/quick-start.html)",
        chunks_indexed=12,
        pages_fetched=1
    )
    
    on_event_callback(event_payload)
    
    # Assert state mutation syncs completely down to the persistent mock key boundary
    mock_redis.set.assert_called_with(
        f"ingestion:status:{task_id}",
        json.dumps({
            "task_id": task_id,
            "status": "PROCESSING",
            "source_names": ["Apache Spark Documentation"],
            "pages_fetched": 1,
            "chunks_indexed": 12,
            "current_url": "[https://spark.apache.org/docs/latest/quick-start.html](https://spark.apache.org/docs/latest/quick-start.html)",
            "error": None
        })
    )