import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from data_engineering_copilot.api.app import app
import data_engineering_copilot.api.run_api
import data_engineering_copilot.api.run_server

client = TestClient(app)

@patch("data_engineering_copilot.api.routes.ingestion_task.delay")
def test_ingest_documents(mock_delay):
    mock_task = MagicMock()
    mock_task.id = "test-task-123"
    mock_task.state = "PENDING"
    mock_delay.return_value = mock_task

    response = client.post("/api/v1/ingest", json={"source_names": ["Test"], "max_pages": 10})
    assert response.status_code == 200
    assert response.json() == {"task_id": "test-task-123", "state": "PENDING", "result": None}
    mock_delay.assert_called_once_with(["Test"], 10)

@patch("data_engineering_copilot.api.routes.AsyncResult")
def test_get_task_status_pending(mock_async_result):
    mock_task = MagicMock()
    mock_task.state = "PENDING"
    mock_task.ready.return_value = False
    mock_async_result.return_value = mock_task

    response = client.get("/api/v1/task/test-task-123")
    assert response.status_code == 200
    assert response.json() == {"task_id": "test-task-123", "state": "PENDING", "result": None}

@patch("data_engineering_copilot.api.routes.AsyncResult")
def test_get_task_status_ready(mock_async_result):
    mock_task = MagicMock()
    mock_task.state = "SUCCESS"
    mock_task.ready.return_value = True
    mock_task.result = {"chunks": 5}
    mock_async_result.return_value = mock_task

    response = client.get("/api/v1/task/test-task-123")
    assert response.status_code == 200
    assert response.json() == {"task_id": "test-task-123", "state": "SUCCESS", "result": {"chunks": 5}}
