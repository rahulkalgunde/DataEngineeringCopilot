import time
import pytest
from unittest.mock import MagicMock, patch
from data_engineering_copilot.domain.models import IngestionEvent
from data_engineering_copilot.ui.streamlit_app import IngestionManager, IngestionProgress, IngestionCancelledError

def test_ingestion_progress_initial():
    IngestionManager.reset_status()
    progress = IngestionManager.get_progress()
    assert not progress.is_running
    assert progress.max_pages == 0

@patch("data_engineering_copilot.ui.streamlit_app.build_ingestion_service")
@patch("data_engineering_copilot.ui.streamlit_app.rag_service")
@patch("data_engineering_copilot.ui.streamlit_app.vector_store")
def test_ingestion_manager_success_flow(mock_vector_store, mock_rag_service, mock_build_service):
    # Setup mock service
    mock_service = MagicMock()
    mock_build_service.return_value = mock_service

    # Define a custom ingest mock that emits events
    def fake_ingest(max_pages_per_source, source_names, on_event):
        # Emit some events
        on_event(IngestionEvent(event_type="fetch_success", source_name="Test Source", message="Fetched page 1", url="http://example.com/1", pages_fetched=1))
        on_event(IngestionEvent(event_type="page_indexed", source_name="Test Source", message="Indexed page 1", url="http://example.com/1", chunks_indexed=5))
        return 5

    mock_service.ingest.side_effect = fake_ingest

    # Reset IngestionManager state
    IngestionManager.reset_status()

    # Start ingestion
    started = IngestionManager.start(10, ("Test Source",))
    assert started
    assert IngestionManager.is_running()

    # Wait for the daemon thread to finish
    for _ in range(50):
        if not IngestionManager.is_running():
            break
        time.sleep(0.05)

    assert not IngestionManager.is_running()
    progress = IngestionManager.get_progress()
    assert progress.pages_fetched == 1
    assert progress.chunks_indexed == 5
    assert "complete" in progress.success_message
    assert progress.error is None
    
    # Assert cache clear methods were called
    mock_rag_service.clear.assert_called_once()
    mock_vector_store.clear.assert_called_once()

@patch("data_engineering_copilot.ui.streamlit_app.build_ingestion_service")
@patch("data_engineering_copilot.ui.streamlit_app.rag_service")
@patch("data_engineering_copilot.ui.streamlit_app.vector_store")
def test_ingestion_manager_cancellation(mock_vector_store, mock_rag_service, mock_build_service):
    # Setup mock service
    mock_service = MagicMock()
    mock_build_service.return_value = mock_service

    # Ingest that sleeps to allow cancellation to be processed
    def fake_ingest(max_pages_per_source, source_names, on_event):
        on_event(IngestionEvent(event_type="fetch_start", source_name="Test Source", message="Fetching...", url="http://example.com/1"))
        # Wait until cancel is requested
        for _ in range(100):
            if IngestionManager.get_progress().cancel_requested:
                break
            time.sleep(0.01)
        on_event(IngestionEvent(event_type="fetch_success", source_name="Test Source", message="Fetched...", url="http://example.com/1", pages_fetched=1))
        return 1

    mock_service.ingest.side_effect = fake_ingest

    IngestionManager.reset_status()

    # Start
    assert IngestionManager.start(10, ("Test Source",))
    assert IngestionManager.is_running()

    # Request stop
    IngestionManager.stop()
    assert IngestionManager.get_progress().cancel_requested

    # Wait for completion
    for _ in range(50):
        if not IngestionManager.is_running():
            break
        time.sleep(0.05)

    assert not IngestionManager.is_running()
    progress = IngestionManager.get_progress()
    assert progress.error == "Ingestion cancelled."
    assert "cancelled" in progress.last_message

@patch("data_engineering_copilot.ui.streamlit_app.build_ingestion_service")
@patch("data_engineering_copilot.ui.streamlit_app.rag_service")
@patch("data_engineering_copilot.ui.streamlit_app.vector_store")
def test_ingestion_manager_error_handling(mock_vector_store, mock_rag_service, mock_build_service):
    # Setup mock service
    mock_service = MagicMock()
    mock_build_service.return_value = mock_service
    mock_service.ingest.side_effect = ValueError("Chroma connection error")

    IngestionManager.reset_status()

    # Start
    assert IngestionManager.start(10, ("Test Source",))
    
    # Wait for completion
    for _ in range(50):
        if not IngestionManager.is_running():
            break
        time.sleep(0.05)

    assert not IngestionManager.is_running()
    progress = IngestionManager.get_progress()
    assert progress.error == "Chroma connection error"
    assert "failed" in progress.last_message
