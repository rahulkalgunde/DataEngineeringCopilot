import time

import pytest

from unittest.mock import patch, Mock, MagicMock

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.ui.streamlit_app import IngestionManager, IngestionProgress
from data_engineering_copilot.domain.models import IngestionEvent

@patch("data_engineering_copilot.ui.streamlit_app.build_ingestion_service")
@patch("data_engineering_copilot.ui.streamlit_app.rag_service")
@patch("data_engineering_copilot.ui.streamlit_app.vector_store")
def test_ingestion_manager_start_and_stop(mock_vector_store, mock_rag_service, mock_build_service):
    # Setup mock service
    mock_service = MagicMock()
    mock_build_service.return_value = mock_service

    # Define a custom ingest mock that sleeps to allow cancellation
    def fake_ingest(max_pages_per_source, source_names, on_event):
        on_event(IngestionEvent(event_type="fetch_start", source_name="Test Source", message="Fetching...", url="http://example.com/1"))
        # Wait until cancel is requested (with timeout)
        for _ in range(200):
            if IngestionManager.get_progress().cancel_requested:
                break
            time.sleep(0.01)
        on_event(IngestionEvent(event_type="fetch_success", source_name="Test Source", message="Fetched...", url="http://example.com/1", pages_fetched=1))
        return 1

    mock_service.ingest.side_effect = fake_ingest

    # Ensure clean state
    IngestionManager.reset_status()
    assert not IngestionManager.is_running()

    started = IngestionManager.start(max_pages_per_source=1, source_names=("Test Source",))
    assert started
    assert IngestionManager.is_running()

    # Give the background thread a moment to start
    time.sleep(0.1)

    # Stop ingestion
    IngestionManager.stop()
    assert IngestionManager.get_progress().cancel_requested

    # Wait until the manager reports not running (with timeout)
    timeout = time.time() + 5.0
    while IngestionManager.is_running() and time.time() < timeout:
        time.sleep(0.05)

    # After stop, should not be running
    assert not IngestionManager.is_running()
    progress = IngestionManager.get_progress()
    # Should have cancellation message
    assert progress.cancel_requested
    assert "Ingestion cancelled by user." in progress.last_message

    # Reset for cleanliness
    IngestionManager.reset_status()
    assert not IngestionManager.is_running()
