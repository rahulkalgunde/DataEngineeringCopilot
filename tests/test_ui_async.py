import time

import pytest

from unittest.mock import patch, Mock

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.ui.streamlit_app import IngestionManager, IngestionProgress

@patch('data_engineering_copilot.infrastructure.qdrant_store.QdrantClient')
def test_ingestion_manager_start_and_stop(mock_qdrant_client):
    # Ensure clean state
    IngestionManager.reset_status()
    assert not IngestionManager.is_running()

    # Use a valid source name to avoid selection error
    valid_source = settings.sources[0].name if settings.sources else ""
    started = IngestionManager.start(max_pages_per_source=1, source_names=(valid_source,))
    assert started
    assert IngestionManager.is_running()

    # Give the background thread a moment to start
    time.sleep(0.1)

    # Stop ingestion
    IngestionManager.stop()

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