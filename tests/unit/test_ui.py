"""Unit tests for the Streamlit UI ingestion flow.

Tests cover the new Celery+Redis-backed IngestionManager that delegates
ingestion to a background task and polls Redis for progress.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from data_engineering_copilot.ui.streamlit_app import (
    IngestionManager,
    _get_ingest_status,
    _post_cancel_ingest,
    _post_ingest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_session():
    """Clear Streamlit session state before and after each test."""
    st.session_state.clear()
    yield
    st.session_state.clear()


# ---------------------------------------------------------------------------
# _post_ingest tests
# ---------------------------------------------------------------------------

class TestPostIngest:
    """Tests for the HTTP helper that starts a Celery ingestion task."""

    @patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen")
    def test_returns_task_id_on_success(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps({"task_id": "task-abc", "state": "PENDING"}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        task_id, error = _post_ingest(["Apache Spark"], 40)

        assert task_id == "task-abc"
        assert error is None

    @patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen")
    def test_returns_error_on_exception(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionError("refused")

        task_id, error = _post_ingest(["Apache Spark"], 40)

        assert task_id is None
        assert error is not None
        assert "refused" in error


# ---------------------------------------------------------------------------
# _get_ingest_status tests
# ---------------------------------------------------------------------------

class TestGetIngestStatus:
    """Tests for the HTTP helper that polls task status from Redis."""

    @patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen")
    def test_returns_status_dict(self, mock_urlopen):
        expected = {
            "task_id": "task-abc",
            "status": "PROCESSING",
            "pages_fetched": 5,
            "chunks_indexed": 40,
            "current_url": "https://example.com",
            "error": None,
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(expected).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = _get_ingest_status("task-abc")

        assert result["status"] == "PROCESSING"
        assert result["pages_fetched"] == 5

    @patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen")
    def test_returns_none_on_404(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs=None, fp=None
        )

        result = _get_ingest_status("nonexistent")

        assert result is None

    @patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen")
    def test_returns_none_on_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionError("refused")

        result = _get_ingest_status("task-abc")

        assert result is None


# ---------------------------------------------------------------------------
# _post_cancel_ingest tests
# ---------------------------------------------------------------------------

class TestPostCancelIngest:
    """Tests for the HTTP helper that cancels a Celery task."""

    @patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen")
    def test_returns_true_on_success(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        assert _post_cancel_ingest("task-abc") is True

    @patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen")
    def test_returns_false_on_exception(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionError("refused")

        assert _post_cancel_ingest("task-abc") is False


# ---------------------------------------------------------------------------
# IngestionManager tests
# ---------------------------------------------------------------------------

class TestIngestionManager:
    """Tests for the IngestionManager class methods."""

    @patch("data_engineering_copilot.ui.streamlit_app._post_ingest")
    def test_start_stores_task_id(self, mock_post):
        mock_post.return_value = ("task-xyz", None)

        started, error = IngestionManager.start(("Spark",), 40)

        assert started is True
        assert error == ""
        assert st.session_state.ingestion_task_id == "task-xyz"
        assert st.session_state.ingestion_source_names == ["Spark"]
        assert st.session_state.ingestion_max_pages == 40

    @patch("data_engineering_copilot.ui.streamlit_app._post_ingest")
    def test_start_returns_error_on_api_failure(self, mock_post):
        mock_post.return_value = (None, "Connection refused")

        started, error = IngestionManager.start(("Spark",), 40)

        assert started is False
        assert "Connection refused" in error

    def test_get_progress_returns_empty_when_no_task(self):
        progress = IngestionManager.get_progress()

        assert progress.is_running is False
        assert progress.error is None

    @patch("data_engineering_copilot.ui.streamlit_app._get_ingest_status")
    def test_get_progress_running(self, mock_status):
        st.session_state.ingestion_task_id = "task-abc"
        st.session_state.ingestion_source_names = ["Spark"]
        st.session_state.ingestion_max_pages = 40
        st.session_state.ingestion_start_time = 1000.0
        mock_status.return_value = {
            "task_id": "task-abc",
            "status": "PROCESSING",
            "source_names": ["Spark"],
            "pages_fetched": 10,
            "chunks_indexed": 80,
            "current_url": "https://example.com",
            "error": None,
        }

        progress = IngestionManager.get_progress()

        assert progress.is_running is True
        assert progress.total_pages_fetched == 10
        assert progress.total_chunks_indexed == 80
        assert progress.current_phase == "crawling"

    @patch("data_engineering_copilot.ui.streamlit_app._get_ingest_status")
    def test_get_progress_completed(self, mock_status):
        st.session_state.ingestion_task_id = "task-abc"
        st.session_state.ingestion_source_names = ["Spark"]
        st.session_state.ingestion_max_pages = 40
        st.session_state.ingestion_start_time = 1000.0
        mock_status.return_value = {
            "task_id": "task-abc",
            "status": "COMPLETED",
            "source_names": ["Spark"],
            "pages_fetched": 40,
            "chunks_indexed": 320,
            "current_url": "",
            "error": None,
        }

        progress = IngestionManager.get_progress()

        assert progress.is_running is False
        assert progress.success_message is not None
        assert "320" in progress.success_message
        assert progress.current_phase == "complete"

    @patch("data_engineering_copilot.ui.streamlit_app._get_ingest_status")
    def test_get_progress_failed(self, mock_status):
        st.session_state.ingestion_task_id = "task-abc"
        st.session_state.ingestion_source_names = ["Spark"]
        st.session_state.ingestion_max_pages = 40
        st.session_state.ingestion_start_time = 1000.0
        mock_status.return_value = {
            "task_id": "task-abc",
            "status": "FAILED",
            "source_names": ["Spark"],
            "pages_fetched": 3,
            "chunks_indexed": 24,
            "current_url": "https://example.com/bad",
            "error": "Connection refused",
        }

        progress = IngestionManager.get_progress()

        assert progress.is_running is False
        assert progress.error == "Connection refused"

    @patch("data_engineering_copilot.ui.streamlit_app._get_ingest_status")
    def test_get_progress_cancelled(self, mock_status):
        st.session_state.ingestion_task_id = "task-abc"
        st.session_state.ingestion_source_names = ["Spark"]
        st.session_state.ingestion_max_pages = 40
        st.session_state.ingestion_start_time = 1000.0
        mock_status.return_value = {
            "task_id": "task-abc",
            "status": "CANCELLED",
            "source_names": ["Spark"],
            "pages_fetched": 5,
            "chunks_indexed": 40,
            "current_url": "",
            "error": None,
        }

        progress = IngestionManager.get_progress()

        assert progress.is_running is False
        assert progress.error is not None

    @patch("data_engineering_copilot.ui.streamlit_app._get_ingest_status")
    def test_get_progress_404_returns_error(self, mock_status):
        st.session_state.ingestion_task_id = "task-abc"
        st.session_state.ingestion_source_names = ["Spark"]
        st.session_state.ingestion_max_pages = 40
        st.session_state.ingestion_start_time = 1000.0
        mock_status.return_value = None

        progress = IngestionManager.get_progress()

        assert progress.is_running is False
        assert progress.error is not None
        assert "not found" in progress.error.lower()

    @patch("data_engineering_copilot.ui.streamlit_app._post_cancel_ingest")
    def test_stop_calls_cancel_api(self, mock_cancel):
        st.session_state.ingestion_task_id = "task-abc"
        mock_cancel.return_value = True

        result = IngestionManager.stop()

        assert result is True
        mock_cancel.assert_called_once_with("task-abc")

    def test_stop_returns_false_when_no_task(self):
        result = IngestionManager.stop()
        assert result is False

    def test_reset_clears_session_state(self):
        st.session_state.ingestion_task_id = "task-abc"
        st.session_state.ingestion_source_names = ["Spark"]
        st.session_state.ingestion_max_pages = 40
        st.session_state.ingestion_start_time = 1000.0

        IngestionManager.reset_status()

        assert "ingestion_task_id" not in st.session_state
        assert "ingestion_source_names" not in st.session_state
        assert "ingestion_max_pages" not in st.session_state
        assert "ingestion_start_time" not in st.session_state

    @patch("data_engineering_copilot.ui.streamlit_app._get_ingest_status")
    def test_get_progress_polls_api(self, mock_status):
        st.session_state.ingestion_task_id = "task-xyz"
        st.session_state.ingestion_source_names = ["Spark"]
        st.session_state.ingestion_max_pages = 40
        st.session_state.ingestion_start_time = 1000.0
        mock_status.return_value = {
            "task_id": "task-xyz",
            "status": "PROCESSING",
            "source_names": ["Spark"],
            "pages_fetched": 0,
            "chunks_indexed": 0,
            "current_url": "",
            "error": None,
        }

        IngestionManager.get_progress()

        mock_status.assert_called_once_with("task-xyz")
