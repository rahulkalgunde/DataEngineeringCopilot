"""Unit tests for ingestion task auto-discovery across sessions.

Tests cover:
- _get_latest_task_id: fetches the running task_id from the API
- IngestionManager.get_progress: auto-discovers a running task when session is fresh
- IngestionManager.get_progress: does not override an existing session task_id
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch


class TestGetLatestTaskId:
    def test_returns_task_id_on_200(self):
        from data_engineering_copilot.ui.streamlit_app import _get_latest_task_id

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "task_id": "abc-123",
                "status": "PROCESSING",
                "pages_fetched": 10,
                "chunks_indexed": 50,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen", return_value=mock_response):
            result = _get_latest_task_id()
        assert result == "abc-123"

    def test_returns_none_on_404(self):
        from data_engineering_copilot.ui.streamlit_app import _get_latest_task_id

        exc = urllib.error.HTTPError(
            url="http://localhost:8000/api/v1/ingest/latest",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=MagicMock(),
        )
        with patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen", side_effect=exc):
            result = _get_latest_task_id()
        assert result is None

    def test_returns_none_on_connection_error(self):
        from data_engineering_copilot.ui.streamlit_app import _get_latest_task_id

        with patch(
            "data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            result = _get_latest_task_id()
        assert result is None

    def test_returns_none_on_timeout(self):
        from data_engineering_copilot.ui.streamlit_app import _get_latest_task_id

        with patch(
            "data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = _get_latest_task_id()
        assert result is None

    def test_returns_none_on_500(self):
        from data_engineering_copilot.ui.streamlit_app import _get_latest_task_id

        exc = urllib.error.HTTPError(
            url="http://localhost:8000/api/v1/ingest/latest",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=MagicMock(),
        )
        with patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen", side_effect=exc):
            result = _get_latest_task_id()
        assert result is None


class TestAutoDiscovery:
    def test_auto_discovers_running_task_when_session_empty(self):
        import streamlit as st

        from data_engineering_copilot.ui.streamlit_app import IngestionManager

        for key in list(st.session_state.keys()):
            del st.session_state[key]
        running_status = {
            "task_id": "discovered-task",
            "status": "PROCESSING",
            "source_names": ["Apache Spark"],
            "pages_fetched": 5,
            "chunks_indexed": 20,
            "current_url": "https://spark.apache.org/docs",
            "error": None,
        }
        with (
            patch("data_engineering_copilot.ui.streamlit_app._get_latest_task_id", return_value="discovered-task"),
            patch("data_engineering_copilot.ui.streamlit_app._get_ingest_status", return_value=(running_status, None)),
        ):
            progress = IngestionManager.get_progress()
        assert progress.is_running is True
        assert progress.source_names == ("Apache Spark",)
        assert "discovered-task" in st.session_state.get("ingestion_task_id", "")

    def test_does_not_override_existing_task_id(self):
        import streamlit as st

        from data_engineering_copilot.ui.streamlit_app import IngestionManager

        st.session_state.ingestion_task_id = "existing-task"
        with (
            patch("data_engineering_copilot.ui.streamlit_app._get_latest_task_id") as mock_discover,
            patch(
                "data_engineering_copilot.ui.streamlit_app._get_ingest_status",
                return_value=(
                    {
                        "task_id": "existing-task",
                        "status": "PROCESSING",
                        "source_names": ["Spark"],
                        "pages_fetched": 3,
                        "chunks_indexed": 10,
                        "current_url": "",
                        "error": None,
                    },
                    None,
                ),
            ),
        ):
            progress = IngestionManager.get_progress()
        mock_discover.assert_not_called()
        assert progress.is_running is True

    def test_no_discovery_returns_idle(self):
        import streamlit as st

        from data_engineering_copilot.ui.streamlit_app import IngestionManager

        for key in list(st.session_state.keys()):
            del st.session_state[key]
        with patch("data_engineering_copilot.ui.streamlit_app._get_latest_task_id", return_value=None):
            progress = IngestionManager.get_progress()
        assert progress.is_running is False
        assert progress.error is None

    def test_discovery_completed_task_shows_success(self):
        import streamlit as st

        from data_engineering_copilot.ui.streamlit_app import IngestionManager

        for key in list(st.session_state.keys()):
            del st.session_state[key]
        completed_status = {
            "task_id": "completed-task",
            "status": "COMPLETED",
            "source_names": ["Spark"],
            "pages_fetched": 100,
            "chunks_indexed": 500,
            "current_url": "",
            "error": None,
        }
        with (
            patch("data_engineering_copilot.ui.streamlit_app._get_latest_task_id", return_value="completed-task"),
            patch(
                "data_engineering_copilot.ui.streamlit_app._get_ingest_status", return_value=(completed_status, None)
            ),
        ):
            progress = IngestionManager.get_progress()
        assert progress.is_running is False
        assert progress.success_message is not None
        assert "500" in progress.success_message
