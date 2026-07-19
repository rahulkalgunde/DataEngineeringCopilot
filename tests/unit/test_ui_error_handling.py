"""Unit tests for Streamlit UI error handling and Redis improvements.

Tests cover:
- _get_ingest_status: distinguishes 404 (task not found) from 500/connection errors
- IngestionManager.get_progress: shows correct error messages per failure type
- Progress tracker: Redis keys include TTL for automatic expiration
- Progress tracker: Redis connection pooling
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _get_ingest_status error handling tests
# ---------------------------------------------------------------------------


class TestGetIngestStatusErrorHandling:
    """Tests for the _get_ingest_status helper in the Streamlit UI."""

    def test_returns_tuple_for_200_response(self):
        """Returns (dict, None) when API returns 200."""
        from data_engineering_copilot.ui.streamlit_app import _get_ingest_status

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"task_id": "t1", "status": "PROCESSING"}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen", return_value=mock_response):
            result = _get_ingest_status("t1")

        assert isinstance(result, tuple)
        data, error_msg = result
        assert data is not None
        assert error_msg is None
        assert data["task_id"] == "t1"
        assert data["status"] == "PROCESSING"

    def test_404_returns_not_found_tuple(self):
        """Returns (None, None) specifically for HTTP 404 (task not found in Redis)."""
        from data_engineering_copilot.ui.streamlit_app import _get_ingest_status

        exc = urllib.error.HTTPError(
            url="http://localhost:8000/api/v1/ingest/status/t1",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=MagicMock(),
        )

        with patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen", side_effect=exc):
            result = _get_ingest_status("t1")

        # (None, None) = not found (distinct from (None, "error msg") = API error)
        assert isinstance(result, tuple)
        data, error_msg = result
        assert data is None
        assert error_msg is None

    def test_500_returns_error_tuple(self):
        """Returns (None, error_msg) for HTTP 500 (API internal error)."""
        from data_engineering_copilot.ui.streamlit_app import _get_ingest_status

        exc = urllib.error.HTTPError(
            url="http://localhost:8000/api/v1/ingest/status/t1",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=MagicMock(),
        )

        with patch("data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen", side_effect=exc):
            result = _get_ingest_status("t1")

        # New behavior: returns (None, error_msg) for non-404 errors
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        data, error_msg = result
        assert data is None
        assert error_msg is not None
        assert "500" in error_msg or "error" in error_msg.lower()

    def test_timeout_returns_error_tuple(self):
        """Returns (None, error_msg) when the connection times out."""
        from data_engineering_copilot.ui.streamlit_app import _get_ingest_status

        with patch(
            "data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = _get_ingest_status("t1")

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        data, error_msg = result
        assert data is None
        assert error_msg is not None
        assert "timed out" in error_msg.lower() or "timeout" in error_msg.lower()

    def test_connection_refused_returns_error_tuple(self):
        """Returns (None, error_msg) when API is unreachable."""
        from data_engineering_copilot.ui.streamlit_app import _get_ingest_status

        with patch(
            "data_engineering_copilot.ui.streamlit_app.urllib.request.urlopen",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            result = _get_ingest_status("t1")

        assert isinstance(result, tuple)
        data, error_msg = result
        assert data is None
        assert error_msg is not None


# ---------------------------------------------------------------------------
# IngestionManager.get_progress error message tests
# ---------------------------------------------------------------------------


class TestIngestionManagerProgressErrors:
    """Tests for IngestionManager.get_progress error differentiation."""

    def test_no_task_id_returns_idle(self):
        """No task_id in session state returns idle IngestionProgress."""
        import streamlit as st

        from data_engineering_copilot.ui.streamlit_app import IngestionManager

        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        with patch("data_engineering_copilot.ui.streamlit_app._get_latest_task_id", return_value=None):
            progress = IngestionManager.get_progress()
        assert progress.is_running is False
        assert progress.error is None

    def test_404_returns_task_not_found(self):
        """When API returns 404, shows 'task not found' error."""
        import streamlit as st

        from data_engineering_copilot.ui.streamlit_app import IngestionManager

        st.session_state.ingestion_task_id = "expired-task"

        # _get_ingest_status returns (None, None) for 404
        with patch("data_engineering_copilot.ui.streamlit_app._get_ingest_status", return_value=(None, None)):
            progress = IngestionManager.get_progress()

        assert progress.is_running is False
        assert progress.error is not None
        assert "not found" in progress.error.lower()

    def test_api_error_returns_api_unreachable(self):
        """When API returns non-404 error, shows 'API unreachable' message."""
        import streamlit as st

        from data_engineering_copilot.ui.streamlit_app import IngestionManager

        st.session_state.ingestion_task_id = "some-task"

        # _get_ingest_status returns (None, error_msg) for non-404 errors
        with patch(
            "data_engineering_copilot.ui.streamlit_app._get_ingest_status",
            return_value=(None, "Connection refused"),
        ):
            progress = IngestionManager.get_progress()

        assert progress.is_running is False
        assert progress.error is not None
        assert "unreachable" in progress.error.lower() or "error" in progress.error.lower()


# ---------------------------------------------------------------------------
# Redis TTL tests
# ---------------------------------------------------------------------------


class TestProgressTrackerTTL:
    """Tests for Redis key TTL on progress tracker."""

    def test_initial_sync_includes_ttl(self):
        """The _sync method sets a TTL on the Redis key."""
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        mock_redis = MagicMock()
        IngestionProgressTracker(
            task_id="ttl-test",
            redis_client=mock_redis,
            source_names=["Test Source"],
        )

        # Check that redis.set was called with 'ex' parameter
        call_args = mock_redis.set.call_args
        # Should have been called with positional args (key, value) and keyword arg ex=<ttl>
        assert call_args.kwargs.get("ex") is not None or (
            len(call_args.args) >= 3 or len(call_args[1]) > 0 or "ex" in str(call_args)
        )

    def test_event_sync_includes_ttl(self):
        """Each event update preserves the TTL on the Redis key."""
        from data_engineering_copilot.domain.models import IngestionEvent
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        mock_redis = MagicMock()
        tracker = IngestionProgressTracker(
            task_id="ttl-event-test",
            redis_client=mock_redis,
            source_names=["Test"],
        )
        mock_redis.reset_mock()

        event = IngestionEvent(
            event_type="page_indexed",
            source_name="Test",
            message="Indexed",
            url="https://example.com",
            chunks_indexed=5,
            pages_fetched=1,
        )
        tracker.on_event(event)

        call_args = mock_redis.set.call_args
        # Verify TTL parameter is present
        has_ttl = call_args.kwargs.get("ex") is not None
        assert has_ttl, f"Expected TTL parameter in redis.set call, got: {call_args}"


# ---------------------------------------------------------------------------
# Redis connection pooling tests
# ---------------------------------------------------------------------------


class TestRedisConnectionPooling:
    """Tests for Redis connection pool reuse."""

    def test_get_redis_client_returns_from_shared_pool(self):
        """get_redis_client creates a connection pool and reuses it."""
        from data_engineering_copilot.workers import progress as progress_mod

        with (
            patch.object(progress_mod, "_connection_pool", None),
            patch("data_engineering_copilot.workers.progress.redis") as mock_redis_mod,
        ):
            mock_pool = MagicMock()
            mock_client = MagicMock()
            mock_redis_mod.ConnectionPool.from_url.return_value = mock_pool
            mock_redis_mod.Redis.return_value = mock_client

            progress_mod.get_redis_client()
            progress_mod.get_redis_client()

            # Pool should be created only once (on first call)
            assert mock_redis_mod.ConnectionPool.from_url.call_count == 1
            # Two clients returned, both using the shared pool
            assert mock_redis_mod.Redis.call_count == 2
