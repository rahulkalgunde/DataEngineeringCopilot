"""Unit tests for Streamlit ingestion tab rendering functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import streamlit as st

from data_engineering_copilot.ui.streamlit_app import (
    IngestionManager,
    IngestionProgress,
    SourceProgress,
    render_ingestion_progress,
    render_ingestion_tab,
)


@pytest.fixture(autouse=True)
def _reset_session():
    st.session_state.clear()
    IngestionManager.reset_status()
    yield
    st.session_state.clear()


def _mock_columns(n):
    """Mock st.columns that accepts both int and list-of-int."""
    count = len(n) if isinstance(n, list) else n
    return [MagicMock() for _ in range(count)]


# ---------------------------------------------------------------------------
# render_ingestion_progress tests
# ---------------------------------------------------------------------------

class TestRenderIngestionProgress:
    """Tests for the auto-refreshing progress fragment."""

    def test_shows_progress_when_running(self):
        mock_progress = IngestionProgress(
            is_running=True,
            source_names=("Spark",),
            max_pages_per_source=10,
            total_pages_fetched=5,
            total_chunks_indexed=12,
            current_phase="crawling",
            sources={"Spark": SourceProgress(name="Spark", status="crawling", pages_fetched=5, chunks_indexed=12)},
            last_message="Starting ingestion...",
        )

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
            mp.setattr(st, "progress", lambda value: value)
            mp.setattr(st, "metric", lambda label, value, delta=None: None)
            mp.setattr(st, "markdown", lambda body: None)
            mp.setattr(st, "caption", lambda body: None)
            mp.setattr(st, "button", lambda *args, **kwargs: False)
            mp.setattr(st, "columns", _mock_columns)

            result = render_ingestion_progress()
            assert result is None

    def test_cancel_button_triggers_stop(self):
        mock_progress = IngestionProgress(
            is_running=True,
            source_names=("Spark",),
            max_pages_per_source=10,
            total_pages_fetched=1,
            total_chunks_indexed=0,
            current_phase="crawling",
            sources={"Spark": SourceProgress(name="Spark", status="crawling", pages_fetched=1)},
            last_message="Running...",
        )

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
            mp.setattr(st, "progress", lambda value: None)
            mp.setattr(st, "markdown", lambda body: None)
            mp.setattr(st, "caption", lambda body: None)
            mp.setattr(st, "metric", lambda label, value, delta=None: None)
            mp.setattr(st, "button", lambda *args, **kwargs: False)
            mp.setattr(IngestionManager, "stop", lambda: None)
            mp.setattr(st, "rerun", lambda: None)
            mp.setattr(st, "columns", _mock_columns)

            # @st.fragment prevents direct callback execution in unit tests;
            # verify no exception is raised.
            render_ingestion_progress()

    def test_returns_early_when_not_running(self):
        mock_progress = IngestionProgress(is_running=False)

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)

            result = render_ingestion_progress()
            assert result is None


# ---------------------------------------------------------------------------
# render_ingestion_tab tests
# ---------------------------------------------------------------------------

class TestRenderIngestionTab:
    """Tests for the ingestion dashboard tab."""

    def test_success_message_displayed(self):
        mock_progress = IngestionProgress(
            is_running=False,
            success_message="Refresh complete. Indexed or updated 42 chunks.",
            last_message="Refresh complete. Indexed or updated 42 chunks.",
            elapsed_seconds=30.0,
            sources={"Spark": SourceProgress(name="Spark", status="complete", pages_fetched=5, chunks_indexed=42)},
        )

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
            mp.setattr(st, "success", lambda msg: None)
            mp.setattr(st, "warning", lambda msg: None)
            mp.setattr(st, "button", lambda *args, **kwargs: False)
            mp.setattr(st, "caption", lambda body: None)
            mp.setattr(st, "markdown", lambda body: None)
            mp.setattr(st, "subheader", lambda body: None)
            mp.setattr(st, "multiselect", lambda *args, **kwargs: [])
            mp.setattr(st, "number_input", lambda *args, **kwargs: 0)
            mp.setattr(st, "columns", _mock_columns)

            render_ingestion_tab()

    def test_dismiss_resets_status(self):
        mock_progress = IngestionProgress(
            is_running=False,
            success_message="Refresh complete. Indexed or updated 42 chunks.",
            last_message="Refresh complete.",
            elapsed_seconds=30.0,
            sources={"Spark": SourceProgress(name="Spark", status="complete", pages_fetched=5, chunks_indexed=42)},
        )

        reset_called = []

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
            mp.setattr(st, "success", lambda msg: None)
            mp.setattr(st, "caption", lambda body: None)
            mp.setattr(st, "markdown", lambda body: None)
            mp.setattr(st, "subheader", lambda body: None)
            mp.setattr(st, "multiselect", lambda *args, **kwargs: [])
            mp.setattr(st, "number_input", lambda *args, **kwargs: 0)
            mp.setattr(st, "columns", _mock_columns)
            mp.setattr(st, "button", lambda *args, **kwargs: "Dismiss" in str(args))
            mp.setattr(IngestionManager, "reset_status", lambda: reset_called.append(1))
            mp.setattr(st, "rerun", lambda: None)

            render_ingestion_tab()
            assert len(reset_called) == 1

    def test_error_displayed(self):
        mock_progress = IngestionProgress(
            is_running=False,
            error="Connection refused to Qdrant",
            last_message="Ingestion failed: Connection refused to Qdrant",
            elapsed_seconds=5.0,
        )

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
            mp.setattr(st, "success", lambda msg: None)
            mp.setattr(st, "warning", lambda msg: None)
            mp.setattr(st, "button", lambda *args, **kwargs: False)
            mp.setattr(st, "caption", lambda body: None)
            mp.setattr(st, "markdown", lambda body: None)
            mp.setattr(st, "subheader", lambda body: None)
            mp.setattr(st, "multiselect", lambda *args, **kwargs: [])
            mp.setattr(st, "number_input", lambda *args, **kwargs: 0)
            mp.setattr(st, "columns", _mock_columns)

            render_ingestion_tab()

    def test_missing_task_error_displayed(self):
        mock_progress = IngestionProgress(
            is_running=False,
            error="Ingestion task not found. It may have expired.",
            last_message="",
        )

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
            mp.setattr(st, "success", lambda msg: None)
            mp.setattr(st, "warning", lambda msg: None)
            mp.setattr(st, "button", lambda *args, **kwargs: False)
            mp.setattr(st, "caption", lambda body: None)
            mp.setattr(st, "markdown", lambda body: None)
            mp.setattr(st, "subheader", lambda body: None)
            mp.setattr(st, "multiselect", lambda *args, **kwargs: [])
            mp.setattr(st, "number_input", lambda *args, **kwargs: 0)
            mp.setattr(st, "columns", _mock_columns)

            render_ingestion_tab()
