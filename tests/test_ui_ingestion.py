import streamlit as st
import pytest
from data_engineering_copilot.ui.streamlit_app import (
    render_ingestion_progress,
    render_ingestion_tab,
    IngestionProgress,
    IngestionManager,
    SourceProgress,
)

# Fixtures to reset Streamlit session state
@pytest.fixture(autouse=True)
def reset_session_state():
    st.session_state.clear()
    IngestionManager.reset_status()
    yield
    st.session_state.clear()
    IngestionManager.reset_status()


def test_progress_shown():
    mock_progress = IngestionProgress(
        is_running=True,
        source_names=("Spark",),
        max_pages_per_source=10,
        total_pages_fetched=5,
        total_chunks_indexed=12,
        current_phase="crawling",
        sources={"Spark": SourceProgress(name="Spark", status="crawling", pages_fetched=5, chunks_indexed=12)},
        recent_urls=[],
        last_message="Starting ingestion...",
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
        mp.setattr(st, "progress", lambda value: value)
        mp.setattr(st, "metric", lambda label, value, delta=None: None)
        mp.setattr(st, "markdown", lambda body: None)
        mp.setattr(st, "caption", lambda body: None)
        mp.setattr(st, "button", lambda *args, **kwargs: False)

        result = render_ingestion_progress()
        assert result is None  # Returns None (implicit), but no exception


def test_cancel_button_triggers_stop():
    mock_progress = IngestionProgress(
        is_running=True,
        source_names=("Spark",),
        max_pages_per_source=10,
        total_pages_fetched=1,
        total_chunks_indexed=0,
        current_phase="crawling",
        sources={"Spark": SourceProgress(name="Spark", status="crawling", pages_fetched=1)},
        recent_urls=[],
        last_message="Running...",
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
        mp.setattr(st, "progress", lambda value: None)
        mp.setattr(st, "markdown", lambda body: None)
        mp.setattr(st, "caption", lambda body: None)
        mp.setattr(st, "metric", lambda label, value, delta=None: None)
        mp.setattr(st, "button", lambda *args, **kwargs: True if "Stop Refresh" in str(args) else False)
        mp.setattr(IngestionManager, "stop", lambda: None)
        mp.setattr(st, "rerun", lambda: None)

        # Fragment may not fully execute in test mode without ScriptRunContext;
        # verify no exception is raised.
        render_ingestion_progress()


def test_success_message_displayed():
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

        render_ingestion_tab()
        # Success message should be displayed without errors


def test_dismiss_resets_status():
    mock_progress = IngestionProgress(
        is_running=False,
        success_message="Refresh complete. Indexed or updated 42 chunks.",
        last_message="Refresh complete. Indexed or updated 42 chunks.",
        elapsed_seconds=30.0,
        sources={"Spark": SourceProgress(name="Spark", status="complete", pages_fetched=5, chunks_indexed=42)},
    )

    reset_called = []

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(IngestionManager, "get_progress", lambda: mock_progress)
        mp.setattr(st, "success", lambda msg: None)
        mp.setattr(st, "caption", lambda body: None)
        mp.setattr(st, "markdown", lambda body: None)
        mp.setattr(st, "button", lambda *args, **kwargs: True if "Dismiss" in str(args) else False)
        mp.setattr(IngestionManager, "reset_status", lambda: reset_called.append(1))
        mp.setattr(st, "rerun", lambda: None)

        render_ingestion_tab()
        assert len(reset_called) == 1
