from __future__ import annotations
from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import Callable
import threading

from dataclasses import dataclass, field
import streamlit as st
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import IngestionEvent
from data_engineering_copilot.factory import build_ingestion_service, build_rag_service
from data_engineering_copilot.infrastructure.vector_store import ChromaVectorStore, VectorStoreReadError
from data_engineering_copilot.logging_config import configure_logging

if settings.logging_enabled:
    configure_logging(settings.project_root)

logger = logging.getLogger(__name__)

@st.cache_resource
def rag_service():
    logger.info("Streamlit cached RAG service requested")
    return build_rag_service()

@st.cache_resource
def vector_store():
    logger.info("Streamlit cached vector store requested")
    return ChromaVectorStore(str(settings.chroma_dir), settings.collection_name)

_log_lock = threading.Lock()

class IngestionCancelledError(Exception):
    """Raised when ingestion is cancelled by the user."""

@dataclass
class IngestionProgress:
    is_running: bool = False
    source_names: tuple[str, ...] = ()
    max_pages: int = 0
    pages_fetched: int = 0
    chunks_indexed: int = 0
    recent_urls: list[str] = field(default_factory=list)
    last_message: str = ""
    error: str | None = None
    success_message: str | None = None
    cancel_requested: bool = False
    pages_by_source: dict[str, int] = field(default_factory=dict)
    chunks_by_source: dict[str, int] = field(default_factory=dict)

class IngestionManager:
    _lock = threading.Lock()
    _progress = IngestionProgress()
    _thread: threading.Thread | None = None

    @classmethod
    def get_progress(cls) -> IngestionProgress:
        with cls._lock:
            logger.info("GET module=%s progress_id=%s running=%s", __name__, id(cls._progress), cls._progress.is_running)
            p = cls._progress
            return IngestionProgress(
                is_running=p.is_running,
                source_names=p.source_names,
                max_pages=p.max_pages,
                pages_fetched=p.pages_fetched,
                chunks_indexed=p.chunks_indexed,
                recent_urls=list(p.recent_urls),
                last_message=p.last_message,
                error=p.error,
                success_message=p.success_message,
                cancel_requested=p.cancel_requested,
                pages_by_source=dict(p.pages_by_source),
                chunks_by_source=dict(p.chunks_by_source),
            )

    @classmethod
    def is_running(cls) -> bool:
        with cls._lock:
            return cls._progress.is_running

    @classmethod
    def start(cls, max_pages_per_source: int, source_names: tuple[str, ...]) -> bool:
        with cls._lock:
            logger.info("START: is_running=%s", cls._progress.is_running)
            logger.info("START CALLED")
            logger.info("PID=%s", os.getpid())
            logger.info("START module=%s progress_id=%s", __name__, id(cls._progress))
            if cls._progress.is_running:
                return False
            cls._progress = IngestionProgress(
                is_running=True,
                source_names=source_names,
                max_pages=max_pages_per_source,
                recent_urls=[],
                last_message="Starting ingestion...",
                pages_by_source={},
                chunks_by_source={},
            )
            cls._thread = threading.Thread(
                target=cls._run_ingestion,
                args=(max_pages_per_source, source_names),
                daemon=True,
            )
            cls._thread.start()
            return True

    @classmethod
    def stop(cls) -> None:
        with cls._lock:
            if cls._progress.is_running:
                cls._progress.cancel_requested = True
                cls._progress.last_message = "Stopping ingestion..."

    @classmethod
    def reset_status(cls) -> None:
        with cls._lock:
            # Always reset to a fresh progress object, regardless of running state.
            # This prevents stale state from persisting across tests or UI reloads.
            cls._progress = IngestionProgress()

    @classmethod
    def _run_ingestion(cls, max_pages: int, source_names: tuple[str, ...]) -> None:
        logger.info("THREAD ENTERED")
        logger.info("THREAD STARTED PID=%s", os.getpid())
        log_path = cls.ingestion_log_path()

        def handle_event(event: IngestionEvent) -> None:
            with cls._lock:
                logger.info("EVENT RECEIVED pages=%s chunks=%s", event.pages_fetched, event.chunks_indexed)
                logger.info("EVENT %s pages=%s chunks=%s", event.event_type, event.pages_fetched, event.chunks_indexed)
                if cls._progress.cancel_requested:
                    raise IngestionCancelledError("Ingestion cancelled by user.")
                # Use the static method to log the event
                cls.append_ingestion_log(log_path, event)

            if event.event_type == "fetch_success":
                cls._progress.pages_by_source[event.source_name] = event.pages_fetched
            elif event.event_type == "page_indexed":
                cls._progress.chunks_by_source[event.source_name] = (
                    cls._progress.chunks_by_source.get(event.source_name, 0) + event.chunks_indexed
                )

            if event.url and event.event_type in {"fetch_start", "fetch_success", "fetch_error"}:
                label = "fetching" if event.event_type == "fetch_start" else event.event_type.replace("_", " ")
                cls._progress.recent_urls.insert(0, f"- `{label}` [{event.url}]({event.url})")
                del cls._progress.recent_urls[25:]

            cls._progress.pages_fetched = sum(cls._progress.pages_by_source.values())
            cls._progress.chunks_indexed = sum(cls._progress.chunks_by_source.values())
            cls._progress.last_message = event.message

        try:
            service = build_ingestion_service()
            total_chunks = service.ingest(
                max_pages_per_source=max_pages,
                source_names=source_names,
                on_event=handle_event,
            )
            rag_service.clear()
            vector_store.clear()
            with cls._lock:
                cls._progress.is_running = False
                cls._progress.success_message = f"Refresh complete. Indexed or updated {total_chunks} chunks."
                cls._progress.last_message = f"Refresh complete. Indexed or updated {total_chunks} chunks."
        except IngestionCancelledError:
            with cls._lock:
                cls._progress.is_running = False
                cls._progress.error = "Ingestion cancelled."
                cls._progress.last_message = "Ingestion cancelled by user."
            logger.info("Ingestion cancelled in background thread.")
        except Exception as exc:
            logger.exception("Ingestion failed in background thread")
            with cls._lock:
                cls._progress.is_running = False
                cls._progress.error = str(exc)
                cls._progress.last_message = f"Ingestion failed: {exc}"

    def run_ingestion_refresh(
        max_pages_per_source: int,
        source_names: tuple[str, ...],
        on_event: Callable[[IngestionEvent], None] | None = None,
    ) -> int:
        logger.info("Streamlit ingestion refresh started max_pages=%s sources=%s", max_pages_per_source, source_names)
        service = build_ingestion_service()
        total_chunks = service.ingest(
            max_pages_per_source=max_pages_per_source,
            source_names=source_names,
            on_event=on_event,
        )
        logger.info("Streamlit ingestion refresh completed chunks=%s", total_chunks)
        return total_chunks

    @staticmethod
    def ingestion_log_path() -> Path:
        """Return the path to the ingestion log file."""
        return settings.project_root / "logs" / "ingestion_refresh.log"

    @staticmethod
    def append_ingestion_log(log_path: Path, event: IngestionEvent) -> None:
        if not settings.logging_enabled:
            return
        with _log_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat(timespec="seconds")
            parts = [
                timestamp,
                f"event={event.event_type}",
                f"source={event.source_name}",
            ]
            if event.url:
                parts.append(f"url={event.url}")
            if event.title:
                parts.append(f"title={event.title}")
            if event.pages_fetched:
                parts.append(f"pages_fetched={event.pages_fetched}")
            if event.chunks_indexed:
                parts.append(f"chunks_indexed={event.chunks_indexed}")
            if event.error:
                parts.append(f"error={event.error}")
            parts.append(f"message={event.message}")
            with log_path.open("a", encoding="utf-8") as file:
                file.write(" | ".join(parts) + "\n")
            logger.info("Ingestion UI event logged event=%s source=%s url=%s", event.event_type, event.source_name, event.url)

@st.fragment(run_every=1.0)
def render_ingestion_section(source_names: tuple[str, ...]) -> None:
    if "last_run_active" not in st.session_state:
        st.session_state.last_run_active = False
    if "ingestion_started" not in st.session_state:
        st.session_state.ingestion_started = False

    progress = IngestionManager.get_progress()
    st.write("is_running =", progress.is_running)

    if progress.is_running:
        st.session_state.last_run_active = True
        st.session_state.ingestion_started = True
        if progress.last_message and "Starting ingestion" in progress.last_message:
            st.success("Ingestion started…")
        st.subheader("Ingestion Active")
        if st.button("Stop Refresh", type="primary", use_container_width=True, key="stop_refresh_btn", help="Cancel the current ingestion"):
            IngestionManager.stop()
            st.rerun()
        total_expected = progress.max_pages * len(progress.source_names) or 1
        percent = min(progress.pages_fetched / total_expected, 1.0)
        # Use a stable placeholder for the progress bar to avoid unsupported `key` argument.
        if "ingest_progress_placeholder" not in st.session_state:
            st.session_state.ingest_progress_placeholder = st.empty()
        st.session_state.ingest_progress_placeholder.progress(percent)
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages fetched", f"{progress.pages_fetched}/{total_expected}")
        col2.metric("Chunks indexed", f"{progress.chunks_indexed}")
        col3.metric("Sources", f"{len(progress.source_names)}")
        st.info(progress.last_message)
        if progress.recent_urls:
            with st.expander("Recent HTML page URLs", expanded=True):
                st.markdown("\n".join(progress.recent_urls))
    else:
        if st.session_state.last_run_active:
            st.session_state.last_run_active = False
            st.rerun()
        if progress.success_message:
            st.success(progress.success_message)
            st.caption(f"Finished at {datetime.now().strftime('%H:%M:%S')}")
            if st.button("Dismiss", key="dismiss_success_btn"):
                IngestionManager.reset_status()
                st.session_state.ingestion_started = False
                st.rerun()
        elif progress.error:
            st.warning(progress.last_message)
            if st.button("Dismiss", key="dismiss_error_btn"):
                IngestionManager.reset_status()
                st.session_state.ingestion_started = False
                st.rerun()

def main() -> None:
    logger.info("Streamlit app render started")
    st.set_page_config(page_title="DataEngineeringCopilot", layout="wide")
    st.title("DataEngineeringCopilot")
    st.caption("Offline RAG over Spark, Airflow, Databricks, and Delta Lake documentation.")
    progress = IngestionManager.get_progress()
    st.sidebar.write("PID:", os.getpid())
    st.sidebar.write("Running:", progress.is_running)
    st.sidebar.write("Chunks:", progress.chunks_indexed)
    st.sidebar.write("Pages:", progress.pages_fetched)
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("Repository")
        try:
            chunk_count = vector_store().count()
        except VectorStoreReadError as exc:
            chunk_count = 0
            logger.exception("Streamlit sidebar vector count failed")
        st.write(f"Chunks indexed: {chunk_count}")
        st.write(f"Ollama model: `{settings.ollama_model}`")
        st.write(f"Ollama timeout: `{settings.ollama_timeout_seconds}s`")
        st.write(f"Ollama output limit: `{settings.ollama_num_predict}` tokens")
        st.write(f"Embedding model: `{settings.embedding_model_name}`")
        st.write(f"Confidence threshold: `{settings.confidence_threshold}`")
        if settings.logging_enabled:
            st.write(f"Application log: `{settings.project_root / 'logs' / 'application.log'}`")
        else:
            st.write("Application logging is disabled.")
        st.divider()
        st.subheader("Ingestion")
        # UI controls for ingestion
        selected_sources = st.multiselect(
            "Select sources to ingest",
            options=[source.name for source in settings.sources],
            default=[source.name for source in settings.sources],
        )
        max_pages = st.number_input(
            "Max pages per source (0 = unlimited)",
            min_value=0,
            value=0,
            step=1,
        )
        if st.button("Refresh Documentation", type="primary"):
            if selected_sources:
                started = IngestionManager.start(
                    max_pages_per_source=int(max_pages),
                    source_names=tuple(selected_sources),
                )
                if not started:
                    st.warning("Ingestion is already running.")
            else:
                st.warning("Please select at least one source.")
        render_ingestion_section(tuple(selected_sources))
    with col_right:
        question = st.text_area(
            "Question",
            placeholder="How do I configure Spark dynamic allocation?",
            height=120,
        )
        ask = st.button("Ask", type="primary")
        if ask:
            if not question.strip():
                logger.info("Streamlit ask ignored because question was empty")
                st.warning("Enter a question.")
            else:
                logger.info("Streamlit ask started question=%r", question.strip()[:200])
                with st.spinner("Searching local repository and asking Ollama..."):
                    answer = rag_service().answer(question.strip())
                logger.info("Streamlit ask completed confidence=%.4f sources=%s answer_chars=%s", answer.confidence, len(answer.sources), len(answer.text))
                st.subheader("Answer")
                st.write(answer.text)

if __name__ == "__main__":
    main()