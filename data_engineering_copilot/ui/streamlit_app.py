from __future__ import annotations
from datetime import datetime
import json
import logging
import socket
import sys
import urllib.request
import urllib.error
from pathlib import Path
import threading
import time as _time

from dataclasses import dataclass, field
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import IngestionEvent
from data_engineering_copilot.factory import build_ingestion_service, build_rag_service
from data_engineering_copilot.infrastructure.vector_store import QdrantVectorStore
from data_engineering_copilot.services.metrics import MetricsCollector
from logger_config import setup_logging

if settings.logging_enabled:
    setup_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service health checks
# ---------------------------------------------------------------------------

def _check_qdrant_reachable(timeout: float = 2.0) -> tuple[bool, str]:
    """Check if Qdrant is reachable. Returns (ok, message)."""
    try:
        url = f"{settings.qdrant_url}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True, f"Qdrant is running at {settings.qdrant_url}"
            return False, f"Qdrant returned HTTP {resp.status}"
    except urllib.error.URLError:
        return False, (
            f"Qdrant is not reachable at {settings.qdrant_url}.\n\n"
            "**Start it with:**\n```\ndocker compose up -d qdrant\n```"
        )
    except (socket.timeout, OSError) as exc:
        return False, f"Qdrant connection failed: {exc}"


def _check_ollama_reachable(timeout: float = 2.0) -> tuple[bool, str]:
    """Check if Ollama is reachable. Returns (ok, message)."""
    try:
        url = f"{settings.ollama_base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                has_embed = any(settings.embedding_model_name in m for m in models)
                has_llm = any(settings.ollama_model in m for m in models)
                missing = []
                if not has_embed:
                    missing.append(settings.embedding_model_name)
                if not has_llm:
                    missing.append(settings.ollama_model)
                if missing:
                    return False, (
                        f"Ollama is running but missing models: **{', '.join(missing)}**\n\n"
                        "**Pull them with:**\n```\n"
                        + "\n".join(f"ollama pull {m}" for m in missing)
                        + "\n```"
                    )
                return True, "Ollama is running with all required models"
            return False, f"Ollama returned HTTP {resp.status}"
    except urllib.error.URLError:
        return False, (
            f"Ollama is not reachable at {settings.ollama_base_url}.\n\n"
            "**Start it with:**\n```\nollama serve\n```\n\n"
            "Then pull the required models:\n```\n"
            f"ollama pull {settings.embedding_model_name}\n"
            f"ollama pull {settings.ollama_model}\n```"
        )
    except (socket.timeout, OSError) as exc:
        return False, f"Ollama connection failed: {exc}"


def _check_langfuse_reachable(timeout: float = 2.0) -> tuple[bool, str]:
    """Check if Langfuse is reachable. Returns (ok, message)."""
    try:
        url = f"{settings.langfuse_host}/api/public/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "OK":
                    return True, "Langfuse is running"
                return False, f"Langfuse health returned status: {data.get('status')}"
            return False, f"Langfuse returned HTTP {resp.status}"
    except (urllib.error.URLError, socket.timeout, OSError):
        return False, (
            "Langfuse is not reachable. Tracing will be disabled.\n\n"
            "**Start it with:**\n```\ndocker compose up -d langfuse langfuse-postgres clickhouse minio\n```"
        )

@st.cache_resource
def _build_rag_service():
    return build_rag_service()

@st.cache_resource
def _build_vector_store():
    return QdrantVectorStore(settings.qdrant_url, settings.collection_name)

def rag_service():
    """Return cached RAG service, or None if Qdrant/Ollama are unavailable."""
    try:
        return _build_rag_service()
    except Exception as exc:
        logger.warning("Failed to create RAG service: %s", exc)
        return None

def vector_store():
    """Return cached vector store, or None if Qdrant is unavailable."""
    try:
        return _build_vector_store()
    except Exception as exc:
        logger.warning("Failed to create vector store: %s", exc)
        return None

_log_lock = threading.Lock()

class IngestionCancelledError(Exception):
    """Raised when ingestion is cancelled by the user."""

@dataclass
class SourceProgress:
    name: str
    status: str = "pending"
    pages_fetched: int = 0
    pages_skipped: int = 0
    chunks_indexed: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class IngestionProgress:
    is_running: bool = False
    start_time: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    source_names: tuple[str, ...] = ()
    max_pages_per_source: int = 0
    current_phase: str = "idle"
    total_pages_fetched: int = 0
    total_pages_skipped: int = 0
    total_errors: int = 0
    total_chunks_indexed: int = 0
    sources: dict[str, SourceProgress] = field(default_factory=dict)
    recent_events: list[IngestionEvent] = field(default_factory=list)
    recent_urls: list[str] = field(default_factory=list)
    last_message: str = ""
    error: str | None = None
    success_message: str | None = None
    cancel_requested: bool = False

class IngestionManager:
    _lock = threading.Lock()
    _progress = IngestionProgress()
    _thread: threading.Thread | None = None

    @classmethod
    def get_progress(cls) -> IngestionProgress:
        with cls._lock:
            p = cls._progress
            return IngestionProgress(
                is_running=p.is_running,
                start_time=p.start_time,
                elapsed_seconds=p.elapsed_seconds,
                estimated_remaining_seconds=p.estimated_remaining_seconds,
                source_names=p.source_names,
                max_pages_per_source=p.max_pages_per_source,
                current_phase=p.current_phase,
                total_pages_fetched=p.total_pages_fetched,
                total_pages_skipped=p.total_pages_skipped,
                total_errors=p.total_errors,
                total_chunks_indexed=p.total_chunks_indexed,
                sources=dict(p.sources),
                recent_events=list(p.recent_events),
                recent_urls=list(p.recent_urls),
                last_message=p.last_message,
                error=p.error,
                success_message=p.success_message,
                cancel_requested=p.cancel_requested,
            )

    @classmethod
    def is_running(cls) -> bool:
        with cls._lock:
            return cls._progress.is_running

    @classmethod
    def start(cls, max_pages_per_source: int, source_names: tuple[str, ...]) -> bool:
        with cls._lock:
            logger.info("START: is_running=%s", cls._progress.is_running)
            if cls._progress.is_running:
                return False
            import time
            cls._progress = IngestionProgress(
                is_running=True,
                start_time=time.time(),
                source_names=source_names,
                max_pages_per_source=max_pages_per_source,
                current_phase="crawling",
                sources={
                    name: SourceProgress(name=name, status="pending")
                    for name in source_names
                },
                recent_urls=[],
                last_message="Starting ingestion...",
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
        log_path = cls.ingestion_log_path()
        import time

        def handle_event(event: IngestionEvent) -> None:
            # Log outside the lock to avoid I/O blocking progress updates
            if settings.logging_enabled:
                cls.append_ingestion_log(log_path, event)

            with cls._lock:
                if cls._progress.cancel_requested:
                    raise IngestionCancelledError("Ingestion cancelled by user.")

                elapsed = time.time() - cls._progress.start_time
                cls._progress.elapsed_seconds = elapsed
                cls._progress.current_phase = event.current_phase or cls._progress.current_phase
                cls._progress.last_message = event.message

                # Update per-source progress
                src = cls._progress.sources.get(event.source_name)
                if src is None:
                    src = SourceProgress(name=event.source_name)
                    cls._progress.sources[event.source_name] = src

                if event.event_type == "source_start":
                    src.status = "crawling"
                elif event.event_type == "fetch_start":
                    src.status = "crawling"
                elif event.event_type == "fetch_success":
                    src.pages_fetched = event.pages_fetched
                elif event.event_type == "fetch_error":
                    src.errors += 1
                elif event.event_type == "page_indexed":
                    src.chunks_indexed += event.chunks_indexed
                elif event.event_type == "page_skipped":
                    src.pages_skipped += 1
                elif event.event_type == "source_complete":
                    src.status = "complete"
                    src.pages_fetched = event.pages_fetched
                    src.chunks_indexed = event.chunks_indexed
                    src.elapsed_seconds = elapsed

                src.elapsed_seconds = elapsed

                # Aggregate global totals
                cls._progress.total_pages_fetched = sum(
                    s.pages_fetched for s in cls._progress.sources.values()
                )
                cls._progress.total_pages_skipped = sum(
                    s.pages_skipped for s in cls._progress.sources.values()
                )
                cls._progress.total_errors = sum(
                    s.errors for s in cls._progress.sources.values()
                )
                cls._progress.total_chunks_indexed = sum(
                    s.chunks_indexed for s in cls._progress.sources.values()
                )

                # Estimate remaining time
                finished_sources = sum(
                    1 for s in cls._progress.sources.values() if s.status == "complete"
                )
                if finished_sources > 0:
                    avg_seconds_per_source = elapsed / finished_sources
                    remaining_sources = len(cls._progress.source_names) - finished_sources
                    cls._progress.estimated_remaining_seconds = avg_seconds_per_source * remaining_sources

                # URL tracking for fetches
                if event.url and event.event_type in {"fetch_start", "fetch_success", "fetch_error"}:
                    label = "fetching" if event.event_type == "fetch_start" else event.event_type.replace("_", " ")
                    cls._progress.recent_urls.insert(0, f"- `{label}` [{event.url}]({event.url})")
                    del cls._progress.recent_urls[25:]

                # Recent events for live feed
                cls._progress.recent_events.insert(0, event)
                del cls._progress.recent_events[50:]

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
                cls._progress.current_phase = "complete"
                cls._progress.elapsed_seconds = time.time() - cls._progress.start_time
                cls._progress.success_message = f"Refresh complete. Indexed or updated {total_chunks} chunks."
                cls._progress.last_message = cls._progress.success_message
        except IngestionCancelledError:
            with cls._lock:
                cls._progress.is_running = False
                cls._progress.current_phase = "cancelled"
                cls._progress.elapsed_seconds = time.time() - cls._progress.start_time
                cls._progress.error = "Ingestion cancelled."
                cls._progress.last_message = "Ingestion cancelled by user."
            logger.info("Ingestion cancelled in background thread.")
        except Exception as exc:
            logger.exception("Ingestion failed in background thread")
            error_msg = str(exc)
            if "Connection refused" in error_msg or "connect" in error_msg.lower():
                error_msg = (
                    "Could not connect to a required service.\n\n"
                    "**Ensure these services are running:**\n"
                    "- Qdrant: `docker compose up -d qdrant`\n"
                    "- Ollama: `ollama serve`"
                )
            with cls._lock:
                cls._progress.is_running = False
                cls._progress.current_phase = "error"
                cls._progress.elapsed_seconds = time.time() - cls._progress.start_time
                cls._progress.error = error_msg
                cls._progress.last_message = f"Ingestion failed: {error_msg}"

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

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


@st.fragment(run_every=1.0)
def render_ingestion_progress() -> None:
    """Fragment that auto-refreshes every second to show live progress."""
    progress = IngestionManager.get_progress()

    if not progress.is_running:
        return

    if "ingestion_started" not in st.session_state:
        st.session_state.ingestion_started = True

    # Phase badge
    phase_colors = {
        "crawling": "blue",
        "embedding": "orange",
        "indexing": "green",
        "complete": "green",
        "error": "red",
        "cancelled": "gray",
    }
    color = phase_colors.get(progress.current_phase, "gray")
    st.markdown(
        f"**Phase:** :{color}[{progress.current_phase.upper()}]  |  "
        f"**Elapsed:** {_format_duration(progress.elapsed_seconds)}  |  "
        f"**ETA:** {_format_duration(progress.estimated_remaining_seconds) if progress.estimated_remaining_seconds > 0 else '...'}",
    )

    # Overall progress bar: use total pages vs an estimate
    finished_sources = sum(1 for s in progress.sources.values() if s.status == "complete")
    total_sources = len(progress.source_names) or 1
    source_progress = finished_sources / total_sources
    # Weighted blend: source completion contributes 70%, pages contribute 30% within current source
    estimated_pages = progress.max_pages_per_source * total_sources or 1
    page_ratio = min(progress.total_pages_fetched / max(estimated_pages, 1), 1.0)
    overall_percent = source_progress * 0.70 + page_ratio * 0.30
    st.progress(overall_percent)

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pages Fetched", progress.total_pages_fetched)
    col2.metric("Pages Skipped", progress.total_pages_skipped)
    col3.metric("Chunks Indexed", progress.total_chunks_indexed)
    col4.metric("Errors", progress.total_errors, delta=None if progress.total_errors == 0 else progress.total_errors)

    # Per-source cards
    if progress.sources:
        st.subheader("Source Details")
        cols_per_row = 2
        source_items = list(progress.sources.items())
        for i in range(0, len(source_items), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(row_cols):
                idx = i + j
                if idx >= len(source_items):
                    break
                name, src = source_items[idx]
                with col:
                    status_icon = {
                        "pending": "⏳",
                        "crawling": "🔄",
                        "complete": "✅",
                    }.get(src.status, "❓")
                    st.markdown(f"**{status_icon} {name}**")
                    st.progress(min(src.pages_fetched / max(progress.max_pages_per_source, 1), 1.0))
                    st.caption(
                        f"Pages: {src.pages_fetched} | Skipped: {src.pages_skipped} | "
                        f"Chunks: {src.chunks_indexed} | Errors: {src.errors}"
                    )

    # Live event feed
    if progress.recent_events:
        with st.expander("Live Event Feed", expanded=False):
            for event in progress.recent_events[:10]:
                ts = _format_duration(event.timestamp)
                st.caption(f"`[{ts}]` {event.event_type}: {event.message}")

    # Stop button
    if st.button("🛑 Stop Refresh", type="primary", use_container_width=True, key="stop_refresh_btn"):
        IngestionManager.stop()
        st.rerun()


def render_qa_tab() -> None:
    """Q&A tab: ask questions against the knowledge base."""
    st.subheader("Ask a Question")

    # Pre-flight: check services before showing the input
    qdrant_ok, qdrant_msg = _check_qdrant_reachable()
    ollama_ok, ollama_msg = _check_ollama_reachable()

    if not qdrant_ok or not ollama_ok:
        if not qdrant_ok:
            st.error(f"**Qdrant unavailable**\n\n{qdrant_msg}")
        if not ollama_ok:
            st.error(f"**Ollama unavailable**\n\n{ollama_msg}")
        st.info("Fix the issues above and refresh the page to use Q&A.")
        return

    question = st.text_area(
        "Question",
        placeholder="How do I configure Spark dynamic allocation?",
        height=120,
        key="qa_question",
    )
    ask = st.button("Ask", type="primary", key="qa_ask_btn")
    if ask:
        if not question.strip():
            st.warning("Enter a question.")
        else:
            service = rag_service()
            if service is None:
                st.error(
                    "Could not connect to the RAG service.\n\n"
                    "**Check that Qdrant and Ollama are running.**\n"
                    "See the **System Health** tab for details."
                )
                return

            logger.info("Streamlit ask started question=%r", question.strip()[:200])
            with st.spinner("Searching local repository and asking Ollama..."):
                try:
                    answer = service.answer(question.strip())
                except Exception as exc:
                    logger.exception("RAG answer failed")
                    st.error(
                        f"**Failed to get answer:** {exc}\n\n"
                        "**Possible causes:**\n"
                        "- Ollama may have timed out or the model is still loading\n"
                        "- Qdrant may have lost connectivity\n\n"
                        "Check the **System Health** tab and try again."
                    )
                    return

            logger.info(
                "Streamlit ask completed confidence=%.4f sources=%s answer_chars=%s",
                answer.confidence,
                len(answer.sources),
                len(answer.text),
            )

            # Record metrics
            collector: MetricsCollector = st.session_state.metrics_collector
            collector.record_query(
                query=question.strip(),
                retrieved_chunks=[],
                answer=answer,
                was_answered=True,
            )

            st.subheader("Answer")
            st.write(answer.text)
            st.caption(f"Confidence: {answer.confidence:.2%}")

            if answer.sources:
                with st.expander(f"Sources ({len(answer.sources)})", expanded=False):
                    for i, source in enumerate(answer.sources, 1):
                        st.markdown(f"**{i}. [{source.title}]({source.url})**")
                        st.caption(f"Source: {source.source_name}")

            # Per-answer detailed metrics
            with st.expander("Answer Metrics", expanded=False):
                qm = collector.queries[-1] if collector.queries else None
                if qm:
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.metric("Query Difficulty", qm.query_difficulty.capitalize())
                        st.metric("Query Length (words)", qm.query_length)
                    with col_a2:
                        st.metric("Answer Length (words)", qm.answer_metrics.answer_length if qm.answer_metrics else "N/A")
                        st.metric("Sources Cited", qm.answer_metrics.source_count if qm.answer_metrics else "N/A")

                    if qm.answer_metrics:
                        sec_status = "Yes" if qm.answer_metrics.has_key_sections else "No"
                        unc_status = "Yes" if qm.answer_metrics.has_uncertainty_markers else "No"
                        st.caption(f"Structured sections: {sec_status}  |  Uncertainty markers: {unc_status}")


def render_ingestion_tab() -> None:
    """Ingestion Dashboard tab: controls, progress, and results."""
    st.subheader("Ingestion Controls")

    # Controls row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        selected_sources = st.multiselect(
            "Select sources to ingest",
            options=[source.name for source in settings.sources],
            default=[source.name for source in settings.sources],
            key="ingest_source_select",
        )
    with ctrl_col2:
        max_pages = st.number_input(
            "Max pages per source",
            min_value=0,
            value=settings.max_pages_per_source,
            step=10,
            help="0 = unlimited (capped by config max_pages_per_source)",
            key="ingest_max_pages",
        )
    with ctrl_col3:
        st.write("")  # spacer
        st.write("")  # spacer
        refresh_clicked = st.button("🔄 Refresh Documentation", type="primary", use_container_width=True)

    if refresh_clicked:
        if not selected_sources:
            st.warning("Please select at least one source.")
        else:
            # Pre-flight check
            qdrant_ok, qdrant_msg = _check_qdrant_reachable()
            if not qdrant_ok:
                st.error(f"**Cannot start ingestion**\n\n{qdrant_msg}")
            else:
                started = IngestionManager.start(
                    max_pages_per_source=int(max_pages) if max_pages > 0 else 0,
                    source_names=tuple(selected_sources),
                )
                if not started:
                    st.warning("Ingestion is already running.")
                else:
                    st.success("Ingestion started!")

    # Progress display
    progress = IngestionManager.get_progress()
    if progress.is_running:
        render_ingestion_progress()
    elif progress.success_message:
        st.success(progress.success_message)
        st.caption(f"Finished in {_format_duration(progress.elapsed_seconds)}")
        # Per-source summary
        if progress.sources:
            with st.expander("Per-Source Summary"):
                for name, src in progress.sources.items():
                    st.markdown(
                        f"- **{name}**: {src.pages_fetched} pages, "
                        f"{src.chunks_indexed} chunks, {src.errors} errors"
                    )
        if st.button("Dismiss", key="dismiss_success_btn"):
            IngestionManager.reset_status()
            st.session_state.pop("ingestion_started", None)
            st.rerun()
    elif progress.error:
        st.warning(progress.last_message)
        if st.button("Dismiss", key="dismiss_error_btn"):
            IngestionManager.reset_status()
            st.session_state.pop("ingestion_started", None)
            st.rerun()


def render_health_tab() -> None:
    """System Health tab: vector store, Ollama, configuration."""
    st.subheader("System Health")

    # Live service status
    st.markdown("### Service Status")
    qdrant_ok, qdrant_msg = _check_qdrant_reachable()
    ollama_ok, ollama_msg = _check_ollama_reachable()
    langfuse_ok, langfuse_msg = _check_langfuse_reachable()

    col_q, col_o, col_l = st.columns(3)
    with col_q:
        if qdrant_ok:
            st.success("Qdrant")
            st.caption(qdrant_msg)
        else:
            st.error("Qdrant")
            st.caption(qdrant_msg)
    with col_o:
        if ollama_ok:
            st.success("Ollama")
            st.caption(ollama_msg)
        else:
            st.error("Ollama")
            st.caption(ollama_msg)
    with col_l:
        if langfuse_ok:
            st.success("Langfuse")
            st.caption(langfuse_msg)
        else:
            st.warning("Langfuse")
            st.caption(langfuse_msg)

    st.divider()

    # Repository stats
    st.markdown("### Vector Store")
    store = vector_store()
    if store is not None:
        try:
            chunk_count = store.count()
            st.metric("Total Chunks Indexed", chunk_count)
        except Exception:
            st.warning("Vector store is connected but returned an error.")
            chunk_count = 0
    else:
        chunk_count = 0
        st.warning(
            "Vector store is not available.\n\n"
            f"**Start Qdrant:**\n```\ndocker compose up -d qdrant\n```"
        )

    st.divider()

    # Ollama status
    st.markdown("### Ollama Configuration")
    col_o1, col_o2, col_o3 = st.columns(3)
    col_o1.metric("Model", settings.ollama_model)
    col_o2.metric("Embedding Model", settings.embedding_model_name)
    col_o3.metric("Base URL", settings.ollama_base_url)

    col_o4, col_o5 = st.columns(2)
    col_o4.metric("Timeout", f"{settings.ollama_timeout_seconds}s")
    col_o5.metric("Output Limit", f"{settings.ollama_num_predict} tokens")

    st.divider()

    # RAG Configuration
    st.markdown("### RAG Configuration")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Retrieval Top-K", settings.retrieval_top_k)
    col_r2.metric("Confidence Threshold", f"{settings.confidence_threshold:.0%}")
    col_r3.metric("Max Context Chars", settings.max_context_chars)

    col_r4, col_r5, col_r6 = st.columns(3)
    col_r4.metric("Chunk Strategy", settings.chunking_strategy)
    col_r5.metric("Chunk Size (words)", settings.chunk_size_words)
    col_r6.metric("Overlap (words)", settings.chunk_overlap_words)

    if settings.reranker_enabled:
        col_r7, col_r8 = st.columns(2)
        col_r7.metric("Reranker", "Enabled")
        col_r8.metric("Reranker k", settings.reranker_top_k)

    if settings.logging_enabled:
        st.caption(f"Application log: `{settings.project_root / 'logs' / 'application.log'}`")

    st.divider()

    # Ingestion history
    st.markdown("### Ingestion History")
    log_path = IngestionManager.ingestion_log_path()
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            if lines:
                last_20 = lines[-20:]
                with st.expander(f"Last {len(last_20)} log entries", expanded=False):
                    for line in reversed(last_20):
                        st.caption(line[:200])
            else:
                st.caption("No ingestion history yet.")
        except Exception:
            st.caption("Could not read ingestion log.")
    else:
        st.caption("No ingestion history yet.")

    # Suggested questions for empty state
    if chunk_count == 0:
        st.info("💡 No documents indexed yet. Go to the **Ingestion** tab to crawl documentation sources.")


def render_metrics_tab() -> None:
    """Metrics Dashboard tab: service performance and quality metrics."""
    st.subheader("RAG Service Metrics")

    collector: MetricsCollector = st.session_state.metrics_collector
    summary = collector.get_session_summary()

    if summary["total_queries"] == 0:
        st.info("No queries recorded yet. Ask questions in the **💬 Ask** tab to see metrics.")
        return

    # --- Session Summary Cards ---
    st.markdown("### Session Summary")
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    col_s1.metric("Total Queries", summary["total_queries"])
    col_s2.metric("Answered", summary["answered_queries"])
    col_s3.metric("Answer Rate", f"{summary['answer_rate']:.0%}")
    col_s4.metric("Avg MRR", f"{summary['avg_mrr']:.3f}")
    col_s5.metric("Avg Answer Length (words)", summary["avg_answer_length"])

    st.divider()

    # --- Query Difficulty Breakdown ---
    st.markdown("### Query Difficulty Breakdown")
    by_diff = summary.get("by_difficulty", {})
    if by_diff:
        diff_cols = st.columns(3)
        for col, (difficulty, data) in zip(diff_cols, sorted(by_diff.items())):
            with col:
                st.metric(
                    f"{difficulty.capitalize()}",
                    data["count"],
                    delta=f"{data['answer_rate']:.0%} answered",
                    delta_color="normal",
                )
                # Color coding
                emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
                st.caption(f"{emoji} {data['count']} total queries")

    st.divider()

    # --- Recent Queries Table ---
    st.markdown("### Recent Queries")
    recent = list(reversed(collector.queries[-20:]))  # Most recent first
    if recent:
        table_data = []
        for qm in recent:
            table_data.append({
                "Query": qm.query[:50] + ("..." if len(qm.query) > 50 else ""),
                "Difficulty": qm.query_difficulty.capitalize(),
                "Confidence": f"{qm.confidence_score:.2%}" if qm.was_answered else "—",
                "Answered": "✅" if qm.was_answered else "❌",
                "Sources": qm.answer_metrics.source_count if qm.answer_metrics and qm.answer_metrics.source_count else 0,
                "Answer Words": qm.answer_metrics.answer_length if qm.answer_metrics else "—",
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.divider()

    # --- Confidence Distribution Chart ---
    st.markdown("### Confidence Distribution")
    answered_queries = [q for q in collector.queries if q.was_answered]
    if answered_queries:
        chart_data = {
            "query_idx": list(range(1, len(answered_queries) + 1)),
            "confidence": [q.confidence_score for q in answered_queries],
        }
        st.bar_chart(chart_data, x="query_idx", y="confidence", height=200)
        st.caption("Confidence score per answered query (in chronological order)")
    else:
        st.caption("No answered queries yet.")

    # --- Answer Length Distribution Chart ---
    st.markdown("### Answer Length Distribution")
    queries_with_answers = [q for q in collector.queries if q.answer_metrics]
    if queries_with_answers:
        length_data = {
            "query_idx": list(range(1, len(queries_with_answers) + 1)),
            "words": [q.answer_metrics.answer_length for q in queries_with_answers],
        }
        st.bar_chart(length_data, x="query_idx", y="words", height=200)
        st.caption("Answer length in words per query (in chronological order)")
    else:
        st.caption("No answer data available yet.")

    # Reset button
    st.divider()
    if st.button("Reset Metrics", type="secondary", key="reset_metrics_btn"):
        st.session_state.metrics_collector = MetricsCollector()
        st.rerun()


def main() -> None:
    logger.info("Streamlit app render started")

    # Initialize metrics collector in session state
    if "metrics_collector" not in st.session_state:
        st.session_state.metrics_collector = MetricsCollector()

    st.set_page_config(page_title="DataEngineeringCopilot", layout="wide")
    st.title("📚 DataEngineeringCopilot")
    st.caption("Offline RAG over Spark, Airflow, Databricks, and Delta Lake documentation.")

    # Sidebar: compact status
    progress = IngestionManager.get_progress()
    with st.sidebar:
        st.markdown("### System Status")
        if progress.is_running:
            st.warning(f"Ingestion running ({_format_duration(progress.elapsed_seconds)})")
        else:
            st.success("Idle")

        # Service indicators
        qdrant_ok, _ = _check_qdrant_reachable(timeout=1.0)
        ollama_ok, _ = _check_ollama_reachable(timeout=1.0)
        if qdrant_ok:
            st.success("Qdrant: up")
        else:
            st.error("Qdrant: down")
        if ollama_ok:
            st.success("Ollama: up")
        else:
            st.error("Ollama: down")

        # Chunk count
        store = vector_store()
        if store is not None:
            try:
                chunk_count = store.count()
                st.metric("Chunks in Store", chunk_count)
            except Exception:
                st.metric("Chunks in Store", "error")
        else:
            st.metric("Chunks in Store", "unavailable")

        # Mini metrics summary in sidebar
        collector: MetricsCollector = st.session_state.metrics_collector
        if collector.queries:
            st.markdown("### Session Metrics")
            answered = sum(1 for q in collector.queries if q.was_answered)
            st.metric("Queries Asked", len(collector.queries))
            st.metric("Answered", answered)
            st.caption("Last answer confidence shown in Q&A tab.")

    # Tab layout
    tab_ask, tab_ingest, tab_health, tab_metrics = st.tabs(["💬 Ask", "📥 Ingestion", "🔧 System Health", "📊 Metrics"])
    with tab_ask:
        render_qa_tab()
    with tab_ingest:
        render_ingestion_tab()
    with tab_health:
        render_health_tab()
    with tab_metrics:
        render_metrics_tab()

if __name__ == "__main__":
    main()
