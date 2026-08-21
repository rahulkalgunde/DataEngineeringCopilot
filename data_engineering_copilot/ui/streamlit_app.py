from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering_copilot.config.logging import setup_logging  # noqa: E402
from data_engineering_copilot.config.settings import settings  # noqa: E402
from data_engineering_copilot.domain.models import RawDocument  # noqa: E402
from data_engineering_copilot.factory import build_pipeline_lab, build_rag_service  # noqa: E402
from data_engineering_copilot.observability.langfuse_client import build_trace_url  # noqa: E402
from data_engineering_copilot.services.metrics import MetricsCollector  # noqa: E402
from data_engineering_copilot.ui.components.animations import (  # noqa: E402
    build_diagram_html,
    render_animated_metric,
    render_lottie_badge,
    render_pipeline_diagram,
    render_stepper,
    render_vector_scatter,
    stream_answer_text,
)
from data_engineering_copilot.ui.components.pipeline_states import (  # noqa: E402
    INGESTION_NODES,
    QUERY_NODES,
    NodeState,
    ingestion_node_states,
    reduce_query_node_states,
)

if settings.logging_enabled:
    setup_logging()

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

ANSWER_TIMEOUT_SECONDS = 300

QUERY_EDGES: tuple[tuple[str, str], ...] = tuple(zip(QUERY_NODES, QUERY_NODES[1:], strict=False))
INGESTION_EDGES: tuple[tuple[str, str], ...] = tuple(zip(INGESTION_NODES, INGESTION_NODES[1:], strict=False))
_QUERY_STAGE_KIND: dict[str, str] = dict(
    zip(QUERY_NODES, ("parse", "embed", "search", "search", "generate"), strict=True)
)
_INGESTION_EVENT_KIND: dict[str, str] = {
    "fetch_success": "parse",
    "page_indexed": "parse",
    "batch_embedding": "embed",
    "batch_indexing": "search",
}

LAB_NODES: tuple[str, ...] = ("HTML Source", "Markdown", "Chunker", "Filter", "Enrich", "Embed", "Qdrant")
LAB_EDGES: tuple[tuple[str, str], ...] = tuple(zip(LAB_NODES, LAB_NODES[1:], strict=False))
LAB_NODE_BY_STAGE: dict[str, str] = {
    "raw": "HTML Source",
    "markdown": "Markdown",
    "chunk": "Chunker",
    "filter": "Filter",
    "enrich": "Enrich",
    "embed": "Embed",
    "qdrant": "Qdrant",
}
LAB_STAGE_TITLES: dict[str, str] = {
    "raw": "① Raw HTML",
    "markdown": "② Markdown conversion",
    "chunk": "③ Header-aware chunking",
    "filter": "④ Quality filtering",
    "enrich": "⑤ Metadata enrichment",
    "embed": "⑥ Vector embeddings",
    "qdrant": "⑦ Qdrant point payload",
}

_SPARK_SAMPLE_HTML = """<!doctype html>
<html><head><title>SparkSession — PySpark API Reference</title></head>
<body>
<h1>SparkSession</h1>
<p>The SparkSession class is the main entry point for DataFrame and SQL
functionality in PySpark. It is the user-facing API to configure Spark, read
data sources, create DataFrames, and execute SQL queries. Every Spark
application needs one active SparkSession that coordinates the execution of
jobs across the cluster.</p>
<h2>Methods</h2>
<h3>spark.sql</h3>
<p>Returns a DataFrame representing the result of the given query string. The
query is parsed and executed by the Catalyst optimizer before execution.</p>
<pre><code>df = spark.sql("SELECT * FROM events WHERE ts &gt; '2024-01-01'")</code></pre>
<h3>spark.read</h3>
<p>Returns a DataFrameReader that can be used to load data from a variety of
sources such as Parquet, JSON, Delta Lake tables, and JDBC endpoints.</p>
<pre><code>df = spark.read.format("parquet").load("/mnt/warehouse/events")</code></pre>
<h3>spark.conf.set</h3>
<p>Sets a runtime Spark configuration property for the session.</p>
<pre><code>spark.conf.set("spark.sql.shuffle.partitions", "200")</code></pre>
</body></html>
"""


def _new_session_identifiers() -> tuple[str, str]:
    """Generate a fresh (session_id, user_id) pair for a browser session.

    user_id is a stable anonymous identifier derived from the session so every
    trace carries a distinct user without requiring authentication.
    """
    session_id = str(uuid.uuid4())
    return session_id, f"anon-{session_id[:8]}"


def _get_chat_user_id() -> str:
    """Return a stable anonymous user_id for chat session scoping.

    Persisted in ``st.session_state`` once so chat sessions survive Streamlit
    reruns within a browser tab (the RAG ``session_id``/``user_id`` pair is
    regenerated per tab and would otherwise orphan chat sessions on reload).
    """
    if "chat_user_id" not in st.session_state:
        st.session_state.chat_user_id = f"anon-{uuid.uuid4().hex[:12]}"
    return st.session_state.chat_user_id


def _init_chat_state() -> None:
    """Initialize the chat tab's session state (messages + active session)."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None
    if "chat_suggestions" not in st.session_state:
        st.session_state.chat_suggestions = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


# ---------------------------------------------------------------------------
# Service health checks
# ---------------------------------------------------------------------------


def _check_qdrant_reachable(timeout: float = 2.0) -> tuple[bool, str]:
    """Check if Qdrant is reachable. Returns (ok, message).

    Honors the ``STREAMLIT_ASSUME_QDRANT_UP`` env var: when set to a truthy
    value, assumes Qdrant is reachable without a network probe. This lets the
    AppTest-based UI tests render the chat tab hermetically (AppTest re-imports
    the module, so a test-process monkeypatch would not propagate).
    """
    if os.environ.get("STREAMLIT_ASSUME_QDRANT_UP", "").strip().lower() in {"1", "true", "yes"}:
        return True, "Qdrant assumed reachable (test override)"
    try:
        url = f"{settings.qdrant_url}/healthz"
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
    except (TimeoutError, OSError) as exc:
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
                        "**Pull them with:**\n```\n" + "\n".join(f"ollama pull {m}" for m in missing) + "\n```"
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
    except (TimeoutError, OSError) as exc:
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
    except (TimeoutError, urllib.error.URLError, OSError):
        return False, (
            "Langfuse is not reachable. Tracing will be disabled.\n\n"
            "**Start it with:**\n```\ndocker compose up -d langfuse langfuse-postgres clickhouse minio\n```"
        )


def _check_deps_fingerprint(timeout: float = 2.0) -> tuple[bool, str]:
    """Check if the Docker image dependencies are fresh. Returns (ok, message)."""
    try:
        url = f"{API_BASE_URL}/api/v1/version"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            deps_ok = data.get("deps_fingerprint_ok")
            if deps_ok is True:
                return True, "Dependencies are fresh"
            elif deps_ok is False:
                msg = data.get("deps_stale_message", "Docker image is stale.")
                return False, msg
            else:
                return True, "Not running in Docker (check skipped)"
    except urllib.error.URLError:
        return True, "API not reachable (check skipped)"
    except (TimeoutError, OSError):
        return True, "API timeout (check skipped)"


# ---------------------------------------------------------------------------
# Ingestion API helpers
# ---------------------------------------------------------------------------


def _record_user_feedback(trace_id: str | None, rating: int, comment: str | None = None) -> None:
    """Record user feedback as a score on a Langfuse trace.

    Posts to ``POST /api/v1/feedback`` (single path shared with other clients);
    falls back to scoring the trace directly when the API is unreachable.
    Fail-open: never raises.
    """
    if not trace_id:
        return
    try:
        payload = json.dumps({"trace_id": trace_id, "rating": rating, "comment": comment}).encode()
        req = urllib.request.Request(
            f"{API_BASE_URL}/api/v1/feedback",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
            return
    except Exception:
        logger.debug("API feedback POST failed; falling back to direct tracer", exc_info=True)
    try:
        from data_engineering_copilot.observability.telemetry import build_telemetry_tracer

        tracer = build_telemetry_tracer()
        if hasattr(tracer, "score"):
            tracer.score(
                trace_id=trace_id,
                name="user_feedback",
                value=float(rating),
                data_type="NUMERIC",
                comment=comment,
            )
            tracer.flush()
    except Exception:
        logger.debug("Direct user-feedback score failed", exc_info=True)


def _post_ingest(source_names: list[str], max_pages: int, use_async: bool = True) -> tuple[str | None, str | None]:
    """POST to /api/v1/ingest to start a background Celery task.

    Returns (task_id, error_message).
    """
    try:
        payload = json.dumps({"source_names": source_names, "max_pages": max_pages, "use_async": use_async}).encode()
        req = urllib.request.Request(
            f"{API_BASE_URL}/api/v1/ingest",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("task_id"), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except Exception:
            detail = body
        return None, detail
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        return None, (
            f"Cannot reach the API server at `{API_BASE_URL}`: {exc}\n\n"
            "**Start the API and Celery worker:**\n"
            "```\ndocker compose up -d backend-api celery_worker\n```"
        )
    except Exception as exc:
        return None, str(exc)


def _get_ingest_status(task_id: str) -> tuple[dict | None, str | None]:
    """GET /api/v1/ingest/status/{task_id} to poll progress from Redis.

    Returns ``(status_dict, None)`` on success, ``(None, None)`` when the
    task is genuinely not found (HTTP 404), or ``(None, error_message)`` for
    any other failure (connection refused, timeout, server error, etc.).
    """
    try:
        req = urllib.request.Request(f"{API_BASE_URL}/api/v1/ingest/status/{task_id}")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, None
        return None, f"HTTP {exc.code}: {exc.reason}"
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        return None, f"Cannot reach API: {exc}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def _get_latest_task_id() -> str | None:
    """GET /api/v1/ingest/latest to discover a running task from any session."""
    try:
        req = urllib.request.Request(f"{API_BASE_URL}/api/v1/ingest/latest")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return data.get("task_id")
    except Exception:
        return None


def _post_cancel_ingest(task_id: str) -> bool:
    """POST /api/v1/ingest/{task_id}/cancel to revoke a Celery task.

    Returns True on success.
    """
    try:
        req = urllib.request.Request(
            f"{API_BASE_URL}/api/v1/ingest/{task_id}/cancel",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


@st.cache_resource
def _build_rag_service():
    return build_rag_service()


@st.cache_resource
def _get_service_loop():
    """Return a single long-lived event loop running on a daemon thread.

    All async components of the RAG service (httpx clients, the redis-asyncio
    pool, the Qdrant client) bind to whichever loop first awaits them. Running
    the pipeline on one persistent loop prevents the cross-loop failures that
    otherwise occur when a cached service is reused across Streamlit reruns.
    """
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_run, name="rag-service-loop", daemon=True).start()
    return loop


def rag_service():
    """Return cached RAG service, or None if Qdrant/Ollama are unavailable."""
    try:
        return _build_rag_service()
    except Exception as exc:
        logger.warning("Failed to create RAG service: %s", exc)
        return None


@st.cache_data(show_spinner=False, max_entries=64)
def _embed_for_scatter(question: str, chunk_texts: tuple[str, ...]) -> tuple[list[float], list[list[float]]]:
    """Lazily embed the query + retrieved chunk texts for the 3D scatter.

    Runs on the shared service event loop so the cached embedder is never
    touched from a foreign loop. Returns empty lists when embeddings cannot be
    produced (caller falls back to a text view).
    """
    service = rag_service()
    if service is None or not hasattr(service, "embedder"):
        return [], []
    loop = _get_service_loop()
    try:
        query_emb = asyncio.run_coroutine_threadsafe(service.embedder.embed_query(question), loop).result(timeout=90)
        if not chunk_texts:
            return query_emb, []
        chunk_embs = asyncio.run_coroutine_threadsafe(service.embedder.embed_texts(list(chunk_texts)), loop).result(
            timeout=180
        )
        return query_emb, chunk_embs
    except TimeoutError:
        logger.warning("Timed out embedding vectors for scatter")
        return [], []
    except Exception as exc:
        logger.warning("Failed to embed vectors for scatter: %s", exc)
        return [], []


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
    recent_events: list[dict] = field(default_factory=list)
    current_url: str = ""
    last_message: str = ""
    error: str | None = None
    success_message: str | None = None


class IngestionManager:
    """Manages ingestion lifecycle via Celery task + Redis polling."""

    @classmethod
    def start(cls, source_names: tuple[str, ...], max_pages: int, use_async: bool = True) -> tuple[bool, str]:
        """Start ingestion via the FastAPI API.

        Returns (started, error_message).
        """
        task_id, error = _post_ingest(list(source_names), max_pages, use_async=use_async)
        if error:
            return False, error
        if task_id:
            st.session_state.ingestion_task_id = task_id
            st.session_state.ingestion_source_names = list(source_names)
            st.session_state.ingestion_max_pages = max_pages
            st.session_state.ingestion_start_time = time.time()
            return True, ""
        return False, "No task ID returned."

    @classmethod
    def get_progress(cls) -> IngestionProgress:
        """Read progress from Redis via the API polling endpoint."""
        # If a final progress snapshot was cached, return it directly
        final = st.session_state.get("_ingest_final_progress")
        if final is not None:
            return final

        task_id = st.session_state.get("ingestion_task_id")
        if not task_id:
            latest_task_id = _get_latest_task_id()
            if latest_task_id:
                status, _ = _get_ingest_status(latest_task_id)
                if status and status.get("status") in ("PROCESSING", "DISPATCHED"):
                    task_id = latest_task_id
                    st.session_state.ingestion_task_id = task_id
                    st.session_state.ingestion_start_time = time.time()
        if not task_id:
            return IngestionProgress()

        status, api_error = _get_ingest_status(task_id)
        if status is None and api_error is not None:
            return IngestionProgress(
                error=f"API unreachable: {api_error}. Ingestion may still be running in the background.",
            )
        if status is None:
            return IngestionProgress(
                error="Ingestion task not found. It may have expired or the session was refreshed.",
            )

        api_status = status.get("status", "")
        is_running = api_status in ("PROCESSING", "DISPATCHED")
        source_names = tuple(status.get("source_names", []))
        start_time = st.session_state.get("ingestion_start_time", time.time())

        # Freeze elapsed time at completion so it doesn't keep growing
        if is_running:
            elapsed_seconds = time.time() - start_time
        else:
            frozen = st.session_state.get("_ingest_final_elapsed")
            if frozen is None:
                frozen = time.time() - start_time
                st.session_state._ingest_final_elapsed = frozen
            elapsed_seconds = frozen

        # Build per-source detail from real Redis source_stats
        sources: dict[str, SourceProgress] = {}
        raw_source_stats = status.get("source_stats", {})
        for name in source_names:
            s = raw_source_stats.get(name, {})
            sources[name] = SourceProgress(
                name=name,
                status="complete" if not is_running else "crawling",
                pages_fetched=s.get("pages_fetched", 0),
                pages_skipped=s.get("pages_skipped", 0),
                chunks_indexed=s.get("chunks_indexed", 0),
                errors=s.get("errors", 0),
            )

        error_msg = status.get("error")
        is_completed = api_status == "COMPLETED"
        is_cancelled = api_status == "CANCELLED"
        is_failed = api_status == "FAILED" or is_cancelled

        if is_cancelled and not error_msg:
            error_msg = "Ingestion cancelled."

        success_msg = None
        if is_completed:
            total_chunks = status.get("chunks_indexed", 0)
            success_msg = f"Refresh complete. Indexed or updated {total_chunks} chunks."

        current_url = status.get("current_url", "")
        recent_events = status.get("recent_events", [])
        last_msg = current_url if is_running else ("Ingestion complete." if is_completed else "")

        result = IngestionProgress(
            is_running=is_running,
            start_time=start_time,
            elapsed_seconds=elapsed_seconds,
            source_names=source_names,
            max_pages_per_source=st.session_state.get("ingestion_max_pages", 0),
            current_phase="crawling" if is_running else ("complete" if is_completed else "error"),
            total_pages_fetched=status.get("pages_fetched", 0),
            total_chunks_indexed=status.get("chunks_indexed", 0),
            total_errors=1 if is_failed else 0,
            sources=sources,
            recent_events=recent_events,
            current_url=current_url,
            error=error_msg if is_failed else None,
            success_message=success_msg,
            last_message=last_msg,
        )

        # Cache the final progress snapshot so it survives Redis TTL expiry
        if not is_running and (success_msg or error_msg):
            st.session_state._ingest_final_progress = result

        return result

    @classmethod
    def is_running(cls) -> bool:
        return cls.get_progress().is_running

    @classmethod
    def stop(cls) -> bool:
        """Cancel the running ingestion task via Celery revoke."""
        task_id = st.session_state.get("ingestion_task_id")
        if not task_id:
            return False
        return _post_cancel_ingest(task_id)

    @classmethod
    def reset_status(cls) -> None:
        """Clear all ingestion session state."""
        for key in (
            "ingestion_task_id",
            "ingestion_source_names",
            "ingestion_max_pages",
            "ingestion_start_time",
            "_ingest_final_progress",
            "_ingest_final_elapsed",
            "_ingest_was_running",
        ):
            st.session_state.pop(key, None)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def _badge_kind_for_event(progress: IngestionProgress) -> str:
    """Choose the Lottie badge matching the most recent ingestion activity."""
    for event in reversed(progress.recent_events):
        kind = _INGESTION_EVENT_KIND.get(event.get("type", ""))
        if kind:
            return kind
    return "parse"


def _render_state_bar(progress: IngestionProgress) -> None:
    """Compact one-line state bar with status, duration, and counts."""
    if progress.is_running:
        icon = "🔄"
        status_label = "Running"
        color = "blue"
    elif progress.success_message:
        icon = "✅"
        status_label = "Completed"
        color = "green"
    elif progress.error:
        icon = "❌"
        status_label = "Failed"
        color = "red"
    else:
        icon = "⏸️"
        status_label = "Idle"
        color = "gray"

    cols = st.columns([1, 3, 8])
    cols[0].markdown(f"# {icon}")
    cols[1].markdown(f"#### :{color}[{status_label}]\n:{color}[{_format_duration(progress.elapsed_seconds)}]")
    stats = []
    if progress.total_pages_fetched > 0:
        stats.append(f"**{progress.total_pages_fetched}** pages")
    if progress.total_chunks_indexed > 0:
        stats.append(f"**{progress.total_chunks_indexed}** chunks")
    if progress.total_errors > 0:
        stats.append(f":red[**{progress.total_errors}** errors]")
    if progress.current_url and progress.is_running:
        stats.append(f"`{progress.current_url[:80]}`")
    cols[2].markdown(" · ".join(stats) if stats else "")


@st.fragment(run_every=2.0)
def _render_progress_panel() -> None:
    """Auto-refreshing fragment that shows ingestion progress."""
    progress = IngestionManager.get_progress()

    was_running = st.session_state.get("_ingest_was_running", False)
    is_now_done = was_running and not progress.is_running and (progress.success_message or progress.error)
    if is_now_done:
        st.session_state._ingest_was_running = False
        st.rerun(scope="app")

    render_pipeline_diagram(
        INGESTION_NODES,
        INGESTION_EDGES,
        ingestion_node_states(progress),
        with_view_switch=True,
        key="ingest_panel",
    )

    if not progress.is_running and not progress.success_message and not progress.error and not progress.source_names:
        return

    if progress.is_running:
        st.session_state._ingest_was_running = True

    _render_state_bar(progress)

    tab_overview, tab_sources, tab_log, tab_history = st.tabs(["Overview", "Sources", "Live Log", "History"])

    with tab_overview:
        _render_overview_tab(progress)
    with tab_sources:
        _render_sources_tab(progress)
    with tab_log:
        _render_live_log_tab(progress)
    with tab_history:
        _render_history_tab(progress)


def _render_overview_tab(progress: IngestionProgress) -> None:
    """Pipeline overview: stages, current URL, throughput."""
    if progress.is_running:
        badge_col, p_col, c_col = st.columns([1, 2, 2])
        with badge_col:
            render_lottie_badge(_badge_kind_for_event(progress), height=80)
        with p_col:
            render_animated_metric("Pages", progress.total_pages_fetched)
        with c_col:
            render_animated_metric("Chunks", progress.total_chunks_indexed)

        total_sources = len(progress.source_names) or 1
        effective_max_pages = progress.max_pages_per_source or settings.max_pages_per_source
        estimated_pages = effective_max_pages * total_sources
        page_ratio = min(progress.total_pages_fetched / max(estimated_pages, 1), 1.0)
        st.progress(page_ratio, text=f"{progress.total_pages_fetched} / {estimated_pages} pages")

        if progress.current_url:
            st.caption(f"Current URL: `{progress.current_url[:120]}`")

        elapsed = progress.elapsed_seconds
        throughput = progress.total_pages_fetched / elapsed if elapsed > 0 else 0
        rolling_remaining = max(0, estimated_pages - progress.total_pages_fetched)
        eta = rolling_remaining / throughput if throughput > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Throughput", f"{throughput:.1f}/s" if throughput > 0 else "—")
        c2.metric("ETA", _format_duration(eta) if eta > 0 else "—")
        c3.metric("Pages", progress.total_pages_fetched)
        c4.metric("Chunks", progress.total_chunks_indexed)
    elif progress.success_message:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", _format_duration(progress.elapsed_seconds))
        c2.metric("Pages", progress.total_pages_fetched)
        c3.metric("Chunks", progress.total_chunks_indexed)
        c4.metric("Sources", len(progress.source_names))
    elif progress.error:
        c1, c2 = st.columns([1, 3])
        c1.metric("Duration", _format_duration(progress.elapsed_seconds))
        c2.error(progress.error)


def _render_sources_tab(progress: IngestionProgress) -> None:
    """Dataframe with per-source progress."""
    if not progress.source_names:
        st.caption("No sources selected.")
        return

    rows = []
    for name in progress.source_names:
        src = progress.sources.get(name)
        if src is None:
            rows.append(
                {
                    "Source": name,
                    "Status": "⏳ pending",
                    "Pages": 0,
                    "Chunks": 0,
                    "Errors": 0,
                    "Progress": 0.0,
                }
            )
        else:
            effective_max = progress.max_pages_per_source or settings.max_pages_per_source
            pct = min(src.pages_fetched / max(effective_max, 1), 1.0)
            if progress.is_running:
                status_icon = "🔄" if src.pages_fetched > 0 else "⏳"
            elif src.errors > 0:
                status_icon = "❌"
            else:
                status_icon = "✅"
            rows.append(
                {
                    "Source": name,
                    "Status": f"{status_icon} {src.status}",
                    "Pages": src.pages_fetched,
                    "Chunks": src.chunks_indexed,
                    "Errors": src.errors,
                    "Progress": pct,
                }
            )

    st.dataframe(
        rows,
        column_config={
            "Progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=1, format="%.0f%%"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Pages": st.column_config.NumberColumn("Pages", width="small"),
            "Chunks": st.column_config.NumberColumn("Chunks", width="small"),
            "Errors": st.column_config.NumberColumn("Errors", width="small"),
        },
        width="stretch",
        hide_index=True,
    )

    if progress.current_url and progress.is_running:
        st.caption(f"Crawling: `{progress.current_url[:120]}`")


def _render_live_log_tab(progress: IngestionProgress) -> None:
    """Scrollable live event log with source filter."""
    events = progress.recent_events
    if not events:
        st.caption("No events yet.")
        return

    source_names = list(dict.fromkeys(e.get("source", "") for e in events if e.get("source")))
    filter_source = st.selectbox("Filter by source", ["All"] + source_names, key="log_source_filter")
    filter_types = st.multiselect(
        "Event types",
        [
            "fetch_success",
            "page_indexed",
            "page_skipped_cached",
            "page_skipped_duplicate",
            "batch_embedding",
            "batch_indexing",
            "source_complete",
            "error",
        ],
        default=[],
        key="log_type_filter",
    )

    filtered = events
    if filter_source != "All":
        filtered = [e for e in filtered if e.get("source") == filter_source]
    if filter_types:
        filtered = [e for e in filtered if e.get("type") in filter_types]

    if not filtered:
        st.caption("No matching events.")
        return

    icon_map = {
        "fetch_success": "📥",
        "page_indexed": "✅",
        "page_skipped_cached": "⏭️",
        "page_skipped_duplicate": "⏭️",
        "batch_embedding": "📦",
        "batch_indexing": "💾",
        "source_complete": "🎯",
        "error": "❌",
    }

    with st.container(height=400, border=True):
        for evt in reversed(filtered[-100:]):
            icon = icon_map.get(evt.get("type", ""), "ℹ️")
            ts = _format_timestamp(evt.get("ts", 0))
            source = evt.get("source", "")
            label = evt.get("title", "") or evt.get("url", "")
            err = evt.get("error", "")
            line = f"{ts}  {icon}  {source + '  ' if source else ''}{label}"
            if err:
                st.markdown(f":red[{line}]")
                st.caption(f":red[Error: {err}]")
            else:
                st.markdown(line)


def _render_history_tab(progress: IngestionProgress) -> None:
    """Past ingestion runs (stored in session state)."""
    history = st.session_state.get("ingestion_history", [])
    if not history:
        st.caption("No previous runs in this session.")
        return

    st.dataframe(
        history,
        column_config={
            "Time": "Time",
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Pages": st.column_config.NumberColumn("Pages", width="small"),
            "Chunks": st.column_config.NumberColumn("Chunks", width="small"),
            "Duration": st.column_config.TextColumn("Duration", width="small"),
        },
        width="stretch",
        hide_index=True,
    )


def _format_timestamp(ts: float) -> str:
    """Format a Unix timestamp to HH:MM:SS."""
    if not ts:
        return ""
    import datetime

    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Conversational RAG (Chat) tab
# ---------------------------------------------------------------------------


def _api_headers() -> dict[str, str]:
    """Headers for chat API calls (stable anonymous user scoping)."""
    return {"X-User-ID": _get_chat_user_id()}


def _load_chat_sessions() -> list[dict]:
    """Fetch the caller's chat sessions from the API (fail-open)."""
    try:
        req = urllib.request.Request(f"{API_BASE_URL}/api/v1/sessions", headers=_api_headers())
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read()).get("sessions", [])
    except Exception:
        return []


def _render_session_manager() -> None:
    """Sidebar widgets for New Chat, session list, and delete."""
    sessions = _load_chat_sessions()
    with st.sidebar:
        st.markdown("### 💬 Chat Sessions")
        if st.button("🆕 New Chat", key="chat_new_chat"):
            st.session_state.chat_session_id = None
            st.session_state.chat_messages = []
            st.rerun()

        if sessions:
            labels = {s["title"]: s["session_id"] for s in sessions}
            current = st.session_state.get("chat_session_id")
            current_title = next((t for t, sid in labels.items() if sid == current), None)
            selected = st.selectbox(
                "Session",
                options=list(labels.keys()),
                index=list(labels.keys()).index(current_title) if current_title in labels else 0,
                key="chat_session_select",
            )
            if selected and labels.get(selected) != st.session_state.get("chat_session_id"):
                st.session_state.chat_session_id = labels[selected]
                st.rerun()
            if st.button("🗑 Delete Session", key="chat_delete_btn"):
                try:
                    sid = labels.get(selected)
                    if sid:
                        req = urllib.request.Request(
                            f"{API_BASE_URL}/api/v1/sessions/{sid}",
                            method="DELETE",
                            headers=_api_headers(),
                        )
                        urllib.request.urlopen(req, timeout=5.0).close()
                    if sid == st.session_state.get("chat_session_id"):
                        st.session_state.chat_session_id = None
                        st.session_state.chat_messages = []
                except Exception:
                    st.sidebar.error("Failed to delete session.")
                st.rerun()
        else:
            st.caption("No saved conversations yet.")


def _iter_chat_events(message: str, session_id: str | None):
    """Yield parsed SSE event dicts for one chat turn as they arrive.

    Iterating this generator consumes the streaming response incrementally so
    the caller can update the UI (progress status + live tokens) between events
    instead of blocking until the whole turn completes.
    """
    import httpx

    body = {"message": message}
    if session_id:
        body["session_id"] = session_id

    with httpx.stream(
        "POST",
        f"{API_BASE_URL}/api/v1/chat",
        json=body,
        headers=_api_headers(),
        timeout=httpx.Timeout(300.0, connect=10.0),
    ) as response:
        response.raise_for_status()
        buffer = ""
        for line in response.iter_lines():
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload_text = line[len("data: ") :]
            if payload_text == "[DONE]":
                return
            try:
                event = json.loads(payload_text)
            except json.JSONDecodeError:
                buffer += payload_text
                continue
            yield event


def _extract_streaming_answer(buffer: str) -> str | None:
    """Return the clean answer if *buffer* is complete valid JSON with an answer.

    Returns ``None`` while the streamed JSON is still incomplete so the UI shows
    the progress status instead of raw ``{"status": ...}`` fragments.
    """
    if not buffer or not buffer.strip():
        return None
    import re

    text = buffer.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    answer = data.get("answer") or data.get("response") or data.get("text") or data.get("content")
    if answer is None:
        return None
    return str(answer)


def _stream_chat_once(message: str, session_id: str | None) -> tuple[list[dict], str, str | None]:
    """Run one chat turn against the API, returning all events + final text.

    ``(events, full_text, new_session_id)``. Kept for tests/aggregate use; the
    chat tab consumes ``_iter_chat_events`` directly for live updates.
    """
    events = list(_iter_chat_events(message, session_id))
    full_text = ""
    resolved_session = session_id
    for event in events:
        if event.get("type") == "session_created":
            resolved_session = event.get("session_id") or resolved_session
        if event.get("type") == "token":
            full_text += event.get("content", "")
        if event.get("type") == "done":
            full_text = event.get("text", full_text)
    return events, full_text, resolved_session


def _render_suggestion_chips(suggestions: list[str]) -> None:
    """Render clickable follow-up suggestion chips (ChatGPT-style).

    Each chip is a button that, when clicked, sets ``pending_prompt`` so the
    unified prompt-resolution block submits it as the next user turn.
    """
    if not suggestions:
        return

    def _set_pending(chip: str) -> None:
        st.session_state.pending_prompt = chip

    st.markdown("**Suggested follow-ups**")
    cols = st.columns(len(suggestions))
    for col, chip in zip(cols, suggestions, strict=False):
        with col:
            st.button(
                chip,
                key=f"sugg_{hash(chip)}",
                type="tertiary",
                width="stretch",
                on_click=_set_pending,
                args=(chip,),
            )


def render_chat_tab() -> None:
    """Conversational RAG tab: multi-turn chat with session memory."""
    from data_engineering_copilot.ui.chat_theme import apply_chat_theme

    st.subheader("💬 Conversational RAG")
    apply_chat_theme()

    qdrant_ok, _ = _check_qdrant_reachable()
    if not qdrant_ok or not settings.chat_enabled:
        st.info("Conversational chat is unavailable (Qdrant unreachable or chat disabled).")
        return

    _init_chat_state()
    _render_session_manager()

    # ChatGPT-style layout: the conversation lives in a scrollable container so
    # the input box always sits at the bottom of the last message. The height is
    # kept modest so the chat input below stays within the visible tab area
    # (inside st.tabs() st.chat_input renders in-flow, not pinned to the page).
    chat_container = st.container(height=360, border=False, autoscroll=True)

    # Render existing messages.
    for msg in st.session_state.chat_messages:
        with chat_container.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. [{source.get('title', 'Source')}]({source.get('url', '#')})**")
                        st.caption(f"Source: {source.get('source_name')}")
            claims = msg.get("groundedness_claims") or []
            if claims:
                st.caption(
                    f"Groundedness {float(msg.get('groundedness_score', 1.0)):.2f} — "
                    f"{len(claims)} claim(s) not directly supported"
                )

    # ChatGPT-style: clickable follow-up suggestions are rendered BELOW the
    # freshly generated answer (see the turn block) so chips always reflect the
    # latest answer — never stale chips from the previous turn.
    prompt = st.chat_input(
        "Ask a follow-up about Spark, Airflow, Delta Lake…",
        submit_mode="stop",
    )
    # A suggestion chip click resolves to a prompt just like typing one.
    pending = st.session_state.pop("pending_prompt", None)
    if not prompt and pending:
        prompt = pending
    if prompt:
        # New turn clears the previous turn's suggestion chips.
        st.session_state.chat_suggestions = []
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with chat_container.chat_message("assistant", avatar="🤖"):
            text_ph = st.empty()
            status_ph = st.empty()
            full_text = ""
            raw_buffer = ""
            resolved_session = st.session_state.get("chat_session_id")
            error_msg: str | None = None
            turn_sources: list[dict] = []
            groundedness_score = 1.0
            groundedness_claims: list[str] = []

            try:
                with status_ph.status("Connecting to chat API…", expanded=True) as status:
                    for event in _iter_chat_events(prompt, st.session_state.get("chat_session_id")):
                        etype = event.get("type")
                        if etype == "session_created":
                            resolved_session = event.get("session_id") or resolved_session
                            status.update(label="Session ready", state="running")
                        elif etype == "status":
                            status.update(label=event.get("message", "Working…"), state="running")
                        elif etype == "token":
                            raw_buffer += event.get("content", "")
                            clean = _extract_streaming_answer(raw_buffer)
                            if clean is not None:
                                full_text = clean
                                text_ph.markdown(full_text)
                        elif etype == "sources":
                            turn_sources = event.get("sources", [])
                        elif etype == "done":
                            full_text = event.get("text", full_text)
                            groundedness_score = float(event.get("groundedness_score", 1.0))
                            groundedness_claims = event.get("groundedness_claims") or []
                            text_ph.markdown(full_text)
                        elif etype == "suggestions":
                            st.session_state.chat_suggestions = event.get("suggestions", [])
                        elif etype == "error":
                            error_msg = event.get("message", "Unknown error")
                            status.update(label="Failed", state="error")
            except Exception as exc:
                logger.exception("Chat turn failed")
                error_msg = str(exc)

            st.session_state.chat_session_id = resolved_session

            # Remove the transient progress widget so no "Done" status lingers.
            if not error_msg:
                status_ph.empty()

            if error_msg:
                text_ph.error(f"**Chat failed:** {error_msg}")
            elif not full_text:
                text_ph.markdown("…")

        # Citations + groundedness for the freshly generated turn. Sources arrive
        # as a separate SSE event; render them under the answer just like the QA
        # tab, then persist them with the assistant message for later renders.
        assistant_payload: dict = {
            "role": "assistant",
            "content": full_text or (f"Error: {error_msg}" if error_msg else "(no answer)"),
            "sources": turn_sources,
            "groundedness_score": groundedness_score,
            "groundedness_claims": groundedness_claims,
        }
        st.session_state.chat_messages.append(assistant_payload)
        if turn_sources and not error_msg:
            with st.expander(f"Sources ({len(turn_sources)})", expanded=False):
                for i, source in enumerate(turn_sources, 1):
                    st.markdown(f"**{i}. [{source.get('title', 'Source')}]({source.get('url', '#')})**")
                    st.caption(f"Source: {source.get('source_name')}")
        if groundedness_claims and not error_msg:
            st.caption(
                f"Groundedness {groundedness_score:.2f} — {len(groundedness_claims)} claim(s) not directly supported"
            )

        # Render the follow-up chips for THIS answer immediately, inside the
        # chat container so they appear under the answer but ABOVE the input box.
        fresh = st.session_state.get("chat_suggestions", [])
        if fresh and not error_msg:
            with chat_container:
                _render_suggestion_chips(fresh)


def render_qa_tab() -> None:
    """Q&A tab: ask questions against the knowledge base."""
    st.subheader("Ask a Question")

    # Pre-flight: check services before showing the input
    qdrant_ok, qdrant_msg = _check_qdrant_reachable()
    llm_ok, llm_msg = True, ""
    if settings.llm_provider.lower() == "ollama" or settings.embedding_provider.lower() == "ollama":
        llm_ok, llm_msg = _check_ollama_reachable()

    if not qdrant_ok or not llm_ok:
        if not qdrant_ok:
            st.error(f"**Qdrant unavailable**\n\n{qdrant_msg}")
        if not llm_ok:
            st.error(f"**LLM provider unavailable**\n\n{llm_msg}")
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
                    "**Check that Qdrant and the LLM provider are reachable.**\n"
                    "See the **System Health** tab for details."
                )
                return

            logger.info("Streamlit ask started question=%r", question.strip()[:200])

            completed_steps: list[str] = []
            result_box: list = []
            error_box: list = []
            step_details: dict[str, list[dict]] = {}

            def _run_in_background() -> None:
                """Run the async pipeline on the shared service event loop."""

                def on_step(step_name: str) -> None:
                    completed_steps.append(step_name)

                def on_step_detail(kind: str, payload: dict) -> None:
                    step_details.setdefault(kind, []).append(payload)

                future = asyncio.run_coroutine_threadsafe(
                    service.answer(
                        question.strip(),
                        on_step=on_step,
                        on_step_detail=on_step_detail,
                        user_id=st.session_state.get("user_id"),
                        session_id=st.session_state.get("session_id"),
                    ),
                    _get_service_loop(),
                )
                try:
                    result_box.append(future.result(timeout=ANSWER_TIMEOUT_SECONDS))
                except Exception as e:
                    error_box.append(e)

            worker = threading.Thread(target=_run_in_background, daemon=True)
            worker.start()

            diagram_ph = st.empty()
            stage_phs = [st.empty() for _ in QUERY_NODES]

            def _render_query_pipeline(events: list[str], completed: bool = False, failed: str | None = None) -> None:
                states = reduce_query_node_states(events, failed_step=failed, completed=completed)
                diagram_ph.html(build_diagram_html(QUERY_NODES, QUERY_EDGES, states, show_legend=False))
                node_details = {node: step_details.get(node.lower(), []) for node in QUERY_NODES}
                render_stepper(
                    stage_phs,
                    [(node, states[node], _QUERY_STAGE_KIND[node]) for node in QUERY_NODES],
                    details=node_details,
                )

            wait_deadline = time.monotonic() + ANSWER_TIMEOUT_SECONDS
            try:
                with st.status("Searching...", expanded=True) as status:
                    while worker.is_alive() and time.monotonic() < wait_deadline:
                        _render_query_pipeline(list(completed_steps))
                        if completed_steps:
                            label = f"Step {len(completed_steps)}/{len(QUERY_NODES)}: {completed_steps[-1]}"
                            status.update(label=label, state="running")
                        time.sleep(0.3)

                    if worker.is_alive() and time.monotonic() >= wait_deadline:
                        raise TimeoutError(
                            f"No answer generated within {ANSWER_TIMEOUT_SECONDS}s. "
                            "The LLM provider chain may be slow, rate-limited, or unavailable. "
                            "Check the **System Health** tab and try again."
                        )

                    if error_box:
                        _render_query_pipeline(list(completed_steps), failed="Generate")
                        raise error_box[0]
                    if not result_box:
                        raise RuntimeError("Background thread completed without result")

                    answer = result_box[0]
                    _render_query_pipeline(list(completed_steps), completed=True)
                    # Store trace_id per question for feedback tracking (robust
                    # across multiple questions in one session).
                    if hasattr(answer, "trace_id") and answer.trace_id:
                        st.session_state.last_trace_id = answer.trace_id
                        st.session_state.trace_ids[question.strip()] = answer.trace_id
                    status.update(label="✅ Answer ready", state="complete")
            except Exception as exc:
                logger.exception("RAG answer failed")
                st.error(
                    f"**Failed to get answer:** {exc}\n\n"
                    "**Possible causes:**\n"
                    "- LLM provider may have timed out or hit rate limits\n"
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
            if answer.text and answer.text.strip():
                st.write_stream(stream_answer_text(answer.text))
            else:
                st.warning(
                    "No answer could be generated for this question. "
                    "The knowledge base may not contain enough information, or the "
                    "LLM returned an empty response. Try rephrasing the question."
                )
            st.caption(f"Confidence: {answer.confidence:.2%}")

            if answer.sources:
                with st.expander("Vector Space", expanded=False):
                    chunk_texts = tuple(source.text for source in answer.sources)
                    with st.spinner("Projecting embeddings..."):
                        query_emb, chunk_embs = _embed_for_scatter(question.strip(), chunk_texts)
                    if chunk_embs and len(chunk_embs) == len(answer.sources):
                        labels = [
                            f"{i + 1}. {source.source_name} — {source.title[:48]}"
                            for i, source in enumerate(answer.sources)
                        ]
                        scores = [float(max(source.token_count, 1)) for source in answer.sources]
                        render_vector_scatter(
                            query_emb,
                            chunk_embs,
                            labels,
                            scores,
                            key=f"scatter_{len(collector.queries)}",
                        )
                        st.caption("Query vector (red) vs retrieved chunk vectors; play ▶ to watch collapse.")
                    else:
                        st.caption("Embeddings unavailable — showing retrieved documents instead.")
                        for i, source in enumerate(answer.sources, 1):
                            st.markdown(f"**{i}. [{source.title}]({source.url})**")
                            st.caption(f"Source: {source.source_name}")

            # User feedback buttons
            col_feedback1, col_feedback2, col_feedback3 = st.columns([1, 1, 4])
            with col_feedback1:
                if st.button("👍 Helpful", key=f"helpful_{len(collector.queries)}"):
                    # Store feedback in session state
                    if "feedback" not in st.session_state:
                        st.session_state.feedback = {}
                    st.session_state.feedback[question.strip()] = {
                        "rating": "helpful",
                        "timestamp": time.time(),
                    }
                    trace_id = st.session_state.trace_ids.get(question.strip()) or getattr(
                        st.session_state, "last_trace_id", None
                    )
                    _record_user_feedback(trace_id, rating=1)
                    st.toast("Thanks for your feedback!")
            with col_feedback2:
                if st.button("👎 Not Helpful", key=f"not_helpful_{len(collector.queries)}"):
                    # Store feedback in session state
                    if "feedback" not in st.session_state:
                        st.session_state.feedback = {}
                    st.session_state.feedback[question.strip()] = {
                        "rating": "not_helpful",
                        "timestamp": time.time(),
                    }
                    trace_id = st.session_state.trace_ids.get(question.strip()) or getattr(
                        st.session_state, "last_trace_id", None
                    )
                    _record_user_feedback(trace_id, rating=0)
                    st.toast("Thanks for your feedback!")

            if answer.sources:
                with st.expander(f"Sources ({len(answer.sources)})", expanded=False):
                    for i, source in enumerate(answer.sources, 1):
                        st.markdown(f"**{i}. [{source.title}]({source.url})**")
                        st.caption(f"Source: {source.source_name}")

            # Per-answer detailed metrics
            with st.expander("Answer Metrics", expanded=False):
                c_anim1, c_anim2, c_anim3 = st.columns(3)
                with c_anim1:
                    render_animated_metric("Answer Words", len(answer.text.split()))
                with c_anim2:
                    render_animated_metric("Sources Cited", len(answer.sources))
                with c_anim3:
                    _total_ms = float(answer.stage_times.get("total", 0))
                    if not _total_ms:
                        _total_ms = float(sum(answer.stage_times.values()))
                    render_animated_metric("Latency (ms)", _total_ms)
                qm = collector.queries[-1] if collector.queries else None
                if qm:
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.metric("Query Difficulty", qm.query_difficulty.capitalize())
                        st.metric("Query Length (words)", qm.query_length)
                    with col_a2:
                        st.metric(
                            "Answer Length (words)", qm.answer_metrics.answer_length if qm.answer_metrics else "N/A"
                        )
                        st.metric("Sources Cited", qm.answer_metrics.source_count if qm.answer_metrics else "N/A")

                    if qm.answer_metrics:
                        sec_status = "Yes" if qm.answer_metrics.has_key_sections else "No"
                        unc_status = "Yes" if qm.answer_metrics.has_uncertainty_markers else "No"
                        st.caption(f"Structured sections: {sec_status}  |  Uncertainty markers: {unc_status}")

            # Per-stage pipeline trace (prompt, context, telemetry)
            with st.expander("Pipeline Trace", expanded=False):
                if answer.rewritten_query and answer.rewritten_query != question.strip():
                    st.markdown("**Effective query**")
                    st.code(answer.rewritten_query)
                if answer.query_variants:
                    st.markdown(f"**Query variants** ({len(answer.query_variants)})")
                    st.caption(" · ".join(answer.query_variants))
                if answer.intent:
                    st.markdown(f"**Intent:** `{answer.intent}`")

                if answer.retrieval_details:
                    st.markdown("**Retrieval scores**")
                    st.dataframe(
                        [
                            {
                                "rank": d.get("rank"),
                                "source": d.get("source_name"),
                                "title": (str(d.get("title")) or "")[:48],
                                "confidence": d.get("confidence"),
                                "distance": d.get("distance"),
                                "words": d.get("word_count"),
                            }
                            for d in answer.retrieval_details
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                if answer.rerank_details:
                    st.markdown("**Rerank**")
                    st.json(dict(answer.rerank_details))

                if answer.prompt:
                    st.markdown(f"**Assembled prompt** ({len(answer.prompt):,} chars)")
                    with st.expander("View prompt"):
                        st.code(answer.prompt, language="text")

                if answer.stage_times:
                    st.markdown("**Stage times**")
                    st.dataframe(
                        [{"stage": k, "ms": round(v, 1)} for k, v in answer.stage_times.items()],
                        use_container_width=True,
                        hide_index=True,
                    )

                if answer.token_usage:
                    st.markdown("**Token usage**")
                    st.json(dict(answer.token_usage))

                st.markdown(f"**Groundedness:** {answer.groundedness_score:.2f}")
                if answer.groundedness_claims:
                    st.markdown("**Unsupported claims**")
                    for claim in answer.groundedness_claims:
                        st.caption(f"- {claim}")

                if answer.trace_id:
                    st.markdown(f"**Trace:** [`{answer.trace_id}`]({build_trace_url(answer.trace_id)})")


def _lab_states(events: list[str]) -> dict[str, NodeState]:
    """Reduce lab stage events to 7-node diagram states."""
    states: dict[str, NodeState] = {node: NodeState.IDLE for node in LAB_NODES}
    done: set[str] = set()
    running: str | None = None
    for stage in events:
        node = LAB_NODE_BY_STAGE.get(stage)
        if node is None:
            continue
        if running is not None:
            done.add(running)
        running = node
    for node in done:
        states[node] = NodeState.COMPLETE
    if running is not None:
        states[running] = NodeState.RUNNING
    return states


def _render_lab_stage_payload(stage_name: str, payload: object, raw_html: str) -> None:
    """Render a single lab stage payload inside an open ``st.status``."""
    if stage_name == "raw":
        st.code(raw_html[:600] + ("..." if len(raw_html) > 600 else ""), language="html")
        return
    if stage_name == "markdown":
        markdown = getattr(payload, "get", lambda *_: None)("text") if isinstance(payload, dict) else None
        if markdown:
            left, right = st.columns(2)
            with left:
                st.caption("Raw HTML (start)")
                st.text(raw_html[:500] + ("..." if len(raw_html) > 500 else ""))
            with right:
                st.caption("Converted Markdown (start)")
                st.code(str(markdown)[:500] + ("..." if len(markdown) > 500 else ""), language="markdown")
        return
    if stage_name in ("chunk", "enrich"):
        rows = [c for c in (payload or []) if isinstance(c, dict)] if isinstance(payload, list) else []
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        return
    if stage_name == "filter":
        if isinstance(payload, dict):
            col_a, col_b = st.columns(2)
            col_a.metric("Kept", payload.get("kept", 0))
            col_b.metric("Dropped", payload.get("dropped", 0))
            reasons = payload.get("reasons") or []
            if reasons:
                st.dataframe(reasons, use_container_width=True, hide_index=True)
        return
    if stage_name == "embed":
        if isinstance(payload, dict):
            col_a, col_b = st.columns(2)
            col_a.metric("Vectors", payload.get("count", 0))
            col_b.metric("Dimensions", payload.get("dimension", 0))
            st.write(payload.get("sample"))
        return
    if stage_name == "qdrant" and isinstance(payload, dict):
        st.json(payload)
        return
    if isinstance(payload, dict):
        st.json(payload)


def render_pipeline_lab_tab() -> None:
    """Pipeline Lab: live 7-stage ingestion inspector (dry-run by default)."""
    st.subheader("Pipeline Lab")
    st.caption(
        "Replay a single documentation page through every ingestion stage — raw HTML → exact Qdrant point payload. "
        "Dry-run by default; nothing is written to the index unless you enable the live upsert."
    )

    source_mode = st.radio(
        "Input source",
        ["URL", "Paste raw HTML", "Sample: PySpark API page"],
        horizontal=True,
        key="lab_source_mode",
    )
    source_name = st.text_input("Source name", value="pipeline-lab", key="lab_source_name")
    url = st.text_input("Documentation URL", key="lab_url") if source_mode == "URL" else ""
    html = (
        st.text_area("Raw HTML", height=180, key="lab_html", placeholder="<!doctype html>…")
        if source_mode == "Paste raw HTML"
        else ""
    )
    inject = st.toggle("Also upsert to Qdrant (live write)", value=False, key="lab_inject")
    run_lab = st.button("Run pipeline", type="primary", key="lab_run")

    if not run_lab:
        return

    content_type = "text/html"
    if source_mode == "Sample: PySpark API page":
        url = "sample://sparksession-api"
        html = _SPARK_SAMPLE_HTML
    elif source_mode == "Paste raw HTML":
        if not html.strip():
            st.error("Paste some HTML first.")
            return
        url = "pasted://html"
    else:
        if not url.strip():
            st.error("Enter a documentation URL.")
            return
        with st.spinner("Fetching HTML…"):
            try:
                req = urllib.request.Request(
                    url.strip(), headers={"User-Agent": "DataEngineeringCopilot-pipeline-lab/1.0"}
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    html_bytes = resp.read()
                content_type = resp.headers.get("Content-Type", "text/html")
                html = html_bytes.decode("utf-8", errors="replace")
            except Exception as exc:
                st.error(f"Failed to fetch URL: {exc}")
                return

    stage_events: list[str] = []
    result_box: list = []
    error_box: list = []

    def _run_lab() -> None:
        try:
            lab = build_pipeline_lab(dry_run=not inject)
        except Exception as exc:
            error_box.append(exc)
            return
        loop = _get_service_loop()
        raw = RawDocument(
            source_name=source_name.strip() or "pipeline-lab",
            url=url,
            html=html,
            content_type=content_type,
        )
        future = asyncio.run_coroutine_threadsafe(
            lab.run(raw, on_stage=stage_events.append),
            loop,
        )
        try:
            result_box.append(future.result(timeout=900))
        except Exception as exc:
            error_box.append(exc)

    worker = threading.Thread(target=_run_lab, daemon=True)
    worker.start()

    diagram_ph = st.empty()
    with st.status("Running pipeline…", expanded=True) as status:
        while worker.is_alive():
            diagram_ph.html(
                build_diagram_html(LAB_NODES, LAB_EDGES, _lab_states(list(stage_events)), show_legend=False)
            )
            if stage_events:
                last = LAB_STAGE_TITLES.get(stage_events[-1], stage_events[-1])
                status.update(label=f"Stage {min(len(stage_events), 7)}/7: {last}")
            time.sleep(0.25)

    diagram_ph.html(build_diagram_html(LAB_NODES, LAB_EDGES, _lab_states(list(stage_events)), show_legend=False))

    if error_box:
        st.error(f"Pipeline failed: {error_box[0]}")
        return
    if not result_box:
        st.error("Pipeline finished without a trace.")
        return

    trace = result_box[0]
    status.update(label="✅ Pipeline complete", state="complete")
    st.caption(
        f"Trace for **{trace.raw_document.source_name}** ({trace.raw_document.url}) — "
        f"{'dry-run' if trace.dry_run else 'live write'} · {len(trace.final_chunks)} final chunks"
    )

    for stage in trace.stages:
        title = LAB_STAGE_TITLES.get(stage.name, stage.name)
        with st.status(
            f"{title} — {stage.output_summary}", state="error" if stage.error else "complete", expanded=False
        ):
            if stage.error:
                st.error(stage.error)
            st.caption(f"**Input:** {stage.input_summary}")
            st.caption(f"**Output:** {stage.output_summary}")
            if trace.raw_document is not None:
                _render_lab_stage_payload(stage.name, stage.payload, trace.raw_document.html)


def render_ingestion_tab() -> None:
    """Ingestion Dashboard tab: controls, tabs with live progress, and history."""
    progress = IngestionManager.get_progress()

    # === COMPACT CONTROLS (always visible, outside fragment) ===
    with st.container(border=True):
        ccol1, ccol2, ccol3 = st.columns([2, 1, 1])
        with ccol1:
            selected_sources = st.multiselect(
                "Sources",
                options=[source.name for source in settings.sources],
                default=[source.name for source in settings.sources],
                key="ingest_source_select",
                label_visibility="collapsed",
                placeholder="Select sources...",
            )
        with ccol2:
            max_pages = st.number_input(
                "Max pages",
                min_value=0,
                value=settings.max_pages_per_source,
                step=10,
                help="0 = unlimited (capped by config)",
                key="ingest_max_pages",
                label_visibility="collapsed",
            )
        with ccol3:
            if progress.is_running or progress.success_message or progress.error:
                if progress.is_running:
                    stop = st.button("⏹ Stop", type="primary", width="stretch", key="stop_btn")
                    if stop:
                        IngestionManager.stop()
                        st.rerun()
                else:
                    dismiss = st.button("Dismiss", type="secondary", width="stretch", key="dismiss_btn")
                    if dismiss:
                        # Save to history before clearing
                        if progress.success_message:
                            history = st.session_state.get("ingestion_history", [])
                            history.append(
                                {
                                    "Time": time.strftime("%H:%M:%S"),
                                    "Status": "✅ Completed",
                                    "Pages": progress.total_pages_fetched,
                                    "Chunks": progress.total_chunks_indexed,
                                    "Duration": _format_duration(progress.elapsed_seconds),
                                }
                            )
                            st.session_state.ingestion_history = history[-20:]
                        IngestionManager.reset_status()
                        st.session_state.pop("ingestion_started", None)
                        st.session_state.pop("_ingest_was_running", None)
                        st.rerun()
            else:
                start_disabled = not selected_sources or progress.is_running
                start = st.button(
                    "🔄 Start",
                    type="primary",
                    width="stretch",
                    disabled=start_disabled,
                    key="start_btn",
                )
                if start:
                    if not selected_sources:
                        st.warning("Select at least one source.")
                    else:
                        qdrant_ok, qdrant_msg = _check_qdrant_reachable()
                        if not qdrant_ok:
                            st.error(f"**Cannot start ingestion**\n\n{qdrant_msg}")
                        else:
                            started, error = IngestionManager.start(
                                source_names=tuple(selected_sources),
                                max_pages=int(max_pages) if max_pages > 0 else 0,
                                use_async=True,
                            )
                            if not started:
                                st.warning(error or "Already running.")
                            else:
                                st.success("Started!")
                                st.rerun()

    # === CONTENT (tabs with live progress) ===
    _render_progress_panel()


def render_health_tab() -> None:
    """System Health tab: vector store, Ollama, configuration."""
    st.subheader("System Health")

    # Live service status
    st.markdown("### Service Status")
    qdrant_ok, qdrant_msg = _check_qdrant_reachable()
    ollama_ok, ollama_msg = _check_ollama_reachable()
    langfuse_ok, langfuse_msg = _check_langfuse_reachable()
    deps_ok, deps_msg = _check_deps_fingerprint()

    col_q, col_o, col_l, col_d = st.columns(4)
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
    with col_d:
        if deps_ok:
            st.success("Docker image")
            st.caption(deps_msg)
        else:
            st.error("Docker image STALE")
            st.caption(f"Run `make docker-dev`\n\n{deps_msg}")

    st.divider()

    # Repository stats
    st.markdown("### Vector Store")
    qdrant_ok, _ = _check_qdrant_reachable()
    if qdrant_ok:
        try:
            url = f"{settings.qdrant_url}/collections/{settings.collection_name}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read())
                chunk_count = data.get("result", {}).get("points_count", 0)
                st.metric("Total Chunks Indexed", chunk_count)
        except Exception:
            st.warning("Vector store is connected but returned an error.")
            chunk_count = 0
    else:
        chunk_count = 0
        st.warning("Vector store is not available.\n\n**Start Qdrant:**\n```\ndocker compose up -d qdrant\n```")

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

    with st.expander("Advanced Configuration", expanded=False):
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
            st.caption(f"Application log: `{settings.project_root / 'logs' / 'app.log'}`")

    st.divider()

    # Ingestion history
    st.markdown("### Ingestion History")
    log_path = settings.project_root / "logs" / "app.log"
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            ingestion_lines = [line for line in lines if "ngestion" in line.lower()]
            if ingestion_lines:
                history_rows = []
                for line in reversed(ingestion_lines[-50:]):
                    row = {"raw": line[:200]}
                    # Try to parse JSON-structured log lines
                    try:
                        parts = line.split(" | ", 1)
                        if len(parts) == 2:
                            row["timestamp"] = parts[0].strip()
                            maybe_json = parts[1].strip()
                            if maybe_json.startswith("{"):
                                data = json.loads(maybe_json)
                                row["event"] = data.get("event", data.get("msg", data.get("message", "")))
                                row["chunks"] = data.get("chunks_indexed", data.get("total_chunks", ""))
                                row["pages"] = data.get("pages_fetched", data.get("total_pages", ""))
                                row["source"] = data.get("source_name", data.get("source", ""))
                    except (json.JSONDecodeError, IndexError):
                        pass
                    if "event" not in row:
                        row["event"] = row["raw"][:100]
                    history_rows.append(row)
                st.dataframe(
                    [
                        {
                            "Time": r.get("timestamp", ""),
                            "Event": r.get("event", ""),
                            "Pages": r.get("pages", ""),
                            "Chunks": r.get("chunks", ""),
                            "Source": r.get("source", ""),
                        }
                        for r in history_rows
                    ],
                    width="stretch",
                    hide_index=True,
                    height=min(len(history_rows) * 35 + 38, 400),
                )
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
    col_s4.metric("Avg MRR", f"{summary['avg_proxy_mrr']:.3f}")
    col_s5.metric("Avg Answer Length (words)", summary["avg_answer_length"])

    st.divider()

    # --- Query Difficulty Breakdown ---
    st.markdown("### Query Difficulty Breakdown")
    by_diff = summary.get("by_difficulty", {})
    if by_diff:
        diff_cols = st.columns(3)
        for col, (difficulty, data) in zip(diff_cols, sorted(by_diff.items()), strict=False):
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
            table_data.append(
                {
                    "Query": qm.query[:50] + ("..." if len(qm.query) > 50 else ""),
                    "Difficulty": qm.query_difficulty.capitalize(),
                    "Confidence": f"{qm.confidence_score:.2%}" if qm.was_answered else "—",
                    "Answered": "✅" if qm.was_answered else "❌",
                    "Sources": qm.answer_metrics.source_count
                    if qm.answer_metrics and qm.answer_metrics.source_count
                    else 0,
                    "Answer Words": qm.answer_metrics.answer_length if qm.answer_metrics else "—",
                }
            )
        st.dataframe(table_data, width="stretch", hide_index=True)

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
            "words": [am.answer_length for q in queries_with_answers if (am := q.answer_metrics) is not None],
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

    # Stable per-browser-session identifiers for Langfuse session/user tracking.
    # Restored across reruns within the same browser tab; regenerated per tab.
    if "session_id" not in st.session_state:
        st.session_state.session_id, st.session_state.user_id = _new_session_identifiers()
    st.session_state.setdefault("trace_ids", {})

    st.set_page_config(page_title="DataEngineeringCopilot", layout="wide")
    st.title("📚 DataEngineeringCopilot")
    st.caption("Offline RAG over Spark, Airflow, Databricks, and Delta Lake documentation.")

    # Sidebar: compact status
    progress = IngestionManager.get_progress()
    with st.sidebar:
        st.markdown("### System Status")
        if progress.is_running:
            st.warning(f"Ingestion running ({_format_duration(progress.elapsed_seconds)})")
            task_id = st.session_state.get("ingestion_task_id")
            if task_id:
                st.caption(f"Task: `{task_id}`")
            total_sources = len(progress.source_names) or 1
            effective_max_pages = progress.max_pages_per_source or settings.max_pages_per_source
            sidebar_estimated = effective_max_pages * total_sources
            mini_ratio = min(progress.total_pages_fetched / max(sidebar_estimated, 1), 1.0)
            st.progress(mini_ratio)
            st.caption(
                f"{progress.total_pages_fetched} / {sidebar_estimated} pages  |  {progress.total_chunks_indexed} chunks"
            )
        elif progress.error:
            st.error("Ingestion failed")
        else:
            st.success("Idle")

        # Service indicators
        qdrant_ok, _ = _check_qdrant_reachable(timeout=1.0)
        ollama_ok, _ = _check_ollama_reachable(timeout=1.0)
        deps_ok, deps_msg = _check_deps_fingerprint(timeout=1.0)
        if qdrant_ok:
            st.success("Qdrant: up")
        else:
            st.error("Qdrant: down")
        if ollama_ok:
            st.success("Ollama: up")
        else:
            st.error("Ollama: down")
        if deps_ok:
            st.success("Docker image: fresh")
        else:
            st.error("Docker image: STALE")
            st.caption(f"Run `make docker-dev` to fix. {deps_msg}")

        # Chunk count
        try:
            url = f"{settings.qdrant_url}/collections/{settings.collection_name}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read())
                chunk_count = data.get("result", {}).get("points_count", 0)
                st.metric("Chunks in Store", chunk_count)
        except Exception:
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
    # Stale image warning banner (top of page, highly visible)
    deps_ok, deps_msg = _check_deps_fingerprint(timeout=2.0)
    if not deps_ok:
        st.error(f"**Docker image is STALE** — ingestion will fail. Run `make docker-dev` to rebuild.\n\n{deps_msg}")

    tab_chat, tab_ask, tab_ingest, tab_lab, tab_health, tab_metrics = st.tabs(
        ["💬 Chat", "💬 Ask", "📥 Ingestion", "🧪 Pipeline Lab", "🔧 System Health", "📊 Metrics"]
    )
    with tab_chat:
        render_chat_tab()
    with tab_ask:
        render_qa_tab()
    with tab_ingest:
        render_ingestion_tab()
    with tab_lab:
        render_pipeline_lab_tab()
    with tab_health:
        render_health_tab()
    with tab_metrics:
        render_metrics_tab()


if __name__ == "__main__":
    main()
