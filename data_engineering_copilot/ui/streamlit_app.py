from __future__ import annotations

from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import Callable

import streamlit as st

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


def ingestion_log_path() -> Path:
    return settings.project_root / "logs" / "ingestion_refresh.log"


def append_ingestion_log(log_path: Path, event: IngestionEvent) -> None:
    if not settings.logging_enabled:
        return
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


def main() -> None:
    logger.info("Streamlit app render started")
    st.set_page_config(page_title="DataEngineeringCopilot", layout="wide")
    st.title("DataEngineeringCopilot")
    st.caption("Offline RAG over Spark, Airflow, Databricks, and Delta Lake documentation.")

    with st.sidebar:
        st.subheader("Repository")
        try:
            chunk_count = vector_store().count()
        except VectorStoreReadError as exc:
            chunk_count = 0
            logger.exception("Streamlit sidebar vector count failed")
            st.warning(str(exc))
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
        source_names = tuple(source.name for source in settings.sources)
        selected_source_names = tuple(
            st.multiselect(
                "Sources to ingest",
                options=source_names,
                default=source_names,
            )
        )
        max_pages = st.number_input(
            "Max pages per source",
            min_value=1,
            max_value=5000,
            value=min(settings.max_pages_per_source, 5000),
            step=10,
        )
        with st.expander("Documentation sources"):
            for source in settings.sources:
                st.markdown(f"**{source.name}**")
                st.write(f"Start URLs: {len(source.start_urls)}")
                for url in source.start_urls:
                    st.markdown(f"- [{url}]({url})")
                st.write(f"Allowed domains: `{', '.join(source.allowed_domains)}`")
                if source.url_prefixes:
                    st.write("URL prefixes:")
                    for prefix in source.url_prefixes:
                        st.markdown(f"- `{prefix}`")
                else:
                    st.write("URL prefixes: all paths on allowed domains")
                st.caption(f"Refresh limit: up to {int(max_pages)} HTML pages for this source")
        refresh_disabled = not selected_source_names
        if refresh_disabled:
            st.warning("Select at least one documentation source to refresh.")
        if st.button(
            "Refresh Documentation",
            type="secondary",
            use_container_width=True,
            disabled=refresh_disabled,
        ):
            log_path = ingestion_log_path()
            status_box = st.empty()
            metrics_box = st.empty()
            urls_box = st.empty()
            log_box = st.empty()
            recent_urls: list[str] = []
            pages_by_source: dict[str, int] = {}
            chunks_by_source: dict[str, int] = {}

            def handle_ingestion_event(event: IngestionEvent) -> None:
                append_ingestion_log(log_path, event)
                if event.event_type == "fetch_success":
                    pages_by_source[event.source_name] = event.pages_fetched
                if event.event_type == "page_indexed":
                    chunks_by_source[event.source_name] = chunks_by_source.get(event.source_name, 0) + event.chunks_indexed
                if event.url and event.event_type in {"fetch_start", "fetch_success", "fetch_error"}:
                    label = "fetching" if event.event_type == "fetch_start" else event.event_type.replace("_", " ")
                    recent_urls.insert(0, f"- `{label}` [{event.url}]({event.url})")
                    del recent_urls[25:]

                total_pages = sum(pages_by_source.values())
                total_chunks = sum(chunks_by_source.values())
                status_box.info(event.message)
                metrics_box.write(f"HTML pages fetched: `{total_pages}` | Chunks indexed: `{total_chunks}`")
                if recent_urls:
                    urls_box.markdown("**Recent HTML page URLs**\n\n" + "\n".join(recent_urls))
                log_box.caption(f"Refresh log: `{log_path}`")

            with st.spinner("Crawling documentation and updating ChromaDB..."):
                try:
                    indexed_chunks = run_ingestion_refresh(
                        max_pages_per_source=int(max_pages),
                        source_names=selected_source_names,
                        on_event=handle_ingestion_event,
                    )
                except Exception as exc:
                    logger.exception("Streamlit ingestion refresh failed")
                    st.error(f"Ingestion failed: {exc}")
                    log_box.caption(f"Refresh log: `{log_path}`")
                else:
                    rag_service.clear()
                    vector_store.clear()
                    st.success(f"Refresh complete. Indexed or updated {indexed_chunks} chunks.")
                    st.caption(f"Refresh log saved to: `{log_path}`")
                    st.rerun()

    question = st.text_area("Question", placeholder="How do I configure Spark dynamic allocation?", height=120)
    ask = st.button("Ask", type="primary")

    if ask:
        if not question.strip():
            logger.info("Streamlit ask ignored because question was empty")
            st.warning("Enter a question.")
            return

        logger.info("Streamlit ask started question=%r", question.strip()[:200])
        with st.spinner("Searching local repository and asking Ollama..."):
            answer = rag_service().answer(question.strip())
        logger.info(
            "Streamlit ask completed confidence=%.4f sources=%s answer_chars=%s",
            answer.confidence,
            len(answer.sources),
            len(answer.text),
        )

        st.subheader("Answer")
        st.write(answer.text)
        st.caption(f"Confidence: {answer.confidence:.2f}")

        if answer.sources:
            st.subheader("Sources")
            for source in answer.sources:
                st.markdown(f"- [{source.title}]({source.url})")


if __name__ == "__main__":
    main()
