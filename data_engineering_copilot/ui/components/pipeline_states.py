"""Pure pipeline-state reducers driving the animated diagrams.

These functions are intentionally free of any Streamlit calls so they can be
exercised hermetically by unit tests.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from typing import Protocol

QUERY_NODES: tuple[str, ...] = ("Rewrite", "Embed", "Retrieve", "Rerank", "Generate")
INGESTION_NODES: tuple[str, ...] = ("HTML Source", "Crawl", "Chunker", "Embedder", "Qdrant Index")


class NodeState(enum.StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


class IngestionProgressLike(Protocol):
    is_running: bool
    success_message: str | None
    error: str | None
    current_url: str
    recent_events: list[dict]


def reduce_query_node_states(
    events: list[str],
    *,
    failed_step: str | None = None,
    completed: bool = False,
) -> dict[str, NodeState]:
    """Derive per-node states from the async RAG ``on_step`` event sequence.

    ``events`` uses the values emitted by ``data_engineering_copilot.services
    async_rag.AsyncRagService.answer``: ``"Rewriting query"``,
    ``"Embedding query"``, ``"Retrieving results"``, ``"Reranking results"``
    and ``"Generating answer"``. The pipeline is monotonic: each event marks
    earlier stages complete and the current retrieval/generation window as
    running. The legacy three-event sequence (``"Embedding query"``,
    ``"Reranking results"``, ``"Generating answer"``) is still supported so
    older callers keep working. Set ``completed=True`` once the answer is
    ready and ``failed_step`` to any node name to surface an error.
    """
    states: dict[str, NodeState] = {node: NodeState.IDLE for node in QUERY_NODES}
    if completed:
        return {node: NodeState.COMPLETE for node in QUERY_NODES}

    done: set[str] = set()
    running: set[str] = set()
    for event in events:
        if event == "Rewriting query":
            running = {"Rewrite"}
        elif event == "Embedding query":
            done |= {"Rewrite"}
            running = {"Embed", "Retrieve"}
        elif event in ("Retrieving results", "Reranking results"):
            done |= {"Embed", "Retrieve"}
            running = {"Rerank"}
        elif event == "Generating answer":
            done |= {"Rerank"}
            running = {"Generate"}

    for node in done:
        states[node] = NodeState.COMPLETE
    for node in running:
        states[node] = NodeState.RUNNING

    if failed_step:
        failed_name = failed_step if failed_step in states else QUERY_NODES[-1]
        # Everything after the failing stage is cancelled; only progress up to
        # the failure is preserved before the red marker.
        failed_idx = QUERY_NODES.index(failed_name)
        for node in QUERY_NODES:
            if QUERY_NODES.index(node) > failed_idx:
                states[node] = NodeState.IDLE
        states[failed_name] = NodeState.ERROR
    return states


def ingestion_node_states(progress: IngestionProgressLike) -> dict[str, NodeState]:
    """Derive ingestion-diagram node states from an ``IngestionProgress``.

    Reads only the structural fields declared in :class:`IngestionProgressLike`
    so ``data_engineering_copilot.ui.streamlit_app.IngestionProgress`` satisfies
    it without an explicit import (avoids a circular dependency).
    """
    states: dict[str, NodeState] = {node: NodeState.IDLE for node in INGESTION_NODES}

    if not progress.is_running and not progress.success_message and not progress.error:
        return states

    if progress.success_message:
        return {node: NodeState.COMPLETE for node in INGESTION_NODES}

    event_types = {event.get("type", "") for event in progress.recent_events}
    if "batch_indexing" in event_types:
        states["HTML Source"] = NodeState.COMPLETE
        states["Crawl"] = NodeState.COMPLETE
        states["Chunker"] = NodeState.COMPLETE
        states["Embedder"] = NodeState.COMPLETE
        states["Qdrant Index"] = NodeState.RUNNING
    elif "batch_embedding" in event_types or "page_indexed" in event_types:
        states["HTML Source"] = NodeState.COMPLETE
        states["Crawl"] = NodeState.COMPLETE
        states["Chunker"] = NodeState.COMPLETE
        states["Embedder"] = NodeState.RUNNING
        states["Qdrant Index"] = NodeState.RUNNING
    elif "fetch_success" in event_types or progress.current_url:
        states["HTML Source"] = NodeState.COMPLETE
        states["Crawl"] = NodeState.RUNNING
        states["Chunker"] = NodeState.RUNNING
    else:
        states["HTML Source"] = NodeState.COMPLETE
        states["Crawl"] = NodeState.RUNNING

    if progress.error:
        # Mark whichever stage was furthest along as failed.
        furthest = "Crawl"
        for node in INGESTION_NODES:
            if states.get(node) in (NodeState.COMPLETE, NodeState.RUNNING):
                furthest = node
        states = {node: NodeState.IDLE for node in INGESTION_NODES}
        for node in INGESTION_NODES:
            if node == furthest:
                states[node] = NodeState.ERROR
                break
            states[node] = NodeState.COMPLETE
    return states


def node_state_palette(states: Mapping[str, NodeState]) -> dict[str, str]:
    """Map node names to their hex accent color for Mermaid graph styling."""
    palette: dict[str, str] = {
        NodeState.COMPLETE: "#15803D",
        NodeState.RUNNING: "#2563EB",
        NodeState.ERROR: "#DC2626",
        NodeState.IDLE: "#64748B",
    }
    return {node: palette[state] for node, state in states.items()}
