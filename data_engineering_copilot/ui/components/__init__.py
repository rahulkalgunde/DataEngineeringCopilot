"""Animated UI components for the Streamlit visualizer.

Exposes the pure pipeline-state reducers and the Streamlit renderers used by
``data_engineering_copilot.ui.streamlit_app``.
"""

from data_engineering_copilot.ui.components.pipeline_states import (
    INGESTION_NODES,
    QUERY_NODES,
    NodeState,
    ingestion_node_states,
    reduce_query_node_states,
)

__all__ = [
    "INGESTION_NODES",
    "QUERY_NODES",
    "NodeState",
    "ingestion_node_states",
    "reduce_query_node_states",
]
