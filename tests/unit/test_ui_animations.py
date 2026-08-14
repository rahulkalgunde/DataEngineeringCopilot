"""Unit tests for the animated UI components (hermetic, no infra).

Covers the pure pipeline-state reducers, the HTML/CSS diagram builder, the
Lottie player markup and its fallback layer, the numpy-PCA projection, and the
typewriter generator.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from data_engineering_copilot.ui.components import assets
from data_engineering_copilot.ui.components.animations import (
    _SEARCH_STEPS,
    LOTTIE_WEB_CDN,
    _css_fallback_icon,
    build_animated_metric_html,
    build_diagram_html,
    build_lottie_player_html,
    build_mermaid_html,
    build_vector_scatter_figure,
    compute_pca_3d,
    format_mermaid,
    node_detail_rows,
    stream_answer_text,
)
from data_engineering_copilot.ui.components.pipeline_states import (
    INGESTION_NODES,
    QUERY_NODES,
    NodeState,
    ingestion_node_states,
    reduce_query_node_states,
)

LINEAR_EDGES = [(QUERY_NODES[i], QUERY_NODES[i + 1]) for i in range(len(QUERY_NODES) - 1)]
INGESTION_EDGES = [(INGESTION_NODES[i], INGESTION_NODES[i + 1]) for i in range(len(INGESTION_NODES) - 1)]


@pytest.fixture(autouse=True)
def _undisturbed():
    """No session state is needed for these pure-components; just yield."""
    yield


# ---------------------------------------------------------------------------
# Query pipeline state reduction
# ---------------------------------------------------------------------------


class TestReduceQueryNodeStates:
    def test_idle_when_no_events(self):
        states = reduce_query_node_states([])
        assert set(states) == set(QUERY_NODES)
        assert all(s is NodeState.IDLE for s in states.values())

    def test_rewriting_event_marks_rewrite_running(self):
        states = reduce_query_node_states(["Rewriting query"])
        assert states["Rewrite"] is NodeState.RUNNING
        assert all(states[n] is NodeState.IDLE for n in QUERY_NODES[1:])

    def test_embedding_event_marks_rewrite_done_and_retrieval_window_running(self):
        states = reduce_query_node_states(["Embedding query"])
        assert states["Rewrite"] is NodeState.COMPLETE
        assert states["Embed"] is NodeState.RUNNING
        assert states["Retrieve"] is NodeState.RUNNING
        assert states["Rerank"] is NodeState.IDLE
        assert states["Generate"] is NodeState.IDLE

    def test_retrieving_event_completes_retrieval_and_starts_rerank(self):
        states = reduce_query_node_states(["Rewriting query", "Embedding query", "Retrieving results"])
        assert states["Rewrite"] is NodeState.COMPLETE
        assert states["Embed"] is NodeState.COMPLETE
        assert states["Retrieve"] is NodeState.COMPLETE
        assert states["Rerank"] is NodeState.RUNNING
        assert states["Generate"] is NodeState.IDLE

    def test_rerank_event_completes_retrieval(self):
        states = reduce_query_node_states(["Embedding query", "Reranking results"])
        assert states["Embed"] is NodeState.COMPLETE
        assert states["Retrieve"] is NodeState.COMPLETE
        assert states["Rerank"] is NodeState.RUNNING
        assert states["Generate"] is NodeState.IDLE

    def test_generate_event_is_last_running_stage(self):
        states = reduce_query_node_states(["Embedding query", "Reranking results", "Generating answer"])
        assert states["Generate"] is NodeState.RUNNING
        assert all(states[n] is NodeState.COMPLETE for n in QUERY_NODES[:-1])

    def test_legacy_three_event_sequence_still_supported(self):
        states = reduce_query_node_states(["Embedding query", "Reranking results", "Generating answer"])
        assert states["Rewrite"] is NodeState.COMPLETE
        assert states["Embed"] is NodeState.COMPLETE
        assert states["Retrieve"] is NodeState.COMPLETE
        assert states["Rerank"] is NodeState.COMPLETE
        assert states["Generate"] is NodeState.RUNNING

    def test_completed_flag_clears_all_to_complete(self):
        states = reduce_query_node_states([], completed=True)
        assert all(s is NodeState.COMPLETE for s in states.values())

    def test_failed_step_surfaces_error(self):
        states = reduce_query_node_states(["Embedding query", "Reranking results"], failed_step="Retrieve")
        assert states["Retrieve"] is NodeState.ERROR
        # Stages after the failure stay idle.
        assert states["Rerank"] is NodeState.IDLE
        assert states["Generate"] is NodeState.IDLE

    def test_unknown_failed_step_defaults_to_generate(self):
        states = reduce_query_node_states([], failed_step="nope")
        assert states["Generate"] is NodeState.ERROR


class TestNodeDetailRows:
    def test_rewrite_rows(self):
        rows = dict(
            node_detail_rows(
                "Rewrite",
                {
                    "original_query": "how do I union?",
                    "rewritten_query": "pyspark union two dataframes",
                    "intent": "code",
                    "hyde_query": "hypothetical document",
                    "expansions": ["a", "b", "c"],
                },
            )
        )
        assert rows["Input query"] == "how do I union?"
        assert rows["Effective query"] == "pyspark union two dataframes"
        assert rows["Intent"] == "code"
        assert rows["HyDE query"] == "hypothetical document"
        assert "Query variants" in rows

    def test_rewrite_without_intent_or_hyde_omits_rows(self):
        rows = dict(node_detail_rows("Rewrite", {"original_query": "q", "rewritten_query": "q2"}))
        assert "Intent" not in rows
        assert "HyDE query" not in rows

    def test_embed_rows(self):
        rows = dict(node_detail_rows("Embed", {"variants": 3, "dimension": 8, "l2_norm": 1.23456}))
        assert rows["Variants embedded"] == "3"
        assert rows["Vector dimension"] == "8"
        assert rows["L2 norm"] == "1.2346"

    def test_embed_handles_missing_l2_norm(self):
        rows = dict(node_detail_rows("Embed", {"variants": 1, "dimension": 8}))
        assert rows["L2 norm"] == "?"

    def test_retrieve_rows(self):
        payload = {"pool_size": 40, "candidates": [{"rank": i} for i in range(5)]}
        rows = dict(node_detail_rows("Retrieve", payload))
        assert rows["Pool size"] == "40"
        assert rows["Candidates"] == "5"

    def test_retrieve_defaults_pool_size_to_candidates(self):
        payload = {"candidates": [{"rank": 0}, {"rank": 1}]}
        rows = dict(node_detail_rows("Retrieve", payload))
        assert rows["Pool size"] == "2"

    def test_rerank_rows(self):
        payload = {"enabled": True, "pool_size": 40, "top_k": 12, "final_top_k": 6, "compressed_dropped": 3}
        rows = dict(node_detail_rows("Rerank", payload))
        assert rows["Reranker"] == "enabled"
        assert rows["Pool size"] == "40"
        assert rows["Top-K"] == "12"
        assert rows["Final top-K"] == "6"
        assert rows["Dropped by budget"] == "3"

    def test_generate_rows(self):
        payload = {"context_chunks": 6, "context_chars": 2000, "prompt_chars": 4000, "model": "stub"}
        rows = dict(node_detail_rows("Generate", payload))
        assert rows["Context chunks"] == "6"
        assert rows["Context chars"] == "2000"
        assert rows["Prompt chars"] == "4000"
        assert rows["Model"] == "stub"


# ---------------------------------------------------------------------------
# Ingestion pipeline state reduction
# ---------------------------------------------------------------------------


class _ProgressStub:
    """Structural stand-in for IngestionProgress (satisfies the protocol)."""

    def __init__(self, **overrides: object) -> None:
        self.is_running: bool = bool(overrides.get("is_running", False))
        self.success_message: str | None = overrides.get("success_message")  # type: ignore[assignment]
        self.error: str | None = overrides.get("error")  # type: ignore[assignment]
        self.current_url: str = str(overrides.get("current_url", ""))
        self.recent_events: list[dict] = list(overrides.get("recent_events", []))  # type: ignore[arg-type]


def _progress(**overrides) -> _ProgressStub:
    return _ProgressStub(**overrides)


class TestIngestionNodeStates:
    def test_idle_when_nothing_happening(self):
        states = ingestion_node_states(_progress())
        assert all(s is NodeState.IDLE for s in states.values())

    def test_success_marks_everything_complete(self):
        states = ingestion_node_states(_progress(success_message="Done"))
        assert all(s is NodeState.COMPLETE for s in states.values())

    def test_running_with_fetches_highlights_crawl(self):
        states = ingestion_node_states(
            _progress(is_running=True, current_url="https://x", recent_events=[{"type": "fetch_success"}])
        )
        assert states["HTML Source"] is NodeState.COMPLETE
        assert states["Crawl"] is NodeState.RUNNING
        assert states["Chunker"] is NodeState.RUNNING
        assert states["Embedder"] is NodeState.IDLE

    def test_running_with_embedding_batch(self):
        states = ingestion_node_states(_progress(is_running=True, recent_events=[{"type": "batch_embedding"}]))
        assert states["Chunker"] is NodeState.COMPLETE
        assert states["Embedder"] is NodeState.RUNNING
        assert states["Qdrant Index"] is NodeState.RUNNING

    def test_running_with_indexing_batch(self):
        states = ingestion_node_states(_progress(is_running=True, recent_events=[{"type": "batch_indexing"}]))
        assert states["Embedder"] is NodeState.COMPLETE
        assert states["Qdrant Index"] is NodeState.RUNNING

    def test_error_flags_furthest_stage(self):
        states = ingestion_node_states(
            _progress(
                error="boom",
                recent_events=[
                    {"type": "fetch_success"},
                    {"type": "page_indexed"},
                    {"type": "batch_embedding"},
                ],
            )
        )
        # Furthest advanced node is Qdrant Index (running alongside Embedder).
        assert states["Qdrant Index"] is NodeState.ERROR
        assert states["Embedder"] is NodeState.ERROR or states["Qdrant Index"] is NodeState.ERROR
        assert states["HTML Source"] is NodeState.COMPLETE
        assert states["Chunker"] is NodeState.COMPLETE


# ---------------------------------------------------------------------------
# Diagram / Mermaid builders
# ---------------------------------------------------------------------------


class TestDiagramHtml:
    def test_nodes_and_state_classes_rendered(self):
        states = reduce_query_node_states(["Embedding query"])
        html = build_diagram_html(QUERY_NODES, LINEAR_EDGES, states, show_legend=False)
        assert 'data-node="Rewrite"' in html
        assert 'data-node="Generate"' in html
        assert "dec-node--running" in html
        assert "dec-node--complete" in html

    def test_edges_flow_only_into_running_node(self):
        """The flow animation must not start on an edge out of a running node:
        dots only stream into the currently-active step, never toward an idle
        successor (e.g. Rerank -> Generate must not flow while Rerank runs)."""
        cases = {
            "Rewrite": (["Rewriting query"], set(), {"0-1"}),
            "Embed/Retrieve": (
                ["Embedding query"],
                {"0-1", "1-2"},
                {"2-3"},
            ),
            "Rerank": (
                ["Retrieving results", "Reranking results"],
                {"2-3"},
                {"3-4"},
            ),
            "Generate": (["Generating answer"], {"3-4"}, set()),
        }
        for stage, (events, flowing, not_flowing) in cases.items():
            states = reduce_query_node_states(events)
            html = build_diagram_html(QUERY_NODES, LINEAR_EDGES, states, show_legend=False)
            for edge in flowing:
                assert f'class="dec-edge dec-edge--flow" data-edge="{edge}"' in html, (
                    f"edge {edge} should flow during {stage}"
                )
            for edge in not_flowing:
                assert f'class="dec-edge" data-edge="{edge}"' in html, f"edge {edge} should be still during {stage}"

    def test_legend_omitted_when_disabled(self):
        html = build_diagram_html(QUERY_NODES, LINEAR_EDGES, reduce_query_node_states([]), show_legend=False)
        assert 'class="dec-legend"' not in html

    def test_error_node_renders_red(self):
        states = reduce_query_node_states([], failed_step="Retrieve")
        html = build_diagram_html(QUERY_NODES, LINEAR_EDGES, states)
        assert 'data-node="Retrieve" data-state="error"' in html
        assert "dec-node--error" in html


class TestMermaid:
    def test_format_contains_flowchart_and_styles(self):
        states = reduce_query_node_states(["Generating answer"])
        spec = format_mermaid(QUERY_NODES, LINEAR_EDGES, states)
        assert spec.startswith("flowchart LR")
        assert "--> " in spec
        assert "#2563EB" in spec  # running styling
        assert "#15803D" in spec  # complete styling

    def test_build_mermaid_html_uses_cdn(self):
        html = build_mermaid_html(QUERY_NODES, LINEAR_EDGES, reduce_query_node_states([]))
        assert '<div class="mermaid">' in html.replace("&quot;", '"') or 'class="mermaid"' in html
        assert "mermaid.min.js" in html


# ---------------------------------------------------------------------------
# Lottie player + fallback
# ---------------------------------------------------------------------------


class TestLottiePlayer:
    def test_bundled_animations_validate(self):
        assert assets.validate_animations() == []
        assert set(assets.ANIMATIONS) == {"parse", "embed", "search", "generate"}

    def test_player_embeds_cdn_and_fallback_icon(self):
        html = build_lottie_player_html("search", height=64)
        assert LOTTIE_WEB_CDN in html
        assert "dec-css-search" in html
        assert "dec-radar" in html

    def test_player_inlines_animation_data_unescaped_scripts(self):
        html = build_lottie_player_html("generate")
        assert "lottie.loadAnimation" in html
        assert "var data=" in html
        assert "dot-2" in html  # bundled layer names are inlined in the payload
        assert html.count("</script>") == 1

    def test_css_fallback_icon_for_all_kinds(self):
        for kind in ("parse", "embed", "search", "generate"):
            assert _css_fallback_icon(kind).strip()
        assert "dec-radar" in _css_fallback_icon("unknown")


# ---------------------------------------------------------------------------
# 3D projection + Plotly figure
# ---------------------------------------------------------------------------


class TestPca3d:
    def test_shape_and_determinism(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(6, 8))
        proj_a = compute_pca_3d(vectors)
        proj_b = compute_pca_3d(vectors)
        assert proj_a.shape == (6, 3)
        np.testing.assert_allclose(proj_a, proj_b)

    def test_preserves_pairwise_distances_for_rank3_embedded_points(self):
        rng = np.random.default_rng(7)
        basis = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]], dtype=float)
        coords = rng.normal(size=(5, 3))
        points = coords @ basis.T
        proj = compute_pca_3d(points)
        assert np.allclose(pairwise_dist(proj), pairwise_dist(points[:5]), rtol=1e-6, atol=1e-9)


class TestVectorScatterFigure:
    def test_shape_and_frames(self):
        rng = np.random.default_rng(3)
        query = rng.normal(size=6).tolist()
        chunks = [rng.normal(size=6).tolist() for _ in range(4)]
        labels = ["a", "b", "c", "d"]
        scores = [0.2, 0.4, 0.6, 0.8]
        fig: Any = build_vector_scatter_figure(query, chunks, labels, scores)
        data = fig.data
        frames = fig.frames
        assert len(data) == 2
        assert len(frames) == _SEARCH_STEPS
        query_trace = data[0]
        assert len(query_trace.x) == 1


def pairwise_dist(points: np.ndarray) -> np.ndarray:
    diff = np.asarray(points)[:, None, :] - np.asarray(points)[None, :, :]
    return np.linalg.norm(diff, axis=2)


# ---------------------------------------------------------------------------
# Typewriter + animated metric
# ---------------------------------------------------------------------------


class TestTypewriter:
    def test_reassembles_full_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert "".join(stream_answer_text(text, chunk_chars=4, delay=0)) == text

    def test_respects_chunk_size(self):
        text = "abcdefghij"
        chunks = list(stream_answer_text(text, chunk_chars=3, delay=0))
        assert chunks == ["abc", "def", "ghi", "j"]
        assert all(len(c) <= 3 for c in chunks)

    def test_delay_sleeps(self):
        started = time.monotonic()
        list(stream_answer_text("hello", chunk_chars=2, delay=0.002))
        assert time.monotonic() - started >= 0.002


class TestAnimatedMetric:
    def test_target_present_for_no_js_fallback(self):
        html = build_animated_metric_html("Pages", 1234)
        assert 'data-target="1234.0"' in html
        assert '<span class="dec-metric-label">Pages</span>' in html.replace("&quot;", '"') or "Pages" in html

    def test_float_digits(self):
        html = build_animated_metric_html("Latency", 12.345, digits=2)
        assert 'data-digits="2"' in html
        assert "12.35" in html  # static fallback uses rounded display
