"""Animated UI renderers for the RAG visualizer.

Every renderer is exception-guarded and degrades to a static equivalent
(dataframe, plain metric, CSS icon) rather than ever raising into the app.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from data_engineering_copilot.ui.components import assets
from data_engineering_copilot.ui.components.pipeline_states import NodeState

if TYPE_CHECKING:
    import plotly.graph_objects as go

LOTTIE_WEB_CDN = "https://cdn.jsdelivr.net/npm/lottie-web@5.13.0/build/player/lottie.min.js"
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

_STATE_CLASS = {
    NodeState.IDLE: "dec-node--idle",
    NodeState.RUNNING: "dec-node--running",
    NodeState.COMPLETE: "dec-node--complete",
    NodeState.ERROR: "dec-node--error",
}

_SEARCH_STEPS = 12


# ---------------------------------------------------------------------------
# Pipeline diagram (pure HTML/CSS, renders without JavaScript)
# ---------------------------------------------------------------------------


def _diagram_css() -> str:
    return """
.dec-diagram{display:flex;align-items:center;flex-wrap:wrap;gap:8px;width:100%;padding:4px 0}
.dec-node{display:flex;flex-direction:column;align-items:center;gap:4px;min-width:96px;padding:8px 10px;border:2px solid #CBD5E1;border-radius:10px;background:#F8FAFC;font-size:12px;color:#475569;text-align:center;position:relative}
.dec-node .dec-dot{width:10px;height:10px;border-radius:50%;background:#94A3B8}
.dec-node--running{border-color:#2563EB;background:#EFF6FF;color:#1E3A8A;animation:decPulse 1.6s ease-in-out infinite}
.dec-node--running .dec-dot{background:#3B82F6;animation:decDot 1.6s ease-in-out infinite}
.dec-node--complete{border-color:#22C55E;background:#F0FDF4;color:#166534}
.dec-node--complete .dec-dot{background:#22C55E}
.dec-node--error{border-color:#EF4444;background:#FEF2F2;color:#991B1B}
.dec-node--error .dec-dot{background:#EF4444}
.dec-edge{flex:1 1 26px;min-width:26px;height:3px;background:#CBD5E1;position:relative}
.dec-edge--flow{background:#93C5FD}
.dec-edge--flow::before,.dec-edge--flow::after{content:"";position:absolute;top:-3px;width:9px;height:9px;border-radius:50%;background:#2563EB;animation:decFlow 1.2s linear infinite}
.dec-edge--flow::after{background:#60A5FA;animation-delay:.6s}
.dec-legend{display:flex;gap:14px;align-items:center;margin:6px 0 2px;font-size:11px;color:#64748B;flex-wrap:wrap}
.dec-legend span{display:inline-flex;align-items:center;gap:4px}
.dec-legend i{width:8px;height:8px;border-radius:50%;display:inline-block}
@keyframes decPulse{0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,0)}50%{box-shadow:0 0 0 7px rgba(59,130,246,.22)}}
@keyframes decDot{0%,100%{transform:scale(1)}50%{transform:scale(1.45)}}
@keyframes decFlow{0%{left:-2px;opacity:0}15%{opacity:1}85%{opacity:1}100%{left:calc(100% - 8px);opacity:0}}
"""


def _edge_names(edges: Sequence[tuple[str, str]], running: set[str]) -> list[bool]:
    return [(a in running or b in running) for a, b in edges]


def build_diagram_html(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    states: Mapping[str, NodeState],
    *,
    show_legend: bool = True,
) -> str:
    """Return a self-contained HTML/CSS pipeline diagram string.

    Node state drives colors and the pulsing animation; edges adjacent to a
    running node render flowing data packets. Works without JavaScript.
    """
    running = {node for node, state in states.items() if state is NodeState.RUNNING}
    flows = _edge_names(edges, running)

    parts = [f"<style>{_diagram_css()}</style>", '<div class="dec-diagram">']
    node_index = {node: idx for idx, node in enumerate(nodes)}
    for idx, node in enumerate(nodes):
        state = states.get(node, NodeState.IDLE)
        cls = _STATE_CLASS[state]
        parts.append(
            f'<div class="dec-node {cls}" data-node="{node}" data-state="{state.value}">'
            f'<span class="dec-dot"></span><span>{node}</span></div>'
        )
        if idx < len(nodes) - 1:
            flow = " dec-edge--flow" if flows[idx] else ""
            parts.append(
                f'<div class="dec-edge{flow}" data-edge="{node_index[node]}-{node_index[nodes[idx + 1]]}"></div>'
            )
    parts.append("</div>")

    if show_legend:
        parts.append(
            '<div class="dec-legend">'
            '<span><i style="background:#94A3B8"></i>Idle</span>'
            '<span><i style="background:#3B82F6"></i>Processing</span>'
            '<span><i style="background:#22C55E"></i>Complete</span>'
            '<span><i style="background:#EF4444"></i>Failed</span>'
            "</div>"
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Mermaid.js optional view (JS/CDN; safe fallback to the CSS diagram)
# ---------------------------------------------------------------------------


def format_mermaid(nodes: Sequence[str], edges: Sequence[tuple[str, str]], states: Mapping[str, NodeState]) -> str:
    """Render the diagram as a Mermaid ``flowchart LR`` string."""
    node_ids = [f"N{i}" for i in range(len(nodes))]
    lines = ["flowchart LR"]
    for idx, node in enumerate(nodes):
        lines.append(f'  {node_ids[idx]}["{node}"]')
    for a, b in edges:
        lines.append(f"  {node_ids[nodes.index(a)]} --> {node_ids[nodes.index(b)]}")
    for idx, node in enumerate(nodes):
        style = _mermaid_fill(states.get(node, NodeState.IDLE))
        lines.append(f"  style {node_ids[idx]} {style}")
    return "\n".join(lines)


def _mermaid_fill(state: NodeState) -> str:
    fill: dict[NodeState, str] = {
        NodeState.COMPLETE: "fill:#15803D,stroke:#15803D,color:#ffffff",
        NodeState.RUNNING: "fill:#2563EB,stroke:#2563EB,color:#ffffff",
        NodeState.ERROR: "fill:#DC2626,stroke:#DC2626,color:#ffffff",
        NodeState.IDLE: "fill:#94A3B8,stroke:#64748B,color:#ffffff",
    }
    return fill[state]


def build_mermaid_html(nodes: Sequence[str], edges: Sequence[tuple[str, str]], states: Mapping[str, NodeState]) -> str:
    """HTML that loads Mermaid from CDN and renders the graph. Falls back to a
    plain note when the CDN is unreachable (JS disabled / offline)."""
    spec = format_mermaid(nodes, edges, states)
    return (
        f"<style>.dec-mermaid-note{{padding:12px;border:1px dashed #94A3B8;border-radius:8px;"
        f"font-size:12px;color:#64748B}}</style>"
        f'<div class="mermaid">{spec}</div>'
        f"<script>(function(){{"
        f"function init(){{"
        f"if(window.mermaid){{mermaid.initialize({{startOnLoad:true,theme:'neutral'}});mermaid.run();return;}}"
        f"var s=document.createElement('script');"
        f"s.src={json.dumps(MERMAID_CDN)};"
        f"s.onload=function(){{mermaid.initialize({{startOnLoad:true,theme:'neutral'}});try{{mermaid.run();}}catch(e){{_noteMermaid();}}}};"
        f"s.onerror=_noteMermaid;document.head.appendChild(s);"
        f"}}"
        f"function _noteMermaid(){{var el=document.querySelector('.mermaid');"
        f"if(el)el.innerHTML='<div class=&quot;dec-mermaid-note&quot;>Mermaid unavailable &mdash; offline? Use the CSS diagram.</div>';}}"
        f"init();"
        f"}})();</script>"
    )


# ---------------------------------------------------------------------------
# Lottie badges (self-built player; CSS/SVG icon fallback on any failure)
# ---------------------------------------------------------------------------


def _badge_css() -> str:
    return """
.dec-badge{width:100%;display:flex;align-items:center;justify-content:center}
.dec-scanbox{position:relative;width:56px;height:56px;border:2px solid #94A3B8;border-radius:6px;background:#F8FAFC}
.dec-scanline{position:absolute;left:4px;right:4px;top:8px;height:3px;border-radius:2px;background:#3B82F6;animation:decScan 1.4s ease-in-out infinite}
.dec-grid{display:grid;grid-template-columns:repeat(3,10px);gap:7px}
.dec-grid i{width:10px;height:10px;border-radius:50%;background:#6366F1;animation:decDot 1.2s ease-in-out infinite}
.dec-radar{width:52px;height:52px;border-radius:50%;border:2px solid #38BDF8;background:conic-gradient(from 0deg,rgba(56,189,248,.4),transparent 90deg);animation:decSpin 1.6s linear infinite}
.dec-typing{display:flex;gap:7px}
.dec-typing i{width:10px;height:10px;border-radius:50%;background:#10B981;animation:decDot 1.1s ease-in-out infinite}
@keyframes decScan{0%,100%{top:8px}50%{top:46px}}
@keyframes decSpin{to{transform:rotate(360deg)}}
"""


def _css_fallback_icon(kind: str) -> str:
    grid_i = "".join(f'<i style="animation-delay:{i * 0.12}s"></i>' for i in range(9))
    typing_i = "".join(f'<i style="animation-delay:{i * 0.18}s"></i>' for i in range(3))
    return {
        "parse": '<div class="dec-scanbox"><span class="dec-scanline"></span></div>',
        "embed": f'<div class="dec-grid">{grid_i}</div>',
        "search": '<div class="dec-radar"></div>',
        "generate": f'<div class="dec-typing">{typing_i}</div>',
    }.get(kind, '<div class="dec-radar"></div>')


def build_lottie_player_html(kind: str, *, height: int = 96) -> str:
    """Self-contained Lottie player. Loads ``lottie-web`` from CDN on demand and
    shows the bundled animation; on any failure the CSS icon remains visible."""
    anim_data = assets.ANIMATIONS.get(kind, assets.ANIMATIONS["search"])
    json_str = json.dumps(anim_data).replace("</", "<\\/")
    return (
        f"<style>{_badge_css()}</style>"
        f'<div class="dec-badge">'
        f'<div id="dec-lottie-{kind}" style="display:none;width:100%;height:{height}px"></div>'
        f'<div id="dec-css-{kind}" style="width:100%;height:{height}px;display:flex;'
        f'align-items:center;justify-content:center">{_css_fallback_icon(kind)}</div>'
        f"</div>"
        f"<script>(function(){{"
        f"var host=document.getElementById('dec-lottie-{kind}');"
        f"var cssEl=document.getElementById('dec-css-{kind}');"
        f"var data={json_str};"
        f"function showLottie(){{try{{"
        f"window.lottie.loadAnimation({{container:host,renderer:'svg',loop:true,autoplay:true,animationData:data}});"
        f"cssEl.style.display='none';host.style.display='flex';"
        f"}}catch(e){{}}}}"
        f"if(window.lottie){{showLottie();return;}}"
        f"var s=document.createElement('script');"
        f"s.src={json.dumps(LOTTIE_WEB_CDN)};"
        f"s.onload=showLottie;s.onerror=function(){{}};document.head.appendChild(s);"
        f"}})();</script>"
    )


# ---------------------------------------------------------------------------
# Stepper (st.status reveal, live-updating in placeholders)
# ---------------------------------------------------------------------------


def render_stepper(
    placeholders: Sequence,
    stages: Sequence[tuple[str, NodeState, str]],
    details: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> None:
    """Render one ``st.status`` per stage into a stable placeholder.

    ``stages`` entries are ``(label, state, lottie_kind)``. Idle stages are
    cleared; running stages expand with a Lottie micro-animation; complete and
    error stages reveal their final output. ``details`` is an optional per-node
    mapping of ``(kind, payload)`` snapshots (from the RAG service's
    ``on_step_detail`` callback) rendered inside each completed stage. All
    renders are guarded so a failure falls back to a plain caption.
    """
    import streamlit as st

    payloads_by_node = details or {}
    for holder, (label, state, kind) in zip(placeholders, stages, strict=False):
        if state is NodeState.IDLE:
            holder.empty()
            continue
        st_state = {
            NodeState.RUNNING: "running",
            NodeState.COMPLETE: "complete",
            NodeState.ERROR: "error",
        }[state]
        try:
            with holder.status(label, state=st_state, expanded=state is NodeState.RUNNING):
                if state is NodeState.RUNNING:
                    render_lottie_badge(kind)
                elif state is NodeState.COMPLETE:
                    st.text("✓ Done")
                    payloads = payloads_by_node.get(label) or ()
                    if payloads:
                        render_node_details(label, payloads[-1])
                else:
                    st.text("✗ Failed")
        except Exception:
            holder.caption(f"{label}: {state.value}")


def _payload_list(payload: Mapping[str, object], key: str) -> list[object]:
    """Coerce a detail-payload list field to ``list`` (guard against None/non-list)."""
    value = payload.get(key)
    return list(value) if isinstance(value, (list, tuple)) else []


def node_detail_rows(node: str, payload: Mapping[str, object]) -> list[tuple[str, str]]:
    """Map a per-node detail payload to ``(label, value)`` display rows.

    Pure (no Streamlit) so the mapping is unit-testable.
    """
    rows: list[tuple[str, str]] = []
    if node == "Rewrite":
        rows.append(("Input query", str(payload.get("original_query", ""))))
        rows.append(("Effective query", str(payload.get("rewritten_query", ""))))
        if payload.get("intent"):
            rows.append(("Intent", str(payload["intent"])))
        if payload.get("hyde_query"):
            rows.append(("HyDE query", str(payload["hyde_query"])))
        expansions = _payload_list(payload, "expansions")
        if len(expansions) > 1:
            rows.append(("Query variants", f"{len(expansions)} (original + decomposed/expanded)"))
    elif node == "Embed":
        rows.append(("Variants embedded", str(payload.get("variants", "?"))))
        rows.append(("Vector dimension", str(payload.get("dimension", "?"))))
        norm = payload.get("l2_norm")
        rows.append(("L2 norm", f"{norm:.4f}" if isinstance(norm, (int, float)) else "?"))
    elif node == "Retrieve":
        candidates = _payload_list(payload, "candidates")
        rows.append(("Pool size", str(payload.get("pool_size", len(candidates)))))
        rows.append(("Candidates", str(len(candidates))))
    elif node == "Rerank":
        rows.append(("Reranker", "enabled" if payload.get("enabled") else "disabled"))
        rows.append(("Pool size", str(payload.get("pool_size", "?"))))
        rows.append(("Top-K", str(payload.get("top_k", "?"))))
        rows.append(("Final top-K", str(payload.get("final_top_k", "?"))))
        rows.append(("Dropped by budget", str(payload.get("compressed_dropped", 0))))
    elif node == "Generate":
        rows.append(("Context chunks", str(payload.get("context_chunks", "?"))))
        rows.append(("Context chars", str(payload.get("context_chars", "?"))))
        rows.append(("Prompt chars", str(payload.get("prompt_chars", "?"))))
        rows.append(("Model", str(payload.get("model", "?"))))
    return rows


def render_node_details(node: str, payload: Mapping[str, object]) -> None:
    """Render a single per-node detail payload inside an open ``st.status``."""
    import streamlit as st

    for label, value in node_detail_rows(node, payload):
        st.markdown(f"- **{label}:** `{value}`")

    if node == "Rewrite":
        st.caption("Input → Output")
        st.code(str(payload.get("original_query", "")), language="text")
        st.code(str(payload.get("rewritten_query", "")), language="text")
        expansions = _payload_list(payload, "expansions")
        if len(expansions) > 1:
            with st.expander(f"All {len(expansions)} query variants"):
                for query in expansions:
                    st.caption(str(query))
    elif node == "Retrieve":
        candidates = [c for c in _payload_list(payload, "candidates") if isinstance(c, Mapping)]
        if candidates:
            rows = [
                {
                    "rank": c.get("rank"),
                    "source": c.get("source_name"),
                    "title": (str(c.get("title")) or "")[:48],
                    "confidence": c.get("confidence"),
                    "distance": (
                        round(float(c["distance"]), 4) if isinstance(c.get("distance"), (int, float)) else None
                    ),
                    "words": c.get("word_count"),
                    "chunk_id": (str(c.get("chunk_id")) or "")[:12],
                }
                for c in candidates[:20]
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            with st.expander(f"Top-{min(3, len(candidates))} snippets"):
                for c in candidates[:3]:
                    st.caption(f"#{c.get('rank')} — {c.get('title')}")
                    st.text(str(c.get("text_snippet")) or "")


# ---------------------------------------------------------------------------
# 3D vector space (Plotly + numpy-PCA; dataframe fallback)
# ---------------------------------------------------------------------------


def compute_pca_3d(vectors: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Project vectors to 3 components via centered eigendecomposition.

    Returns an ``(n, 3)`` numpy array. Deterministic for identical input; the
    output is an isometry of the centered data, so pairwise distances are
    preserved (within the span of the top-3 components).
    """
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        width = arr.shape[1]
        arr = np.hstack([arr, np.zeros((arr.shape[0], 3 - width))])
    if arr.shape[0] == 1:
        arr = np.vstack([arr, arr + 1e-6])
    centered = arr - arr.mean(axis=0)
    cov = centered.T @ centered
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        vecs = np.eye(arr.shape[1])[:, :3]
        vals = np.ones(arr.shape[1])
    top = np.argsort(vals)[::-1][:3]
    comps = vecs[:, top]
    proj = centered @ comps
    if proj.shape[1] < 3:
        proj = np.hstack([proj, np.zeros((proj.shape[0], 3 - proj.shape[1]))])
    return proj


def build_vector_scatter_figure(
    query_embedding: Sequence[float],
    chunk_embeddings: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Sequence[float],
) -> go.Figure:
    """Plotly ``scatter3d`` of the query vector vs retrieved chunks with a
    collapse-to-nearest animation over ``_SEARCH_STEPS`` frames."""
    import plotly.graph_objects as go

    vectors = [list(query_embedding), *[list(emb) for emb in chunk_embeddings]]
    proj = compute_pca_3d(vectors)
    qpt = proj[0]
    cpts = proj[1:]

    max_score = max(scores) if scores else 1.0
    norm = [(s / max_score) if max_score > 0 else 0.0 for s in scores]

    def chunk_trace(points: np.ndarray, name: str) -> go.Scatter3d:
        pts = np.asarray(points)
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name=name,
            text=labels,
            marker={
                "size": 4 + 12 * np.asarray(norm),
                "color": np.asarray(norm),
                "colorscale": "Viridis",
                "opacity": 0.9,
                "reversescale": False,
            },
        )

    query_trace = go.Scatter3d(
        x=[qpt[0]],
        y=[qpt[1]],
        z=[qpt[2]],
        mode="markers+text",
        name="Query",
        text=["Query"],
        marker={"size": 14, "color": "#DC2626", "symbol": "diamond"},
        textposition="top center",
    )

    frames: list[go.Frame] = []
    for step in range(_SEARCH_STEPS):
        t = step / (_SEARCH_STEPS - 1)
        interp = np.asarray(cpts) + (np.asarray([qpt]) - np.asarray(cpts)) * (0.65 * t)
        frames.append(go.Frame(name=str(step), data=[chunk_trace(interp, "chunks")]))

    fig = go.Figure(data=[query_trace, chunk_trace(cpts, "chunks")], frames=frames)
    fig.update_layout(
        height=460,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": False}],
                        "label": "▶ Collapse",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        "label": "Initial",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 60},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [[str(i)], {"frame": {"duration": 90, "redraw": True}, "mode": "immediate"}],
                        "label": str(i),
                    }
                    for i in range(_SEARCH_STEPS)
                ],
                "currentvalue": {"prefix": "frame: "},
            }
        ],
    )
    fig.update_scenes(zaxis_title="PC3", yaxis_title="PC2", xaxis_title="PC1")
    return fig


def render_vector_scatter(
    query_embedding: Sequence[float],
    chunk_embeddings: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Sequence[float],
    *,
    key: str,
) -> None:
    """Render the 3D scatter, degrading to a static dataframe if Plotly fails."""
    import streamlit as st

    try:
        fig = build_vector_scatter_figure(query_embedding, chunk_embeddings, labels, scores)
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.dataframe(
            [{"Chunk": label, "Score": round(float(score), 3)} for label, score in zip(labels, scores, strict=False)],
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Typewriter + animated counters
# ---------------------------------------------------------------------------


def stream_answer_text(text: str, *, chunk_chars: int = 24, delay: float = 0.006) -> Generator[str, None, None]:
    """Yield the answer in small slices for ``st.write_stream`` (typewriter)."""
    for i in range(0, len(text), max(1, chunk_chars)):
        yield text[i : i + chunk_chars]
        if delay > 0:
            time.sleep(delay)


def build_animated_metric_html(label: str, value: float, *, digits: int = 0) -> str:
    """HTML count-up metric. Target value is present in the markup so a browser
    with JavaScript disabled still shows the final number."""
    try:
        target = float(value)
    except (TypeError, ValueError):
        target = 0.0
    display = f"{target:.{digits}f}"
    return (
        f"<style>.dec-metric{{display:inline-flex;flex-direction:column;padding:8px 4px;}}"
        f".dec-metric-label{{font-size:0.8rem;color:#64748B;}}"
        f".dec-metric-value{{font-size:1.4rem;font-weight:600;color:#0F172A;}}</style>"
        f'<span class="dec-metric">'
        f'<span class="dec-metric-label">{label}</span>'
        f'<span class="dec-metric-value" data-target="{target}" data-digits="{int(digits)}">{display}</span>'
        f"</span>"
        f"<script>(function(){{"
        f"var el=document.querySelector('.dec-metric-value');var target=parseFloat(el.dataset.target);"
        f"var digits=parseInt(el.dataset.digits,10)||0;"
        f"var start=null,dur=750;function tick(ts){{if(!start)start=ts;"
        f"var p=Math.min((ts-start)/dur,1);var v=target*p;"
        f"el.textContent=v.toFixed(digits);if(p<1)requestAnimationFrame(tick);"
        f"else el.textContent=el.dataset.target;}}"
        f"el.textContent='0';requestAnimationFrame(tick);"
        f"}})();</script>"
    )


# ---------------------------------------------------------------------------
# Public renderers (thin wrappers over the pure builders)
# ---------------------------------------------------------------------------


def render_pipeline_diagram(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    states: Mapping[str, NodeState],
    *,
    with_view_switch: bool = False,
    key: str = "diagram",
) -> None:
    """Render the animated diagram, optionally with a CSS/Mermaid toggle."""
    import streamlit as st

    view = (
        st.pills(
            "Diagram",
            ["CSS", "Mermaid"],
            default="CSS",
            selection_mode="single",
            key=f"{key}_view",
            label_visibility="collapsed",
        )
        if with_view_switch
        else "CSS"
    )

    if view == "Mermaid":
        st.html(build_mermaid_html(nodes, edges, states), unsafe_allow_javascript=True)
    else:
        st.html(build_diagram_html(nodes, edges, states), unsafe_allow_javascript=False)


def render_lottie_badge(kind: str, *, height: int = 96) -> None:
    """Render a Lottie badge with a guarded static CSS/SVG fallback."""
    import streamlit as st

    st.html(build_lottie_player_html(kind, height=height), unsafe_allow_javascript=True)


def render_animated_metric(label: str, value: float, *, digits: int = 0) -> None:
    """Render a count-up metric with a no-JS static fallback."""
    import streamlit as st

    st.html(build_animated_metric_html(label, value, digits=digits), unsafe_allow_javascript=True)
