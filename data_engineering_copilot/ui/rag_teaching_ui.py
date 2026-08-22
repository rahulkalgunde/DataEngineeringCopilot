"""Interactive teaching UI that animates the RAG pipeline layer by layer.

A standalone Streamlit app (``dec_venv/bin/streamlit run
data_engineering_copilot/ui/rag_teaching_ui.py``) for showing students how a
document and a question travel through the production pipeline. Every stage
executes real codebase components wired exactly like ``factory.py`` builds
them; only the pacing delays between reveal steps are simulated.

Posture: read-only by default (the ingestion run never writes to Qdrant) and
fail-open at the UI layer — missing infrastructure or a failed stage degrades
to an explanatory warning instead of raising into the app.
"""

from __future__ import annotations

import asyncio
import html as html_mod
import math
import sys
import threading
import time
import urllib.request
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering_copilot.config.settings import settings  # noqa: E402
from data_engineering_copilot.domain.models import (  # noqa: E402
    Answer,
    DocumentChunk,
    ParsedDocument,
    RawDocument,
)
from data_engineering_copilot.factory import (  # noqa: E402
    build_chunker,
    build_embedding_fallback_chain,
    build_llm_fallback_chain,
    build_rag_service,
)
from data_engineering_copilot.infrastructure.async_qdrant_store import chunk_to_payload  # noqa: E402
from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer  # noqa: E402
from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder  # noqa: E402
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser  # noqa: E402
from data_engineering_copilot.infrastructure.rst_parser import RstParser  # noqa: E402
from data_engineering_copilot.services.api_extractor import ApiDocExtractor  # noqa: E402
from data_engineering_copilot.services.code_block_parser import CodeBlockParser  # noqa: E402
from data_engineering_copilot.services.contextual_chunk_enricher import (  # noqa: E402
    ContextualChunkEnricher,
    LLMContextSummarizer,
)
from data_engineering_copilot.services.prompt_builder import PromptBuilder  # noqa: E402
from data_engineering_copilot.services.text_filter import ChunkFilter  # noqa: E402

STEP_DELAY_SECONDS = 1.5
INGEST_TIMEOUT_SECONDS = 600
ANSWER_TIMEOUT_SECONDS = 300
HEATMAP_CELLS = 64
HEATMAP_COLUMNS = 16
EMBED_DIM_PREVIEW = 10
SNIPPET_CHARS = 220
CONTEXT_PREVIEW_CHARS = 1600
_RST_SUFFIXES = (".rst", ".rst.txt")

INGESTION_STEPS: tuple[tuple[str, str], ...] = (
    ("Parse", "html → markdown"),
    ("Chunk", "semantic split"),
    ("Enrich", "metadata + context"),
    ("Embed", "text → vector"),
    ("Store", "Qdrant point"),
)
QA_STEPS: tuple[tuple[str, str], ...] = (
    ("Guard & Decompose", "sanitize + sub-queries"),
    ("Retrieve", "dense + BM25 + RRF"),
    ("Rerank", "re-order by relevance"),
    ("Generate", "context → answer"),
)
_INGESTION_ICONS: tuple[str, ...] = ("parse", "chunk", "enrich", "embed", "store")
_QA_ICONS: tuple[str, ...] = ("guard", "retrieve", "rerank", "generate")

_ICONS: dict[str, str] = {
    "parse": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h4"/>',
    "chunk": '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/>',
    "enrich": '<path d="M12 3l1.8 4.9L19 9.7l-5.2 1.8L12 16.4l-1.8-4.9L5 9.7l5.2-1.8z"/><path d="M18.5 15.5l.9 2.3 2.3.9-2.3.9-.9 2.3-.9-2.3-2.3-.9 2.3-.9z"/>',
    "embed": '<path d="M5 19L17 7"/><path d="M11 6h7v7"/><circle cx="5" cy="19" r="1.6"/>',
    "store": '<ellipse cx="12" cy="6" rx="7" ry="2.6"/><path d="M5 6v12c0 1.45 3.1 2.6 7 2.6s7-1.15 7-2.6V6"/><path d="M5 12c0 1.45 3.1 2.6 7 2.6s7-1.15 7-2.6"/>',
    "guard": '<path d="M12 3l7 2.8v5.4c0 4.3-2.9 7.9-7 9.3-4.1-1.4-7-5-7-9.3V5.8z"/><path d="m9.2 11.8 2 2 3.6-4"/>',
    "retrieve": '<circle cx="11" cy="11" r="6.5"/><path d="m20.5 20.5-4.2-4.2"/>',
    "rerank": '<path d="M8 5v14"/><path d="m5 16 3 3 3-3"/><path d="M16 19V5"/><path d="m13 8 3-3 3 3"/>',
    "generate": '<path d="M21 12a8 8 0 0 1-8 8H4l2.3-2.9A8 8 0 1 1 21 12z"/><path d="M9 12h.01M13 12h.01M17 12h.01"/>',
}

TEACH_CSS = """
.tw{color-scheme:normal}
.tpanel{background:#0B1220;border:1px solid #1E293B;border-radius:14px;padding:14px 16px;color:#E2E8F0;font-size:13px}
.tpanel .ptitle{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#7DD3FC;margin-bottom:10px;display:flex;align-items:center;gap:7px}
.tflow{display:flex;align-items:stretch;width:100%;margin:4px 0 2px}
.tnode{flex:1;display:flex;flex-direction:column;align-items:center;gap:6px;padding:13px 6px 11px;border-radius:13px;border:1.5px solid #1E293B;background:#0F172A;color:#5A6B85;text-align:center;min-width:0;transition:all .45s ease}
.tnode svg{width:21px;height:21px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
.tnode .ico{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#141F35;color:#4C5C77;transition:all .45s ease}
.tnode .lbl{font-size:12.5px;font-weight:700;line-height:1.2}
.tnode .sub{font-size:9.5px;line-height:1.25;color:#3E4E68}
.tnode.done{border-color:#14532D;background:#07160D;color:#4ADE80}
.tnode.done .ico{background:#0A2413;color:#34D399}
.tnode.active{border-color:#1D4ED8;background:#0A1730;color:#93C5FD;animation:twPulse 1.7s ease-in-out infinite}
.tnode.active .ico{background:#12275A;color:#60A5FA}
.tnode.failed{border-color:#7F1D1D;background:#1A0A0A;color:#FCA5A5}
.tnode.failed .ico{background:#2B1010;color:#F87171}
.tlink{flex:0 0 30px;align-self:center;height:2px;background:#1E293B;position:relative;border-radius:2px}
.tlink.on{background:linear-gradient(90deg,#075985,#38BDF8)}
.tw-pkt{position:absolute;top:-3px;width:8px;height:8px;border-radius:50%;background:#38BDF8;box-shadow:0 0 9px rgba(56,189,248,.9);animation:twPacket 1.15s linear infinite}
.tw-pkt:nth-child(2){animation-delay:.55s}
@keyframes twPacket{0%{left:-4px;opacity:0}15%{opacity:1}85%{opacity:1}100%{left:calc(100% - 4px);opacity:0}}
@keyframes twPulse{0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,.28)}55%{box-shadow:0 0 0 10px rgba(59,130,246,0)}}
@keyframes twIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes twGrow{from{transform:scaleX(0)}}
.tgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:9px}
.tcard{background:#0F172A;border:1px solid #1E293B;border-radius:11px;padding:10px 12px;color:#CBD5E1;animation:twIn .5s both cubic-bezier(.22,1,.36,1)}
.tcard:hover{border-color:#334155}
.tcard .row1{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.tcard .rank{font-size:11px;font-weight:800;color:#7DD3FC}
.tcard .ttl{font-size:12px;font-weight:600;color:#E2E8F0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.tbadge{font-size:9px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;padding:2px 7px;border-radius:99px}
.b-text{background:#0C2B3D;color:#67E8F9}.b-code{background:#2B1055;color:#C4B5FD}.b-api{background:#3D2B0A;color:#FCD34D}.b-table{background:#0A332B;color:#5EEAD4}.b-mixed{background:#12314A;color:#93C5FD}
.tmeta{font-size:10.5px;color:#64748B;margin-top:6px;display:flex;gap:10px;flex-wrap:wrap}
.tsnip{font-size:11px;color:#94A3B8;line-height:1.45;margin-top:6px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.tmets{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:9px;margin:8px 0}
.tmet{background:#0F172A;border:1px solid #1E293B;border-radius:11px;padding:9px 12px;animation:twIn .45s both}
.tmet .k{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:#64748B;font-weight:700}
.tmet .v{font-size:17px;font-weight:700;color:#E2E8F0;margin-top:3px;font-variant-numeric:tabular-nums}
.sbar{height:7px;border-radius:99px;background:#16223A;overflow:hidden;margin-top:6px}
.sbar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#38BDF8,#818CF8);transform-origin:left;animation:twGrow .9s cubic-bezier(.22,1,.36,1) both}
.tchip{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:999px;border:1px solid #155E75;background:#082F3C;color:#67E8F9;font-size:12px;font-weight:600;animation:twIn .45s both;margin:0 6px 6px 0}
.tchip.c-intent{border-color:#4C1D95;background:#221046;color:#C4B5FD}
.tchip.c-hyde{border-color:#78350F;background:#2B1606;color:#FCD34D}
.tchip.c-mode{border-color:#134E4A;background:#062A26;color:#5EEAD4}
.heat{display:grid;gap:3px;margin:8px 0}
.hcell{aspect-ratio:1;border-radius:3.5px;animation:twIn .4s both}
.difftag{font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin:8px 0 4px;color:#94A3B8}
.dadd{background:rgba(52,211,153,.13);border-left:3px solid #34D399;color:#A7F3D0;padding:7px 10px;border-radius:0 8px 8px 0;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.dbase{color:#CBD5E1;padding:7px 10px;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:900px){.cmp{grid-template-columns:1fr}}
.cmphd{font-size:11px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#7DD3FC;margin-bottom:8px}
.rrow{display:flex;align-items:center;gap:9px;background:#0F172A;border:1px solid #1E293B;border-radius:10px;padding:7px 11px;margin-bottom:6px;animation:twIn .45s both}
.rrow .pos{width:26px;font-weight:800;color:#7DD3FC;font-size:12px}
.rrow .body{flex:1;min-width:0}
.rrow .t{font-size:12px;font-weight:600;color:#E2E8F0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rrow .m{font-size:10px;color:#64748B;margin-top:2px}
.rrow .sc{font-size:12px;font-weight:700;color:#A5B4FC;font-variant-numeric:tabular-nums;width:52px;text-align:right}
.delta{font-size:11px;font-weight:800;width:46px;text-align:right}
.up{color:#4ADE80}.down{color:#F87171}.same{color:#475569}
"""

SAMPLE_HTML = """<!doctype html>
<html><head><title>SparkSession — PySpark API Reference</title></head>
<body>
<h1>SparkSession</h1>
<p>The SparkSession is the unified entry point for DataFrame and SQL functionality in PySpark.
It coordinates job execution across the cluster, exposes the Catalyst optimizer, and manages
the shared state of every running Spark application.</p>
<h2>Methods</h2>
<h3>spark.sql</h3>
<p>Returns a DataFrame representing the result of the given query string. The query is parsed,
analyzed, and optimized by Catalyst before execution on the cluster.</p>
<pre><code>df = spark.sql("SELECT * FROM events WHERE ts &gt; '2024-01-01'")</code></pre>
<h3>spark.read</h3>
<p>Returns a DataFrameReader that loads data from Parquet, JSON, Delta Lake tables, CSV files,
and JDBC endpoints into a lazily evaluated DataFrame.</p>
<pre><code>df = spark.read.format("parquet").load("/mnt/warehouse/events")</code></pre>
<h3>spark.conf.set</h3>
<p>Sets a runtime configuration property on the session, such as shuffle partitions or
executor memory settings.</p>
<pre><code>spark.conf.set("spark.sql.shuffle.partitions", "200")</code></pre>
</body></html>
"""

SAMPLE_QUESTION = "How does spark.sql execute a SQL query?"


def _esc(value: Any) -> str:
    return html_mod.escape(str(value), quote=True)


def _truncate(text: str, limit: int) -> str:
    clean = text.strip()
    return clean if len(clean) <= limit else clean[:limit].rstrip() + " …"


def _icon(name: str) -> str:
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{_ICONS[name]}</svg>'


# ---------------------------------------------------------------------------
# Custom HTML builders (dark console aesthetic, pure CSS animation)
# ---------------------------------------------------------------------------


def flow_diagram_html(steps: tuple[tuple[str, str], ...], icons: tuple[str, ...], states: list[str]) -> str:
    parts = ['<div class="tflow">']
    for i, ((label, sub), icon_key, state) in enumerate(zip(steps, icons, states, strict=True)):
        parts.append(
            f'<div class="tnode {state}"><div class="ico">{_icon(icon_key)}</div>'
            f'<div><div class="lbl">{_esc(label)}</div><div class="sub">{_esc(sub)}</div></div></div>'
        )
        if i < len(steps) - 1:
            on = " on" if states[i] == "done" else ""
            pkts = '<span class="tw-pkt"></span><span class="tw-pkt"></span>' if on else ""
            parts.append(f'<div class="tlink{on}">{pkts}</div>')
    parts.append("</div>")
    return "".join(parts)


def metric_cards_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="tmet" style="animation-delay:{i * 70}ms"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for i, (k, v) in enumerate(items)
    )
    return f'<div class="tmets">{cells}</div>'


def chunk_cards_html(chunks: list[DocumentChunk], limit: int = 9) -> str:
    known_types = {"text", "code", "api", "table", "mixed"}
    cards = []
    for i, chunk in enumerate(chunks[:limit]):
        ctype = chunk.chunk_type if chunk.chunk_type in known_types else "text"
        heading = chunk.section_header or (chunk.heading_path[-1] if chunk.heading_path else "(no heading)")
        cards.append(
            f'<div class="tcard" style="animation-delay:{i * 90}ms">'
            f'<div class="row1"><span class="rank">#{chunk.chunk_index}</span>'
            f'<span class="tbadge b-{ctype}">{ctype}</span><span class="ttl">{_esc(heading)}</span></div>'
            f'<div class="tsnip">{_esc(_truncate(chunk.text, SNIPPET_CHARS))}</div>'
            f'<div class="tmeta"><span>{chunk.word_count} words</span><span>{chunk.token_count} tok</span>'
            f"<span>hash {_esc(chunk.content_hash[:8])}</span></div></div>"
        )
    more = (
        f'<div class="tmeta" style="margin-top:8px">+ {len(chunks) - limit} more chunks…</div>'
        if len(chunks) > limit
        else ""
    )
    return f'<div class="tgrid">{"".join(cards)}</div>{more}'


def heatmap_html(values: list[float], columns: int = HEATMAP_COLUMNS) -> str:
    max_abs = max((abs(v) for v in values), default=1.0) or 1.0
    cells = []
    for i, v in enumerate(values):
        alpha = 0.10 + 0.82 * min(abs(v) / max_abs, 1.0)
        color = f"rgba(56,189,248,{alpha:.3f})" if v >= 0 else f"rgba(251,146,60,{alpha:.3f})"
        cells.append(
            f'<span class="hcell" style="background:{color};animation-delay:{i * 12}ms" title="{v:+.4f}"></span>'
        )
    return f'<div class="heat" style="grid-template-columns:repeat({columns},1fr)">{"".join(cells)}</div>'


def chips_row_html(items: list[str], kind: str = "") -> str:
    cls = f" tchip {kind}" if kind else " tchip"
    chips = "".join(
        f'<span class="{cls}" style="animation-delay:{i * 70}ms">{_esc(item)}</span>' for i, item in enumerate(items)
    )
    return f'<div style="margin:6px 0">{chips}</div>' if chips else ""


def candidate_rows_html(candidates: list[dict[str, Any]], lookup: dict[str, dict[str, Any]], limit: int = 8) -> str:
    rows = []
    for i, cand in enumerate(candidates[:limit]):
        conf = float(cand.get("confidence") or 0.0)
        cid = str(cand.get("chunk_id") or "")
        title = lookup.get(cid, {}).get("title") or cand.get("title") or cid[:16]
        source = cand.get("source_name") or ""
        distance = float(cand.get("distance") or 0.0)
        rows.append(
            f'<div class="rrow" style="animation-delay:{i * 70}ms">'
            f'<span class="pos">#{int(cand.get("rank") or i) + 1}</span>'
            f'<div class="body"><div class="t">{_esc(title)}</div>'
            f'<div class="m">{_esc(source)} · distance {distance:.3f}</div>'
            f'<div class="sbar"><i style="width:{conf * 100:.0f}%"></i></div></div>'
            f'<span class="sc">{conf:.3f}</span></div>'
        )
    return "".join(rows)


def rerank_compare_html(
    fused: list[dict[str, Any]],
    final_refs: list[dict[str, Any]],
    dropped_records: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
) -> str:
    fused_rank = {str(ref.get("chunk_id")): int(ref.get("rank", 0)) for ref in fused}

    def row(ref: dict[str, Any], delta: int | None) -> str:
        rank = int(ref.get("rank", 0))
        cid = str(ref.get("chunk_id") or "")
        conf = float(ref.get("confidence") or 0.0)
        title = lookup.get(cid, {}).get("title") or cid[:16]
        arrow = '<span class="delta same">—</span>'
        if delta is not None and delta > 0:
            arrow = f'<span class="delta up">▲ {delta}</span>'
        elif delta is not None and delta < 0:
            arrow = f'<span class="delta down">▼ {abs(delta)}</span>'
        return (
            f'<div class="rrow"><span class="pos">#{rank + 1}</span>'
            f'<div class="body"><div class="t">{_esc(title)}</div>'
            f'<div class="sbar"><i style="width:{conf * 100:.0f}%"></i></div></div>'
            f"{arrow}<span class='sc'>{conf:.3f}</span></div>"
        )

    before_rows = "".join(row(ref, None) for ref in fused[:8])
    after_rows = ""
    for ref in final_refs[:8]:
        previous = fused_rank.get(str(ref.get("chunk_id")))
        delta = None if previous is None else previous - int(ref.get("rank", 0))
        after_rows += row(ref, delta)
    dropped_names = [
        f"{lookup.get(str(rec.get('chunk_id')), {}).get('title') or rec.get('chunk_id', '')[:12]} ({rec.get('reason', '')})"
        for rec in dropped_records[:6]
    ]
    dropped_html = (
        f'<div class="difftag">dropped by budget / per-source cap</div>{chips_row_html(dropped_names)}'
        if dropped_names
        else ""
    )
    return (
        f'<div class="cmp"><div><div class="cmphd">Before · fusion ranking</div>{before_rows}</div>'
        f'<div><div class="cmphd">After · reranked + assembled</div>{after_rows}</div></div>{dropped_html}'
    )


def before_after_html(before_text: str, after_text: str) -> str:
    if after_text.endswith(before_text) and after_text != before_text:
        added = after_text[: len(after_text) - len(before_text)]
        return (
            f'<div class="difftag">added by enrichment</div><div class="dadd">{_esc(added)}</div>'
            f'<div class="dbase">{_esc(_truncate(before_text, 400))}</div>'
        )
    return (
        '<div class="cmp">'
        f'<div><div class="cmphd">Before</div><div class="dbase">{_esc(_truncate(before_text, 320))}</div></div>'
        f'<div><div class="cmphd">After</div><div class="dbase">{_esc(_truncate(after_text, 320))}</div></div></div>'
    )


# ---------------------------------------------------------------------------
# Backend wiring (factory-mirrored components, shared event loop)
# ---------------------------------------------------------------------------


@st.cache_resource
def _service_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_run, name="teaching-rag-loop", daemon=True).start()
    return loop


@st.cache_resource
def _build_ingestion_components() -> dict[str, Any]:
    rst_parser = RstParser()
    html_parser = MarkdownParser()
    try:
        enrichment_chain = build_llm_fallback_chain(
            purpose="enrichment",
            purpose_provider=settings.enrichment_llm_provider or "ollama",
            purpose_model=settings.enrichment_llm_model,
        )
        enricher: ContextualChunkEnricher | None = ContextualChunkEnricher(
            summarizer=LLMContextSummarizer(llm_client=enrichment_chain),
            enabled=settings.contextual_enrichment_enabled,
            batch_size=settings.enrichment_batch_size,
        )
    except Exception:
        enricher = None
    try:
        embedder: FallbackEmbedder | None = FallbackEmbedder(build_embedding_fallback_chain())
    except Exception:
        embedder = None
    return {
        "rst_parser": rst_parser,
        "html_parser": html_parser,
        "chunker": build_chunker(settings),
        "code_parser": CodeBlockParser(enabled=getattr(settings, "code_block_parsing_enabled", True)),
        "chunk_filter": ChunkFilter(enabled=getattr(settings, "chunk_filtering_enabled", True)),
        "api_extractor": ApiDocExtractor(enabled=getattr(settings, "api_extraction_enabled", True)),
        "enricher": enricher,
        "embedder": embedder,
    }


@st.cache_resource
def _build_rag_service() -> Any:
    return build_rag_service()


def rag_service() -> Any | None:
    try:
        return _build_rag_service()
    except Exception:
        return None


async def _execute_ingestion(raw: RawDocument) -> dict[str, Any]:
    comps = _build_ingestion_components()
    trace: dict[str, Any] = {"errors": {}, "raw": raw}

    parsed: ParsedDocument | None
    if raw.content_type != "text/html" or raw.url.lower().endswith(_RST_SUFFIXES):
        try:
            parsed = comps["rst_parser"].parse(raw)
            parsed = parsed if parsed is not None else comps["html_parser"].parse(raw)
        except ValueError:
            parsed = comps["html_parser"].parse(raw)
    else:
        parsed = comps["html_parser"].parse(raw)
    if parsed is None or not parsed.text:
        trace["errors"]["parse"] = "Parser dropped the document (<40 words after conversion)."
        return trace
    trace["parsed"] = parsed

    try:
        chunks = await comps["chunker"].chunk(parsed)
    except Exception as exc:
        trace["errors"]["chunk"] = f"{type(exc).__name__}: {exc}"
        return trace
    chunks = comps["code_parser"].extract(list(chunks))
    kept = comps["chunk_filter"].extract(list(chunks))
    kept_ids = {c.chunk_id for c in kept}
    trace["chunks_raw"] = list(chunks)
    trace["chunks_kept"] = list(kept)
    trace["chunks_dropped"] = [c for c in chunks if c.chunk_id not in kept_ids]
    if not kept:
        trace["errors"]["chunk"] = "All chunks were removed by quality filtering."
        return trace

    pre_enrich = list(kept)
    enriched = comps["api_extractor"].extract(list(pre_enrich))
    if comps["enricher"] is not None:
        enriched = await comps["enricher"].enrich(parsed, list(enriched))
    trace["pre_enrich"] = pre_enrich
    trace["enriched"] = list(enriched)

    if comps["embedder"] is not None and enriched:
        try:
            trace["vectors"] = await comps["embedder"].embed_texts([c.text for c in enriched])
        except Exception as exc:
            trace["errors"]["embed"] = f"{type(exc).__name__}: {exc}"
    return trace


def execute_ingestion(raw: RawDocument) -> tuple[dict[str, Any] | None, str | None]:
    box: dict[str, Any] = {}

    def _work() -> None:
        future = asyncio.run_coroutine_threadsafe(_execute_ingestion(raw), _service_loop())
        try:
            box["trace"] = future.result(timeout=INGEST_TIMEOUT_SECONDS)
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    worker.join(INGEST_TIMEOUT_SECONDS + 10)
    if "error" in box:
        return None, str(box["error"])
    return box.get("trace"), None


def execute_qa(service: Any, question: str) -> dict[str, Any]:
    completed: list[str] = []
    details: dict[str, list[dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    box: dict[str, Any] = {}

    def _on_detail(kind: str, payload: dict[str, Any]) -> None:
        details.setdefault(kind, []).append(payload)

    def _work() -> None:
        future = asyncio.run_coroutine_threadsafe(
            service.answer(
                question,
                on_step=completed.append,
                on_step_detail=_on_detail,
                provenance=provenance,
                bypass_cache=True,
                user_id=st.session_state.get("teach_user_id"),
                session_id=st.session_state.get("teach_session_id"),
            ),
            _service_loop(),
        )
        try:
            box["answer"] = future.result(timeout=ANSWER_TIMEOUT_SECONDS)
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    worker.join(ANSWER_TIMEOUT_SECONDS + 10)
    return {
        "answer": box.get("answer"),
        "error": box.get("error"),
        "steps": completed,
        "details": details,
        "provenance": provenance,
    }


def _probe(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def qdrant_up() -> bool:
    return _probe(f"{settings.qdrant_url}/collections")


def ollama_up() -> bool:
    return _probe(f"{settings.ollama_base_url.rstrip('/')}/api/tags")


def _typewriter(text: str, chunk_chars: int = 28, delay: float = 0.008) -> Generator[str, None, None]:
    for i in range(0, len(text), chunk_chars):
        yield text[i : i + chunk_chars]
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Ingestion tab
# ---------------------------------------------------------------------------


def _set_flow(
    ph: Any,
    steps: tuple[tuple[str, str], ...],
    icons: tuple[str, ...],
    done: int,
    failed: int | None = None,
) -> None:
    states = ["idle"] * len(steps)
    for i in range(done):
        states[i] = "done"
    if failed is not None:
        states[failed] = "failed"
    elif done < len(steps):
        states[done] = "active"
    ph.html(flow_diagram_html(steps, icons, states))


def _render_parse_step(container: Any, trace: dict[str, Any]) -> bool:
    with container.container(border=True):
        st.markdown("**Step 1 — Parsing** · `MarkdownParser.parse(RawDocument) → ParsedDocument`")
        error = trace["errors"].get("parse")
        if error:
            st.error(error)
            return False
        parsed: ParsedDocument | None = trace.get("parsed")
        if parsed is None:
            st.warning("The parser dropped this document (fewer than 40 words after conversion).")
            return False
        st.html(
            metric_cards_html(
                [("Title", parsed.title), ("Words", str(len(parsed.text.split()))), ("Chars", str(len(parsed.text)))]
            )
        )
        left, right = st.columns(2)
        with left:
            st.caption("Input · raw HTML")
            st.code(_truncate(trace["raw"].html, 700), language="html")
        with right:
            st.caption("Output · clean markdown")
            st.code(_truncate(parsed.text, 700), language="markdown")
    return True


def _render_chunk_step(container: Any, trace: dict[str, Any]) -> bool:
    with container.container(border=True):
        st.markdown("**Step 2 — Chunking** · `chunker.chunk(ParsedDocument) → list[DocumentChunk]`")
        error = trace["errors"].get("chunk")
        if error:
            st.error(error)
            return False
        chunks: list[DocumentChunk] = trace.get("chunks_raw", [])
        dropped: list[DocumentChunk] = trace.get("chunks_dropped", [])
        st.html(metric_cards_html([("Chunks produced", str(len(chunks))), ("Dropped by filter", str(len(dropped)))]))
        if chunks:
            st.html(chunk_cards_html(chunks))
        if dropped:
            with st.expander(f"Quality filter removed {len(dropped)} chunk(s)", expanded=False):
                st.dataframe(
                    [
                        {
                            "index": c.chunk_index,
                            "words": c.word_count,
                            "type": c.chunk_type,
                            "text": _truncate(c.text, 90),
                        }
                        for c in dropped
                    ],
                    width="stretch",
                    hide_index=True,
                )
    return True


def _render_enrich_step(container: Any, trace: dict[str, Any]) -> bool:
    with container.container(border=True):
        st.markdown("**Step 3 — Enrichment** · `ApiDocExtractor.extract` + `ContextualChunkEnricher.enrich`")
        pre: list[DocumentChunk] = trace.get("pre_enrich", [])
        post: list[DocumentChunk] = trace.get("enriched", [])
        post_by_id = {c.chunk_id: c for c in post}
        changed = sum(1 for c in pre if post_by_id.get(c.chunk_id) and post_by_id[c.chunk_id].text != c.text)
        st.html(
            metric_cards_html(
                [("Chunks in", str(len(pre))), ("Chunks out", str(len(post))), ("Modified", str(changed))]
            )
        )
        shown = 0
        for original in pre:
            updated = post_by_id.get(original.chunk_id)
            if updated is None or updated.text == original.text or shown >= 3:
                continue
            st.caption(f"Chunk #{original.chunk_index}")
            st.html(before_after_html(original.text, updated.text))
            shown += 1
        if shown == 0:
            st.info("No textual changes — the contextual enricher is disabled, offline, or found nothing to add.")
    return True


def _render_embed_step(container: Any, trace: dict[str, Any]) -> bool:
    with container.container(border=True):
        st.markdown("**Step 4 — Embedding** · `FallbackEmbedder.embed_texts(list[str]) → list[list[float]]`")
        vectors: list[list[float]] = trace.get("vectors", [])
        if not vectors:
            error = trace["errors"].get("embed")
            if error:
                st.warning(f"Embedding skipped — {error}. Start Ollama (`make up`) and re-run to see this stage live.")
            else:
                st.info("Embedding unavailable in this environment — later stages still work offline.")
            return False
        first = vectors[0]
        l2 = math.sqrt(sum(v * v for v in first))
        st.html(
            metric_cards_html(
                [
                    ("Vectors", str(len(vectors))),
                    ("Dimensions", str(len(first))),
                    ("L2 norm ‖v‖₂", f"{l2:.4f}"),
                ]
            )
        )
        st.caption(
            f"Each chunk's text becomes one dense vector. First vector of chunk #0, values {HEATMAP_CELLS}/{len(first)}:"
        )
        st.html(heatmap_html(first[:HEATMAP_CELLS]))
        rows = [
            {"chunk": i, **{f"d{j}": round(vec[j], 4) for j in range(min(EMBED_DIM_PREVIEW, len(vec)))}}
            for i, vec in enumerate(vectors[:6])
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption("Blue cells are positive components, orange negative — hover any cell for its exact value.")
    return True


def _render_store_step(container: Any, trace: dict[str, Any]) -> bool:
    with container.container(border=True):
        st.markdown("**Step 5 — Vector storage** · `chunk_to_payload(chunk)` + dense/sparse point preview")
        enriched: list[DocumentChunk] = trace.get("enriched", [])
        vectors: list[list[float]] = trace.get("vectors", [])
        if not enriched:
            st.warning("Nothing reached the storage stage.")
            return False
        chunk = enriched[0]
        vector = vectors[0] if vectors else []
        point: dict[str, Any] = {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))}
        if vector:
            point["vector"] = {"dense": {"dims": len(vector), "head": [round(v, 6) for v in vector[:8]]}}
            try:
                sparse = BM25Tokenizer().tokenize_query(chunk.text)
                point["vector"]["sparse"] = {
                    "indices_head": [int(i) for i in list(sparse.indices)[:12]],
                    "values_head": [round(float(v), 4) for v in list(sparse.values)[:12]],
                    "nnz": len(sparse.indices),
                }
            except Exception:
                pass
        point["payload"] = chunk_to_payload(chunk)
        st.json(point)
        st.caption(
            "Dry-run only — nothing was written. In production `AsyncQdrantVectorStore.upsert_chunks` writes "
            f"points like this to collection `{settings.collection_name}`."
        )
    return True


def _render_ingestion_static(trace: dict[str, Any]) -> None:
    with st.expander("Last ingestion run (static view)", expanded=False):
        parsed: ParsedDocument | None = trace.get("parsed")
        if parsed is not None:
            st.caption(f"Parsed “{parsed.title}” · {len(parsed.text.split())} words")
        st.caption(
            f"{len(trace.get('chunks_kept', []))} chunks survived filtering · {len(trace.get('vectors', []))} embedded"
        )
        for name, message in trace.get("errors", {}).items():
            st.warning(f"{name}: {message}")


def _load_sample() -> None:
    st.session_state["teach_html"] = SAMPLE_HTML


def ingestion_tab() -> None:
    st.subheader("Ingestion flow")
    st.caption("Paste HTML and watch it become a Qdrant point. Every stage runs the real codebase code; dry-run only.")

    with st.container(border=True):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            source_name = st.text_input("Source name", value="teaching-sample", key="teach_src_name")
        with col_b:
            url = st.text_input("URL label", value="sample://spark-session", key="teach_src_url")
        html_text = st.text_area("Raw HTML", height=170, key="teach_html", placeholder="<!doctype html> …")
        btn_load, btn_run, _ = st.columns([1, 1.4, 2])
        btn_load.button("Load sample page", icon=":material/description:", key="teach_sample", on_click=_load_sample)
        busy = bool(st.session_state.get("teach_busy", False))
        run_clicked = btn_run.button(
            "Run ingestion animation",
            type="primary",
            icon=":material/play_arrow:",
            disabled=busy,
            key="teach_run_ingest",
        )

    flow_ph = st.empty()
    progress_ph = st.empty()
    step_slots = [st.empty() for _ in INGESTION_STEPS]

    if run_clicked:
        if not html_text.strip():
            st.error("Paste some HTML first, or load the sample page.")
            return
        st.session_state["teach_busy"] = True
        try:
            progress_ph.progress(0.05, text="Executing real pipeline stages…")
            raw = RawDocument(source_name=source_name or "teaching-sample", url=url or "pasted://html", html=html_text)
            with st.spinner("Running parse → chunk → enrich → embed…"):
                trace, error = execute_ingestion(raw)
            if error or trace is None:
                st.error(f"Ingestion failed: {error or 'no trace returned'}")
                return
            renderers = (
                _render_parse_step,
                _render_chunk_step,
                _render_enrich_step,
                _render_embed_step,
                _render_store_step,
            )
            failure_node: int | None = None
            for idx, renderer in enumerate(renderers):
                _set_flow(flow_ph, INGESTION_STEPS, _INGESTION_ICONS, done=idx, failed=failure_node)
                ok = renderer(step_slots[idx], trace)
                if not ok and failure_node is None:
                    failure_node = idx
                progress_ph.progress((idx + 1) / len(renderers), text=f"Step {idx + 1}/{len(renderers)} revealed")
                time.sleep(STEP_DELAY_SECONDS)
            _set_flow(flow_ph, INGESTION_STEPS, _INGESTION_ICONS, done=len(renderers))
            progress_ph.progress(1.0, text="Ingestion walkthrough complete")
            st.session_state["teach_ingest"] = trace
        finally:
            st.session_state["teach_busy"] = False
    elif st.session_state.get("teach_ingest") is not None:
        _set_flow(flow_ph, INGESTION_STEPS, _INGESTION_ICONS, done=len(INGESTION_STEPS))
        progress_ph.progress(1.0, text="Last run result below — click the button to replay the animation")
        _render_ingestion_static(st.session_state["teach_ingest"])


# ---------------------------------------------------------------------------
# Question/answer tab
# ---------------------------------------------------------------------------


def _detail(details: dict[str, list[dict[str, Any]]], kind: str) -> dict[str, Any]:
    items = details.get(kind) or []
    return items[-1] if items else {}


def _render_guard_step(container: Any, question: str, details: dict[str, list[dict[str, Any]]]) -> bool:
    with container.container(border=True):
        st.markdown(
            "**Step 1 — Guardrails & decomposition** · `PromptBuilder.sanitize_query` + `QueryRewriter.rewrite`"
        )
        sanitized = PromptBuilder.sanitize_query(question)
        if sanitized != question:
            st.caption("Input guardrail rewrote the raw question:")
            st.code(question, language="text")
            st.code(sanitized, language="text")
        rewrite = _detail(details, "rewrite")
        if not rewrite:
            st.info("Rewrite details unavailable (rewriter offline or cache bypass skipped it).")
            return False
        intent = str(rewrite.get("intent") or "unknown")
        st.html(chips_row_html([f"intent: {intent}"], kind="c-intent"))
        left, right = st.columns(2)
        with left:
            st.caption("Original question")
            st.code(str(rewrite.get("original_query") or question), language="text")
        with right:
            st.caption("Effective (rewritten) query")
            st.code(str(rewrite.get("rewritten_query") or ""), language="text")
        decomposed = [str(s) for s in (rewrite.get("decomposed_steps") or [])]
        if decomposed:
            st.caption(f"Decomposed into {len(decomposed)} sub-query steps:")
            st.html(chips_row_html(decomposed))
        expansions = [str(q) for q in (rewrite.get("expansions") or [])]
        if len(expansions) > 1:
            st.caption(f"{len(expansions)} query variants will be retrieved in parallel:")
            st.html(chips_row_html(expansions))
        hyde = rewrite.get("hyde_query")
        if hyde:
            st.caption("HyDE hypothetical answer document (retrieved alongside the queries):")
            st.html(chips_row_html([_truncate(str(hyde), 160)], kind="c-hyde"))
    return True


def _render_retrieve_step(container: Any, details: dict[str, list[dict[str, Any]]]) -> bool:
    with container.container(border=True):
        st.markdown("**Step 2 — Hybrid retrieval** · `AsyncQdrantVectorStore.query` (dense + BM25 sparse + RRF)")
        embed_info = _detail(details, "embed")
        retrieve = _detail(details, "retrieve")
        if not retrieve:
            st.error("Retrieval produced no details — is Qdrant reachable and populated?")
            return False
        candidates: list[dict[str, Any]] = list(retrieve.get("candidates") or [])
        lookup = {str(c.get("chunk_id")): c for c in candidates}
        profiles = [str(p) for p in (retrieve.get("rrf_profiles") or [])]
        st.html(
            metric_cards_html(
                [
                    ("Query variants embedded", str(embed_info.get("variants", "?"))),
                    ("Vector dimension", str(embed_info.get("dimension", "?"))),
                    ("Query L2 norm", str(embed_info.get("l2_norm", "?"))),
                    ("Raw pool size", str(retrieve.get("pool_size", len(candidates)))),
                ]
            )
        )
        if profiles:
            st.caption("Search profile(s):")
            st.html(chips_row_html(profiles, kind="c-mode"))
        st.caption(f"Top {min(len(candidates), 8)} raw results before any reranking:")
        st.html(candidate_rows_html(candidates, lookup))
        with st.expander("Candidate snippets", expanded=False):
            for cand in candidates[:5]:
                st.markdown(f"**#{int(cand.get('rank', 0)) + 1} — {cand.get('title')}**")
                st.text(_truncate(str(cand.get("text_snippet") or ""), 300))
    return True


def _render_rerank_step(
    container: Any, details: dict[str, list[dict[str, Any]]], provenance: list[dict[str, Any]]
) -> bool:
    with container.container(border=True):
        st.markdown(f"**Step 3 — Reranking** · `{settings.reranker_type}` reranker over the fusion pool")
        rerank = _detail(details, "rerank")
        prov = provenance[0] if provenance else {}
        fused: list[dict[str, Any]] = list(prov.get("fused") or [])
        final_refs: list[dict[str, Any]] = list(prov.get("final_context") or [])
        dropped: list[dict[str, Any]] = list(prov.get("dropped") or [])
        if not fused or not final_refs:
            st.info("Provenance trace unavailable — cannot show the before/after ordering.")
            return False
        st.html(
            metric_cards_html(
                [
                    ("Reranker", "enabled" if rerank.get("enabled") else "disabled"),
                    ("Pool size", str(rerank.get("pool_size", "?"))),
                    ("Final top-K", str(rerank.get("final_top_k", "?"))),
                    ("Budget-dropped", str(len(dropped))),
                ]
            )
        )
        candidates = _detail(details, "retrieve").get("candidates") or []
        lookup = {str(c.get("chunk_id")): c for c in candidates}
        st.html(rerank_compare_html(fused, final_refs, dropped, lookup))
        st.caption("▲ moved up · ▼ moved down — scores are the reranker's normalized relevance in [0, 1].")
    return True


def _render_generate_step(container: Any, details: dict[str, list[dict[str, Any]]], answer: Answer | None) -> bool:
    with container.container(border=True):
        st.markdown(
            "**Step 4 — Context assembly & generation** · `ContextAssembler.assemble` + `PromptBuilder.build_rag_prompt` + LLM"
        )
        gen = _detail(details, "generate")
        st.html(
            metric_cards_html(
                [
                    ("Context chunks", str(gen.get("context_chunks", "?"))),
                    ("Context chars", str(gen.get("context_chars", "?"))),
                    ("Prompt chars", str(gen.get("prompt_chars", "?"))),
                    ("Model", str(gen.get("model", "?"))),
                ]
            )
        )
        if answer is not None and getattr(answer, "context", None):
            with st.expander("Assembled context sent to the LLM (truncated)", expanded=False):
                st.code(_truncate(str(answer.context), CONTEXT_PREVIEW_CHARS), language="markdown")
        if answer is None or not str(getattr(answer, "text", "")).strip():
            st.warning("No answer text was generated — the knowledge base may lack coverage or the LLM refused.")
            return False
        st.caption("Streaming the synthesized answer:")
        st.write_stream(_typewriter(answer.text))
        confidence = float(getattr(answer, "confidence", 0.0) or 0.0)
        groundedness = getattr(answer, "groundedness_score", None)
        total_ms = (
            float(answer.stage_times.get("total", 0) or 0) * 1000 if getattr(answer, "stage_times", None) else 0.0
        )
        metrics = [("Confidence", f"{confidence:.1%}"), ("Sources cited", str(len(answer.sources)))]
        if groundedness is not None:
            metrics.append(("Groundedness", f"{float(groundedness):.2f}"))
        if total_ms:
            metrics.append(("Latency", f"{total_ms:,.0f} ms"))
        st.html(metric_cards_html(metrics))
        sources = list(answer.sources)
        if sources:
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**{i}. [{src.title}]({src.url})**")
                    st.caption(f"Source: {src.source_name}")
    return True


def _render_qa_static(result: dict[str, Any]) -> None:
    answer: Answer | None = result.get("answer")
    with st.expander("Last Q&A run (static view)", expanded=False):
        if answer is None:
            st.warning(str(result.get("error") or "No answer recorded."))
            return
        st.markdown(answer.text)
        st.caption(f"Confidence {float(answer.confidence):.1%} · {len(answer.sources)} sources")


def _qa_failure_hint(error: str) -> str:
    lowered = error.lower()
    if "bm25" in lowered:
        return (
            "The collection's BM25 sparse index is not query-ready. Rebuild or activate an index generation "
            "(`dec gen-build` → `dec gen-activate`) or re-ingest the source."
        )
    if "connection" in lowered or "refused" in lowered:
        return "Qdrant or Ollama is unreachable — start the stack with `make up` and retry."
    return ""


def qa_tab() -> None:
    st.subheader("Question & answer flow")
    st.caption("Ask a question against the live index and watch retrieval, reranking, and generation unfold.")

    with st.container(border=True):
        question = st.text_input("Question", value=SAMPLE_QUESTION, key="teach_question")
        run_clicked = st.button(
            "Run QA animation",
            type="primary",
            icon=":material/play_arrow:",
            disabled=bool(st.session_state.get("teach_busy", False)),
            key="teach_run_qa",
        )

    flow_ph = st.empty()
    progress_ph = st.empty()
    step_slots = [st.empty() for _ in QA_STEPS]

    if run_clicked:
        service = rag_service()
        if service is None:
            st.error("RAG service unavailable — start the stack (`make up`) so Qdrant and Ollama are reachable.")
            return
        st.session_state["teach_busy"] = True
        try:
            progress_ph.progress(0.05, text="Executing real pipeline stages…")
            with st.spinner("Running guardrails → retrieval → rerank → generate…"):
                result = execute_qa(service, question.strip())
            answer: Answer | None = result["answer"]
            if result["error"] is not None and answer is None:
                _set_flow(flow_ph, QA_STEPS, _QA_ICONS, done=0, failed=0)
                st.error(f"Q&A failed: {result['error']}")
                hint = _qa_failure_hint(str(result["error"]))
                if hint:
                    st.info(hint)
                return
            details = result["details"]
            provenance = result["provenance"]
            renderers: tuple[Any, ...] = (
                lambda slot: _render_guard_step(slot, question.strip(), details),
                lambda slot: _render_retrieve_step(slot, details),
                lambda slot: _render_rerank_step(slot, details, provenance),
                lambda slot: _render_generate_step(slot, details, answer),
            )
            failure_node: int | None = None
            for idx, renderer in enumerate(renderers):
                _set_flow(flow_ph, QA_STEPS, _QA_ICONS, done=idx, failed=failure_node)
                ok = renderer(step_slots[idx])
                if not ok and failure_node is None:
                    failure_node = idx
                progress_ph.progress((idx + 1) / len(renderers), text=f"Step {idx + 1}/{len(renderers)} revealed")
                time.sleep(STEP_DELAY_SECONDS)
            _set_flow(flow_ph, QA_STEPS, _QA_ICONS, done=len(renderers))
            progress_ph.progress(1.0, text="Q&A walkthrough complete")
            st.session_state["teach_qa"] = result
        finally:
            st.session_state["teach_busy"] = False
    elif st.session_state.get("teach_qa") is not None:
        _set_flow(flow_ph, QA_STEPS, _QA_ICONS, done=len(QA_STEPS))
        progress_ph.progress(1.0, text="Last run result below — click the button to replay the animation")
        _render_qa_static(st.session_state["teach_qa"])


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Environment")
        q_ok, o_ok = qdrant_up(), ollama_up()
        st.badge("Qdrant: up" if q_ok else "Qdrant: down", color="green" if q_ok else "red")
        st.badge("Ollama: up" if o_ok else "Ollama: down", color="green" if o_ok else "red")
        st.markdown("### Active pipeline config")
        st.caption(
            "\n".join(
                [
                    f"- Chunker: `{settings.chunking_strategy}` ({settings.chunk_size_words}w ± {settings.chunk_overlap_words}w overlap)",
                    f"- Embeddings: `{settings.embedding_provider}/{settings.active_embedding_model_name()}` ({settings.get_embedding_dimension()}d)",
                    f"- LLM: `{settings.llm_provider}/{settings.llm_model}`",
                    f"- Reranker: `{settings.reranker_type}`",
                    f"- Retrieval top-K: {settings.retrieval_top_k} · RRF k={settings.hybrid_rrf_k}",
                    f"- Collection: `{settings.collection_name}`",
                ]
            )
        )


def main() -> None:
    if "teach_session_id" not in st.session_state:
        session_id = str(uuid.uuid4())
        st.session_state["teach_session_id"] = session_id
        st.session_state["teach_user_id"] = f"anon-{session_id[:8]}"
    st.session_state.setdefault("teach_busy", False)

    st.set_page_config(page_title="RAG Teaching Lab", page_icon=":material/school:", layout="wide")
    st.html(f"<style>{TEACH_CSS}</style>")
    st.title(":material/school: RAG teaching lab")
    st.caption("Watch a document and a question travel through every layer of the production RAG pipeline.")

    _render_sidebar()

    tab_ingest, tab_qa = st.tabs(["Ingestion flow", ":material/chat: Question / answer flow"])
    with tab_ingest:
        ingestion_tab()
    with tab_qa:
        qa_tab()


if __name__ == "__main__":
    main()
