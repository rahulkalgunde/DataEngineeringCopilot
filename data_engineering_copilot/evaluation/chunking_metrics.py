"""Offline chunking-quality metrics.

All functions in this module are deterministic and require no LLM calls.

Metrics
-------
- **Token-level IoU** (`token_iou`) and **Excerpt Precision** (`excerpt_precision`)
- **SegEval boundary metrics** (`boundary_similarity`, `pk`, `windowdiff`)
- **Structural Boundary Fracture Rate** (`structural_fracture_rate`)
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import tiktoken

from data_engineering_copilot.domain.models import DocumentChunk

_ENCODER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_whole_doc(doc: str) -> tuple[list[int], list[int]]:
    """Encode *doc* and build a cumulative byte-offset table.

    Returns ``(token_ids, byte_offsets)`` where ``byte_offsets[i]`` is the byte
    position of token ``i`` in the UTF-8 encoded original string.
    ``byte_offsets`` has length ``len(token_ids) + 1`` so that ``byte_offsets[-1]``
    equals ``len(doc.encode("utf-8"))``.
    """
    token_ids = _ENCODER.encode(doc)
    byte_offsets = [0] * (len(token_ids) + 1)
    for i in range(len(token_ids)):
        prefix = _ENCODER.decode(token_ids[: i + 1])
        byte_offsets[i + 1] = len(prefix.encode("utf-8"))
    return token_ids, byte_offsets


def char_span_to_token_interval(
    start: int, end: int, byte_offsets: list[int]
) -> tuple[int, int]:
    """Map a character ``[start, end)`` span to a token index interval.

    Uses ``bisect_left`` on the cumulative byte table so the mapping is stable
    regardless of whether the character boundary falls inside a token or on a
    token boundary.
    """
    token_start = bisect.bisect_left(byte_offsets, start)
    token_end = bisect.bisect_left(byte_offsets, end)
    return token_start, token_end


# ---------------------------------------------------------------------------
# Token IoU + Excerpt Precision
# ---------------------------------------------------------------------------


def _span_start(span) -> int:
    return span.start if hasattr(span, "start") else span["start"]


def _span_end(span) -> int:
    return span.end if hasattr(span, "end") else span["end"]


def token_iou(
    doc_text: str, gold_spans: list[dict], pred_chunks: list[DocumentChunk]
) -> float:
    """Mean token-level Intersection-over-Union against gold spans."""
    if not pred_chunks or not gold_spans:
        return 0.0
    _, byte_offsets = tokenize_whole_doc(doc_text)
    ious: list[float] = []
    for span in gold_spans:
        g_start, g_end = char_span_to_token_interval(_span_start(span), _span_end(span), byte_offsets)
        best_iou = 0.0
        for chunk in pred_chunks:
            c_start, c_end = char_span_to_token_interval(
                chunk.start_offset, chunk.end_offset, byte_offsets
            )
            inter = max(0, min(g_end, c_end) - max(g_start, c_start))
            union = max(g_end, c_end) - min(g_start, c_start)
            if union > 0:
                best_iou = max(best_iou, inter / union)
        ious.append(best_iou)
    return sum(ious) / len(ious) if ious else 0.0


def excerpt_precision(
    doc_text: str, gold_spans: list[dict], pred_chunks: list[DocumentChunk]
) -> float:
    """Mean excerpt precision: ``|gold ∩ best_pred| / |best_pred|``."""
    if not pred_chunks or not gold_spans:
        return 0.0
    _, byte_offsets = tokenize_whole_doc(doc_text)
    precisions: list[float] = []
    for span in gold_spans:
        g_start, g_end = char_span_to_token_interval(_span_start(span), _span_end(span), byte_offsets)
        best_overlap = 0
        best_pred_size = 1
        for chunk in pred_chunks:
            c_start, c_end = char_span_to_token_interval(
                chunk.start_offset, chunk.end_offset, byte_offsets
            )
            inter = max(0, min(g_end, c_end) - max(g_start, c_start))
            pred_size = max(1, c_end - c_start)
            if inter > best_overlap:
                best_overlap = inter
                best_pred_size = pred_size
        precisions.append(best_overlap / best_pred_size)
    return sum(precisions) / len(precisions) if precisions else 0.0


# ---------------------------------------------------------------------------
# SegEval boundary metrics
# ---------------------------------------------------------------------------

# ``segeval`` is an optional dev dependency. Import lazily so the module can
# still be imported in environments where it is not installed (e.g. minimal
# test runs); the boundary metric functions will raise ``ImportError`` if
# ``segeval`` is absent.
try:
    import segeval  # noqa: F401

    _SEGEVAL_AVAILABLE = True
except ImportError:
    _SEGEVAL_AVAILABLE = False


def _boundaries_to_segments(boundaries: list[int], doc_length: int) -> list[int]:
    """Convert a sorted list of boundary positions to SegEval segment sizes."""
    if not boundaries:
        return [doc_length]
    sorted_b = sorted(set(boundaries))
    if sorted_b[0] != 0:
        sorted_b = [0] + sorted_b
    if sorted_b[-1] != doc_length:
        sorted_b = sorted_b + [doc_length]
    return [sorted_b[i + 1] - sorted_b[i] for i in range(len(sorted_b) - 1)]


def gold_spans_to_boundaries(gold_spans: list[dict], doc_length: int) -> list[int]:
    """Return the sorted unique boundary positions implied by gold spans."""
    boundaries = {0, doc_length}
    for span in gold_spans:
        boundaries.add(_span_start(span))
        boundaries.add(_span_end(span))
    return sorted(boundaries)


def pred_chunks_to_boundaries(pred_chunks: list[DocumentChunk], doc_length: int) -> list[int]:
    """Return the sorted unique boundary positions implied by pred chunks."""
    boundaries: set[int] = {0, doc_length}
    for chunk in pred_chunks:
        boundaries.add(chunk.start_offset)
        boundaries.add(chunk.end_offset)
    return sorted(boundaries)


def _boundaries_to_masses(boundaries: list[int], doc_length: int) -> tuple[int, ...]:
    """Convert sorted boundary positions to a mass tuple."""
    if not boundaries:
        return (doc_length,)
    sorted_b = sorted(set(boundaries))
    if sorted_b[0] != 0:
        sorted_b = [0] + sorted_b
    if sorted_b[-1] != doc_length:
        sorted_b = sorted_b + [doc_length]
    return tuple(sorted_b[i + 1] - sorted_b[i] for i in range(len(sorted_b) - 1))


def boundary_similarity(
    gold_spans: list[dict], pred_chunks: list[DocumentChunk], doc_length: int
) -> float:
    """Wrap ``segeval.boundary_similarity`` for chunk boundaries."""
    if not _SEGEVAL_AVAILABLE:
        raise ImportError("segeval is required for boundary_similarity")
    gold_masses = _boundaries_to_masses(gold_spans_to_boundaries(gold_spans, doc_length), doc_length)
    pred_masses = _boundaries_to_masses(pred_chunks_to_boundaries(pred_chunks, doc_length), doc_length)
    if len(gold_masses) == 1 and len(pred_masses) == 1:
        return 1.0
    return float(segeval.boundary_similarity(gold_masses, pred_masses))  # type: ignore[possibly-undefined]


def pk(gold_spans: list[dict], pred_chunks: list[DocumentChunk], doc_length: int) -> float:
    """Wrap ``segeval.pk`` for chunk boundaries."""
    if not _SEGEVAL_AVAILABLE:
        raise ImportError("segeval is required for pk")
    gold_masses = _boundaries_to_masses(gold_spans_to_boundaries(gold_spans, doc_length), doc_length)
    pred_masses = _boundaries_to_masses(pred_chunks_to_boundaries(pred_chunks, doc_length), doc_length)
    if len(gold_masses) == 1 and len(pred_masses) == 1:
        return 0.0
    return float(segeval.pk(gold_masses, pred_masses))  # type: ignore[possibly-undefined]


def windowdiff(
    gold_spans: list[dict], pred_chunks: list[DocumentChunk], doc_length: int
) -> float:
    """Wrap ``segeval.windowdiff`` for chunk boundaries."""
    if not _SEGEVAL_AVAILABLE:
        raise ImportError("segeval is required for windowdiff")
    gold_masses = _boundaries_to_masses(gold_spans_to_boundaries(gold_spans, doc_length), doc_length)
    pred_masses = _boundaries_to_masses(pred_chunks_to_boundaries(pred_chunks, doc_length), doc_length)
    if len(gold_masses) == 1 and len(pred_masses) == 1:
        return 0.0
    return float(segeval.window_diff(gold_masses, pred_masses))  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# Structural Boundary Fracture Rate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralNode:
    """A structural unit extracted from a document."""

    node_type: str
    start: int
    end: int
    content: str


def extract_markdown_structure(doc: str) -> list[StructuralNode]:
    """Extract markdown structural nodes using ``markdown-it-py``."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return []

    md = MarkdownIt()
    tokens = md.parse(doc)
    lines = doc.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def char_at(line_no: int) -> int:
        return line_offsets[min(line_no, len(line_offsets) - 1)]

    nodes: list[StructuralNode] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            start = char_at(tok.map[0])  # type: ignore[optional-subscript]
            end = doc.find("\n", start)
            if end == -1:
                end = len(doc)
            nodes.append(StructuralNode("header", start, end, doc[start:end]))
            i += 1
            continue
        if tok.type == "fence":
            start = char_at(tok.map[0])  # type: ignore[optional-subscript]
            end = char_at(tok.map[1])  # type: ignore[optional-subscript]
            nodes.append(StructuralNode("code_fence", start, end, doc[start:end]))
            i += 1
            continue
        i += 1
    return nodes


def extract_code_structure(doc: str, language: str) -> list[StructuralNode]:
    """Extract function/class nodes from Python source via tree-sitter."""
    if language != "python":
        return []
    try:
        from tree_sitter import Language, Parser  # type: ignore[import-not-found]
        from tree_sitter_python import language as python_language  # type: ignore[import-not-found]
    except ImportError:
        return []

    parser = Parser(Language(python_language()))
    tree = parser.parse(doc.encode("utf-8"))
    nodes: list[StructuralNode] = []

    def _walk(node):
        if node.type in ("function_definition", "class_definition"):
            start = len(
                doc.encode("utf-8")[: node.start_byte].decode("utf-8", errors="replace")  # type: ignore[optional-subscript]
            )
            end = len(
                doc.encode("utf-8")[: node.end_byte].decode("utf-8", errors="replace")  # type: ignore[optional-subscript]
            )
            nodes.append(StructuralNode(node.type, start, end, doc[start:end]))
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return nodes


def structural_fracture_rate(
    pred_chunks: list[DocumentChunk], structural_nodes: list[StructuralNode]
) -> float:
    """Fraction of structural nodes split by at least one chunk boundary."""
    if not structural_nodes:
        return 0.0
    boundaries: set[int] = set()
    for chunk in pred_chunks:
        boundaries.add(chunk.start_offset)
        boundaries.add(chunk.end_offset)
    fractured = 0
    for node in structural_nodes:
        for b in boundaries:
            if node.start < b < node.end:
                fractured += 1
                break
    return fractured / len(structural_nodes)
