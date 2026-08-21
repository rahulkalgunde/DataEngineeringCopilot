"""Unit tests for chunking quality metrics."""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.evaluation.chunking_metrics import (
    StructuralNode,
    char_span_to_token_interval,
    excerpt_precision,
    extract_code_structure,
    extract_markdown_structure,
    structural_fracture_rate,
    token_iou,
    tokenize_whole_doc,
)


def _chunk(start: int, end: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id="c",
        source_name="s",
        title="t",
        url="u",
        text="x" * (end - start),
        start_offset=start,
        end_offset=end,
    )


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def test_tokenize_whole_doc():
    doc = "hello world"
    ids, offsets = tokenize_whole_doc(doc)
    assert len(ids) >= 2
    assert len(offsets) == len(ids) + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(doc.encode("utf-8"))


def test_char_span_to_token_interval():
    _, offsets = tokenize_whole_doc("hello world")
    start, end = char_span_to_token_interval(0, 11, offsets)
    assert start < end
    assert start >= 0
    assert end <= len(offsets) - 1


# ---------------------------------------------------------------------------
# Token IoU
# ---------------------------------------------------------------------------


def test_identical_span_and_chunk():
    doc_text = "hello world"
    gold = [{"start": 0, "end": 11}]
    pred = [_chunk(0, 11)]
    assert token_iou(doc_text, gold, pred) == 1.0
    assert excerpt_precision(doc_text, gold, pred) == 1.0


def test_adjacent_non_overlapping():
    doc_text = "hello world"
    gold = [{"start": 0, "end": 5}]
    pred = [_chunk(5, 11)]
    assert token_iou(doc_text, gold, pred) == 0.0
    assert excerpt_precision(doc_text, gold, pred) == 0.0


def test_nested_chunk_contains_gold():
    doc_text = "hello world"
    gold = [{"start": 0, "end": 5}]
    pred = [_chunk(0, 11)]
    assert 0.0 < token_iou(doc_text, gold, pred) < 1.0
    assert 0.0 < excerpt_precision(doc_text, gold, pred) < 1.0


def test_empty_gold_spans():
    doc_text = "hello"
    pred = [_chunk(0, 5)]
    assert token_iou(doc_text, [], pred) == 0.0
    assert excerpt_precision(doc_text, [], pred) == 0.0


def test_empty_pred_chunks():
    doc_text = "hello"
    gold = [{"start": 0, "end": 5}]
    assert token_iou(doc_text, gold, []) == 0.0
    assert excerpt_precision(doc_text, gold, []) == 0.0


# ---------------------------------------------------------------------------
# SegEval boundary metrics
# ---------------------------------------------------------------------------


def test_boundary_similarity_perfect():
    gold = [{"start": 0, "end": 100}]
    pred = [_chunk(0, 100)]
    doc_length = 100
    pytest.importorskip("segeval")
    from data_engineering_copilot.evaluation.chunking_metrics import (
        boundary_similarity,
        pk,
        windowdiff,
    )

    assert boundary_similarity(gold, pred, doc_length) == 1.0
    assert pk(gold, pred, doc_length) == 0.0
    assert windowdiff(gold, pred, doc_length) == 0.0


def test_boundary_similarity_shifted():
    gold = [{"start": 0, "end": 100}, {"start": 200, "end": 300}]
    pred = [_chunk(0, 105), _chunk(105, 200), _chunk(200, 300)]
    doc_length = 300
    pytest.importorskip("segeval")
    from data_engineering_copilot.evaluation.chunking_metrics import (
        boundary_similarity,
        pk,
        windowdiff,
    )

    assert boundary_similarity(gold, pred, doc_length) < 1.0
    assert pk(gold, pred, doc_length) > 0.0
    assert windowdiff(gold, pred, doc_length) > 0.0


def test_boundary_empty():
    gold: list[dict] = []
    pred: list[DocumentChunk] = []
    doc_length = 100
    pytest.importorskip("segeval")
    from data_engineering_copilot.evaluation.chunking_metrics import (
        boundary_similarity,
        pk,
        windowdiff,
    )

    assert boundary_similarity(gold, pred, doc_length) == 1.0
    assert pk(gold, pred, doc_length) == 0.0
    assert windowdiff(gold, pred, doc_length) == 0.0


# ---------------------------------------------------------------------------
# Structural fracture rate
# ---------------------------------------------------------------------------


def test_fracture_no_fracture():
    nodes = [StructuralNode("header", 0, 10, "# Header")]
    pred = [_chunk(10, 30)]
    assert structural_fracture_rate(pred, nodes) == 0.0


def test_fracture_with_fracture():
    nodes = [StructuralNode("header", 0, 10, "# Header")]
    pred = [_chunk(0, 5)]
    assert structural_fracture_rate(pred, nodes) == 1.0


def test_fracture_empty_nodes():
    pred = [_chunk(0, 10)]
    assert structural_fracture_rate(pred, []) == 0.0


def test_extract_markdown_structure_simple():
    doc = "# Header\n\nParagraph.\n"
    nodes = extract_markdown_structure(doc)
    assert nodes
    assert nodes[0].node_type == "header"
    assert nodes[0].content == "# Header"


def test_extract_code_structure_python():
    doc = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    nodes = extract_code_structure(doc, "python")
    types = [n.node_type for n in nodes]
    assert "function_definition" in types


def test_extract_code_structure_unsupported():
    assert extract_code_structure("some text", "scala") == []
