"""Tests for evaluation/chunking_gold.py."""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.chunking_gold import (
    ChunkingGoldDoc,
    ChunkingGoldSpan,
    validate_gold_doc,
)


def _make_doc(text: str = "hello world", spans: list[ChunkingGoldSpan] | None = None) -> ChunkingGoldDoc:
    if spans is None:
        spans = [ChunkingGoldSpan(content="hello", start=0, end=5, structural_type="text")]
    return ChunkingGoldDoc(doc_id="doc1", text=text, gold_spans=spans)


class TestValidateGoldDoc:
    def test_valid_passes(self) -> None:
        doc = _make_doc()
        validate_gold_doc(doc)

    def test_out_of_bounds(self) -> None:
        doc = _make_doc("hello", [ChunkingGoldSpan(content="hello", start=0, end=10, structural_type="text")])
        with pytest.raises(ValueError, match="out of bounds"):
            validate_gold_doc(doc)

    def test_content_mismatch(self) -> None:
        doc = _make_doc("hello world", [ChunkingGoldSpan(content="wrong", start=0, end=5, structural_type="text")])
        with pytest.raises(ValueError, match="mismatch"):
            validate_gold_doc(doc)

    def test_empty_spans(self) -> None:
        doc = _make_doc("hello", [])
        validate_gold_doc(doc)
