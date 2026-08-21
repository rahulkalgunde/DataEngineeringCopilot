"""Invariant tests: chunks must carry correct character offsets.

These verify the P0 task — every chunker populates ``start_offset`` /
``end_offset`` and (for chunkers that emit contiguous text) the chunk text is
exactly ``doc.text[start_offset:end_offset]``. Semantic and header-aware
chunkers join/merge sections and so are not included in the strict-slice
assertion; they are exercised by the reconstruction and bounds checks.
"""

import pytest

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.evaluation.chunking_metrics import (
    boundary_similarity,
    extract_markdown_structure,
    structural_fracture_rate,
    token_iou,
)
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.sentence_preserving_chunker import SentencePreservingChunker
from data_engineering_copilot.services.structured_data_chunker import StructuredDataChunker

DOC = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source_name="test", title="t", url="http://x", text=text)


# Chunkers whose emitted text is a contiguous substring of the source.
SLICE_CHUNKERS = [
    ("document", lambda: DocumentChunker(chunk_size_chars=100, chunk_overlap_chars=20)),
    ("sentence", lambda: SentencePreservingChunker(max_tokens=3800, max_chars=6000)),
    ("header", lambda: HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)),
    ("structured", lambda: StructuredDataChunker(max_tokens=3800, max_chars=6000)),
]

# Chunkers that preserve every source character (so joined chunks reconstruct
# the document). DocumentChunker uses langchain's splitter which drops
# separators, so it is excluded from the reconstruction check.
RECON_CHUNKERS = [
    ("sentence", lambda: SentencePreservingChunker(max_tokens=3800, max_chars=6000)),
    ("header", lambda: HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)),
    ("structured", lambda: StructuredDataChunker(max_tokens=3800, max_chars=6000)),
]


@pytest.mark.parametrize("name,factory", SLICE_CHUNKERS)
def test_chunker_offsets_are_slices(name, factory):
    doc = _doc(DOC)
    chunker = factory()
    chunks = chunker._sync_chunk(doc)
    assert chunks, f"{name} produced no chunks"
    for chunk in chunks:
        assert chunk.text == doc.text[chunk.start_offset : chunk.end_offset], f"{name} slice mismatch"
        assert 0 <= chunk.start_offset <= chunk.end_offset <= len(doc.text), f"{name} bounds"


@pytest.mark.parametrize("name,factory", RECON_CHUNKERS)
def test_chunker_reconstruction(name, factory):
    doc = _doc(DOC)
    chunker = factory()
    chunks = chunker._sync_chunk(doc)
    assert "".join(c.text for c in chunks) == doc.text, f"{name} reconstruction"


@pytest.mark.slow
@pytest.mark.parametrize("name,factory", SLICE_CHUNKERS)
def test_golden_regression(name, factory, gold_chunking_dataset):
    """Smoke-test that chunkers produce valid metrics against the gold dataset.

    Initial baselines are permissive; tighten as fixtures and chunkers mature.
    """
    for gold_doc in gold_chunking_dataset:
        doc = _doc(gold_doc.text)
        chunker = factory()
        chunks = chunker._sync_chunk(doc)
        assert chunks, f"{name} produced no chunks for {gold_doc.doc_id}"
        assert token_iou(gold_doc.text, gold_doc.gold_spans, chunks) >= 0.0
        assert boundary_similarity(gold_doc.gold_spans, chunks, len(gold_doc.text)) >= 0.0
        if gold_doc.text.lstrip().startswith("#") or "```" in gold_doc.text:
            nodes = extract_markdown_structure(gold_doc.text)
            assert structural_fracture_rate(chunks, nodes) <= 1.0
