"""Syrupy snapshot tests for chunking offsets.

On first run, set ``PYTEST_SNAPSHOT_UPDATE=1`` to write baseline snapshots.
Subsequent runs diff against the stored snapshots so any offset change is
flagged immediately.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.sentence_preserving_chunker import SentencePreservingChunker
from data_engineering_copilot.services.structured_data_chunker import StructuredDataChunker


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source_name="test", title="t", url="http://x", text=text)


CHUNKERS = [
    ("document", lambda: DocumentChunker(chunk_size_chars=100, chunk_overlap_chars=20)),
    ("sentence", lambda: SentencePreservingChunker(max_tokens=3800, max_chars=6000)),
    ("header", lambda: HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)),
    ("structured", lambda: StructuredDataChunker(max_tokens=3800, max_chars=6000)),
]


@pytest.mark.parametrize("name,factory", CHUNKERS)
def test_chunk_snapshots(name, factory, snapshot, gold_chunking_dataset):
    gold_doc = gold_chunking_dataset[0]
    doc = _doc(gold_doc.text)
    chunker = factory()
    chunks = chunker._sync_chunk(doc)
    data = [(c.start_offset, c.end_offset, c.text[:50]) for c in chunks]
    assert data == snapshot(name=name)
