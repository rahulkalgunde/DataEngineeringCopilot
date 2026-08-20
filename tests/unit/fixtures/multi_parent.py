"""Fixtures for multi-parent validation scenarios."""

from __future__ import annotations

import hashlib

import pytest

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.infrastructure.token_budget import count_tokens


@pytest.fixture
def duplicate_parent_texts() -> list[DocumentChunk]:
    """Two distinct parents with IDENTICAL text - tests grouping by chunk_id not content_hash."""
    parent_text = "```python\nx = 1\n```\n" + "word " * 500
    return [
        DocumentChunk(
            chunk_id="parent-A",
            source_name="Test",
            title="T",
            url="http://x",
            text=parent_text,
            content_hash=hashlib.sha256(parent_text.encode()).hexdigest(),
            doc_type="guide",
            language="conceptual",
            index_generation="gen-1",
            source_commit="a" * 40,
            parent_content_hash="",
            segment_index=0,
            segment_total=1,
            token_count=count_tokens(parent_text),
            character_count=len(parent_text),
            parent_chunk_id="",
        ),
        DocumentChunk(
            chunk_id="parent-B",
            source_name="Test",
            title="T",
            url="http://y",
            text=parent_text,
            content_hash=hashlib.sha256(parent_text.encode()).hexdigest(),
            doc_type="guide",
            language="conceptual",
            index_generation="gen-1",
            source_commit="a" * 40,
            parent_content_hash="",
            segment_index=0,
            segment_total=1,
            token_count=count_tokens(parent_text),
            character_count=len(parent_text),
            parent_chunk_id="",
        ),
    ]
