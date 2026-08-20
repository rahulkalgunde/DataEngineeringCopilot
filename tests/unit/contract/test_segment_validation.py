"""Test segment validation groups by parent_chunk_id, not parent_content_hash."""

from __future__ import annotations

from data_engineering_copilot.services.hierarchical_chunker import hierarchical_chunk
from data_engineering_copilot.services.spark_index_builder import _validate_segment_budgets


def test_validation_groups_by_parent_chunk_id_not_content_hash(duplicate_parent_texts):
    all_chunks = []
    for parent in duplicate_parent_texts:
        children = hierarchical_chunk(parent, parent_max_tokens=1024, child_max_tokens=256)
        all_chunks.extend(children)

    failures = _validate_segment_budgets(all_chunks)
    assert failures == [], f"Unexpected failures: {failures}"
