"""Phase 5 tests: hierarchical (parent-child) chunking."""

from __future__ import annotations

import hashlib

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.infrastructure.token_budget import count_tokens
from data_engineering_copilot.services.hierarchical_chunker import hierarchical_chunk


def _chunk(text: str, chunk_id: str = "c0") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name="Apache Spark 4.0.0",
        title="Doc",
        url="http://x",
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        doc_type="guide",
        language="conceptual",
        index_generation="gen-1",
        source_commit="a" * 40,
    )


def test_small_chunk_returns_as_is() -> None:
    chunk = _chunk("Spark SQL supports window functions. " * 5)
    result = hierarchical_chunk(chunk)
    assert len(result) == 1
    assert result[0].chunk_id == "c0"
    assert result[0].parent_chunk_id == ""


def test_large_chunk_produces_parent_and_children() -> None:
    # ~1500 words forces a parent split into children under the child budget.
    text = " ".join(f"word{i}" for i in range(1500))
    chunk = _chunk(text)
    result = hierarchical_chunk(chunk)

    parents = [c for c in result if not c.parent_chunk_id]
    children = [c for c in result if c.parent_chunk_id]
    assert parents, "hierarchical output must contain a parent chunk"
    assert children, "hierarchical output must contain child chunks"

    parent_ids = {c.chunk_id for c in parents}
    for child in children:
        assert child.parent_chunk_id in parent_ids, "child must reference a persisted parent"


def test_parent_has_empty_parent_chunk_id_children_set() -> None:
    text = " ".join(f"word{i}" for i in range(1500))
    result = hierarchical_chunk(_chunk(text))

    parents = [c for c in result if not c.parent_chunk_id]
    children = [c for c in result if c.parent_chunk_id]
    assert parents
    assert children
    # Each child references the parent it was split from.
    parent_ids = {c.chunk_id for c in parents}
    assert all(c.parent_chunk_id in parent_ids for c in children)


def test_children_reconstruct_parent_text_losslessly() -> None:
    text = " ".join(f"word{i}" for i in range(1500))
    result = hierarchical_chunk(_chunk(text))

    for parent in (c for c in result if not c.parent_chunk_id):
        siblings = sorted(
            (c for c in result if c.parent_chunk_id == parent.chunk_id),
            key=lambda c: c.segment_index,
        )
        assert siblings, "parent must have children"
        joined = "".join(s.text for s in siblings).strip()
        assert joined == parent.text.strip(), "children must reconstruct parent text losslessly"


def test_all_children_satisfy_token_budget() -> None:
    text = " ".join(f"word{i}" for i in range(3000))
    result = hierarchical_chunk(_chunk(text), child_max_tokens=256)

    for child in (c for c in result if c.parent_chunk_id):
        assert count_tokens(child.text) <= 256
    # Parents are bounded by the parent budget too.
    for parent in (c for c in result if not c.parent_chunk_id):
        assert count_tokens(parent.text) <= 1024


def test_segment_metadata_validates_under_budget_checks() -> None:
    """Hierarchical output must pass _validate_segment_budgets' group checks."""
    from data_engineering_copilot.services.spark_index_builder import _validate_segment_budgets

    text = " ".join(f"word{i}" for i in range(3000))
    result = hierarchical_chunk(_chunk(text))
    assert _validate_segment_budgets(result) == []


def test_child_ids_are_unique() -> None:
    text = " ".join(f"word{i}" for i in range(3000))
    result = hierarchical_chunk(_chunk(text))
    ids = [c.chunk_id for c in result]
    assert len(ids) == len(set(ids))
