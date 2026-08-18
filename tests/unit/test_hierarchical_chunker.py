"""Phase 5 tests: hierarchical (parent-child) chunking."""

from __future__ import annotations

import hashlib

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.infrastructure.token_budget import count_tokens
from data_engineering_copilot.services.hierarchical_chunker import _merge_blank_pieces, hierarchical_chunk


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


def test_oversized_atomic_piece_falls_back_to_larger_child_budget() -> None:
    """A single atomic line/URL longer than the child budget must not fail the
    build; the child split falls back to the parent budget / hard cap."""
    # One fenced line exceeds the 256-token child char budget (1024 chars)
    # but is a single atomic piece that cannot be split losslessly.
    fence = "```python\n" + "y" * 1500 + "\n```"
    result = hierarchical_chunk(_chunk(fence))
    # No ValueError; parents + children produced, still lossless.
    parents = [c for c in result if not c.parent_chunk_id]
    children = [c for c in result if c.parent_chunk_id]
    assert parents and children
    for parent in parents:
        siblings = sorted(
            (c for c in result if c.parent_chunk_id == parent.chunk_id),
            key=lambda c: c.segment_index,
        )
        assert siblings
        assert "".join(s.text for s in siblings).strip() == parent.text.strip()


def test_oversized_plain_token_falls_back_to_larger_budget() -> None:
    """A single plain (non-fenced) token longer than the child budget must not
    raise; the fallback splits at the provider hard cap."""
    text = "before " + "z" * 1500 + " after"
    result = hierarchical_chunk(_chunk(text))
    assert result, "must not raise for an oversized atomic token"


def test_merge_blank_pieces_folds_whitespace_into_neighbors() -> None:
    """Whitespace-only pieces must be folded into content neighbors while
    preserving every character and the original ordering (lossless join)."""
    pieces = ["a" * 10, "\n\n", "b" * 10, " ", "c" * 10, "\n", ""]
    merged = _merge_blank_pieces(pieces)
    assert "".join(merged) == "".join(pieces), "merge must be lossless"
    assert all(p.strip() for p in merged), "no blank piece may remain"
    assert merged == ["a" * 10, "\n\n" + "b" * 10, " " + "c" * 10 + "\n"]


def test_merge_blank_pieces_leading_and_all_blank() -> None:
    assert _merge_blank_pieces(["\n\n", "x"]) == ["\n\nx"]
    assert _merge_blank_pieces(["x", "\n\n"]) == ["x\n\n"]
    assert _merge_blank_pieces(["x", "\n\n", "y"]) == ["x", "\n\ny"]
    assert _merge_blank_pieces(["\n\n"]) == ["\n\n"]


def test_no_blank_chunks_from_paragraph_separator_before_fence() -> None:
    """Regression: a paragraph separator (``"\\n\\n"``) atom emitted as its own
    child produced whitespace-only chunks that embedding providers reject with
    HTTP 400. Hierarchical output must never contain a blank chunk, and the
    children must still reconstruct the parent losslessly."""
    # Real Spark SQL doc parent text that reproduced the standalone "\n\n" child
    # (a paragraph separator atom preceding an atomic code fence).
    parent_text = (
        "Under ANSI mode(spark.sql.ansi.enabled=true), the function invocation of Spark SQL:\n"
        "- In general, it follows the `Store assignment` rules as storing the input values "
        "as the declared parameter type of the SQL functions\n"
        "- Special rules apply for untyped NULL. A NULL can be promoted to any other type.\n"
        "\n"
        "```sql\n"
        "> SET spark.sql.ansi.enabled=true;\n"
        "-- implicitly cast Int to String type\n"
        "> SELECT concat('total number: ', 1);\n"
        "total number: 1\n"
        "-- implicitly cast Timestamp to Date type\n"
        "> select datediff(now(), current_date);\n"
        "0\n"
        "\n"
        "-- implicitly cast String to Double type\n"
        "> SELECT ceil('0.1');\n"
        "1\n"
        "-- special rule: implicitly cast NULL to Date type\n"
        "> SELECT year(null);\n"
        "NULL\n"
        "\n"
        "> CREATE TABLE t(s string);\n"
        "-- Can't store String column as Numeric types.\n"
        "> SELECT ceil(s) from t;\n"
        "Error in query: cannot resolve 'CEIL(spark_catalog.default.t.s)' due to data type mismatch\n"
        "-- Can't store String column as Date type.\n"
        "> select year(s) from t;\n"
        "Error in query: cannot resolve 'year(spark_catalog.default.t.s)' due to data type mismatch\n"
        "```\n"
        "\n"
        "The behavior of some SQL functions can be different under ANSI mode "
        "(`spark.sql.ansi.enabled=true`).\n"
        "  - `size`: This function returns null for null input.\n"
        "  - `element_at`: This function throws `ArrayIndexOutOfBoundsException` if using "
        "invalid indices.\n"
        "  - `parse_url`: This function throws `IllegalArgumentException` if an input string "
        "is not a valid url.\n"
        "  - `to_date`: This function should fail with an exception if the input string can't "
        "be parsed, or the pattern string is invalid.\n"
        "  - `to_timestamp`: This function should fail with an exception if the input string "
        "can't be parsed, or the pattern string is invalid.\n"
        "  - `make_date`: This function should fail with an exception if the result date is "
        "invalid.\n"
        "  - `make_timestamp`: This function should fail with an exception if the result "
        "timestamp is invalid.\n"
        "  - `next_day`: This function throws `IllegalArgumentException` if input is not a "
        "valid day of week.\n"
    )
    assert len(parent_text) > 1200, "parent_text must exceed the child budget to split"
    result = hierarchical_chunk(_chunk(parent_text), child_max_tokens=256, parent_max_tokens=1024)

    blanks = [c for c in result if not c.text.strip()]
    assert blanks == [], f"hierarchical output must not contain blank chunks: {[c.chunk_id for c in blanks]}"

    for parent in (c for c in result if not c.parent_chunk_id):
        siblings = sorted(
            (c for c in result if c.parent_chunk_id == parent.chunk_id),
            key=lambda c: c.segment_index,
        )
        assert siblings, "parent must have children"
        assert "".join(s.text for s in siblings).strip() == parent.text.strip(), "lossless reconstruction"


def test_no_blank_chunks_for_corpus_shaped_paragraph_and_fence_text() -> None:
    """Invariant across corpus-shaped inputs: chunking a long paragraph + code
    fence + paragraph combination must never yield a blank chunk."""
    para = " ".join(["word"] * 400)
    fence = "```sql\n> SET spark.sql.ansi.enabled=true;\n-- implicitly cast Int to String\n```"
    for text in (f"{para}\n\n{fence}\n\n{para}\n", f"{para}\n\n{fence}\n\n"):
        result = hierarchical_chunk(_chunk(text), child_max_tokens=256, parent_max_tokens=1024)
        assert all(c.text.strip() for c in result), "no chunk may be blank"
        for parent in (c for c in result if not c.parent_chunk_id):
            siblings = sorted(
                (c for c in result if c.parent_chunk_id == parent.chunk_id),
                key=lambda c: c.segment_index,
            )
            if siblings:
                assert "".join(s.text for s in siblings).strip() == parent.text.strip(), "lossless reconstruction"
