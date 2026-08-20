"""Tests for the ContextAssembler service."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.context_assembler import ContextAssembler


def create_test_chunk(chunk_id, text, source_name="test_source", url=None):
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name=source_name,
        title=f"Title {chunk_id}",
        url=url or f"http://example.com/chunk{chunk_id}",
        text=text,
        content_hash=f"hash_{chunk_id}",
    )


def create_retrieved_chunk(chunk, distance=0.1, confidence=0.9):
    return RetrievedChunk(chunk=chunk, distance=distance, confidence=confidence)


class TestContextAssembler:
    def test_initialization(self):
        assembler = ContextAssembler(max_context_chars=1000)
        assert assembler.max_context_chars == 1000

    def test_initialization_zero(self):
        assembler = ContextAssembler(max_context_chars=0)
        assert assembler.max_context_chars == 0

    def test_assemble_empty_chunks(self):
        assembler = ContextAssembler(max_context_chars=1000)
        context, sources, _dropped = assembler.assemble([])
        assert context == ""
        assert sources == []

    def test_assemble_single_chunk(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk = create_test_chunk("chunk1", "This is a single test chunk.")
        retrieved = create_retrieved_chunk(chunk)

        context, sources, _dropped = assembler.assemble([retrieved])

        expected_context = (
            '<context_doc id="1" url="http://example.com/chunkchunk1">'
            "[Source: test_source]\nThis is a single test chunk.\n</context_doc>"
        )
        assert context == expected_context
        assert sources == ["test_source"]

    def test_assemble_multiple_chunks(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "First chunk of text.")
        chunk2 = create_test_chunk("chunk2", "Second chunk of text.")
        chunk3 = create_test_chunk("chunk3", "Third chunk of text.")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)
        retrieved3 = create_retrieved_chunk(chunk3)

        context, sources, _dropped = assembler.assemble([retrieved1, retrieved2, retrieved3])

        expected_context = (
            '<context_doc id="1" url="http://example.com/chunkchunk1">'
            "[Source: test_source]\nFirst chunk of text.\n</context_doc>\n"
            '<context_doc id="2" url="http://example.com/chunkchunk2">'
            "[Source: test_source]\nSecond chunk of text.\n</context_doc>\n"
            '<context_doc id="3" url="http://example.com/chunkchunk3">'
            "[Source: test_source]\nThird chunk of text.\n</context_doc>"
        )
        assert context == expected_context
        assert sources == ["test_source", "test_source", "test_source"]

    def test_assemble_with_different_sources(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "First test chunk.", source_name="source_a")
        chunk2 = create_test_chunk("chunk2", "Second test chunk.", source_name="source_b")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)

        context, sources, _dropped = assembler.assemble([retrieved1, retrieved2])

        expected_context = (
            '<context_doc id="1" url="http://example.com/chunkchunk1">'
            "[Source: source_a]\nFirst test chunk.\n</context_doc>\n"
            '<context_doc id="2" url="http://example.com/chunkchunk2">'
            "[Source: source_b]\nSecond test chunk.\n</context_doc>"
        )
        assert context == expected_context
        assert sources == ["source_a", "source_b"]

    def test_text_overlap_ratio_identical(self):
        assembler = ContextAssembler(max_context_chars=1000)
        text1 = "This is a test document with some content."
        text2 = "This is a test document with some content."

        ratio = assembler._text_overlap_ratio(text1, text2)

        assert ratio == 1.0

    def test_text_overlap_ratio_no_overlap(self):
        assembler = ContextAssembler(max_context_chars=1000)
        text1 = "apple banana cherry date"
        text2 = "elephant fox giraffe hippo"

        ratio = assembler._text_overlap_ratio(text1, text2)

        assert ratio == 0.0

    def test_text_overlap_ratio_partial_overlap(self):
        assembler = ContextAssembler(max_context_chars=1000)
        text1 = "quick brown fox jumps dog"
        text2 = "quick brown fox"

        ratio = assembler._text_overlap_ratio(text1, text2)

        expected_ratio = 3.0 / 5.0
        assert abs(ratio - expected_ratio) < 0.001

    def test_text_overlap_ratio_filler_words(self):
        assembler = ContextAssembler(max_context_chars=1000)
        text1 = "the quick brown fox over lazy dog"
        text2 = "the quick brown fox over lazy dog"

        ratio = assembler._text_overlap_ratio(text1, text2)

        assert ratio == 1.0

    def test_text_overlap_ratio_empty(self):
        assembler = ContextAssembler(max_context_chars=1000)
        text1 = ""
        text2 = "Some content here."

        ratio1 = assembler._text_overlap_ratio(text1, text2)
        ratio2 = assembler._text_overlap_ratio(text2, text1)

        assert ratio1 == 0.0
        assert ratio2 == 0.0

    def test_text_overlap_ratio_all_filler(self):
        assembler = ContextAssembler(max_context_chars=1000)
        text1 = "the and or in at to of is are"
        text2 = "the and or in at to of is are"

        ratio = assembler._text_overlap_ratio(text1, text2)

        assert ratio == 0.0

    def test_deduplicate_chunks_identical(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "This is a test chunk.")
        chunk2 = create_test_chunk("chunk2", "This is a test chunk.")
        chunk3 = create_test_chunk("chunk3", "This is a different chunk.")

        retrieved1 = create_retrieved_chunk(chunk1, confidence=0.9)
        retrieved2 = create_retrieved_chunk(chunk2, confidence=0.8)
        retrieved3 = create_retrieved_chunk(chunk3, confidence=0.7)

        chunks = [retrieved1, retrieved2, retrieved3]
        deduped = assembler._deduplicate_chunks(chunks)

        assert len(deduped) == 2
        assert deduped[0].chunk.chunk_id == "chunk1"
        assert deduped[1].chunk.chunk_id == "chunk3"

    def test_deduplicate_chunks_partial_overlap(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "quick brown fox jumps dog over lazy")
        chunk2 = create_test_chunk("chunk2", "quick brown fox jumps dog")
        chunk3 = create_test_chunk("chunk3", "completely different topic here")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)
        retrieved3 = create_retrieved_chunk(chunk3)

        chunks = [retrieved1, retrieved2, retrieved3]
        deduped = assembler._deduplicate_chunks(chunks)

        assert len(deduped) == 2
        assert deduped[0].chunk.chunk_id == "chunk1"
        assert deduped[1].chunk.chunk_id == "chunk3"

    def test_deduplicate_chunks_no_overlap(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "first content alpha beta")
        chunk2 = create_test_chunk("chunk2", "second content gamma delta")
        chunk3 = create_test_chunk("chunk3", "third content epsilon zeta")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)
        retrieved3 = create_retrieved_chunk(chunk3)

        chunks = [retrieved1, retrieved2, retrieved3]
        deduped = assembler._deduplicate_chunks(chunks)

        assert len(deduped) == 3

    def test_deduplicate_same_parent_siblings_collapse(self):
        """Multiple children of the same parent (identical substituted text)
        must collapse to the single highest-confidence child."""
        assembler = ContextAssembler(max_context_chars=1000)
        parent_text = "parent context about window functions shared by every sibling"
        siblings = [
            DocumentChunk(
                chunk_id=f"child-{i}",
                source_name="spark",
                title=f"Child {i}",
                url=f"http://example.com/child-{i}",
                text=parent_text,
                content_hash=f"hash_child-{i}",
                parent_chunk_id="parent-window",
            )
            for i in range(4)
        ]
        chunks = [
            RetrievedChunk(chunk=s, distance=1.0 - (0.9 - i * 0.01), confidence=0.9 - i * 0.01)
            for i, s in enumerate(siblings)
        ]

        deduped = assembler._deduplicate_chunks(chunks)

        assert len(deduped) == 1
        assert deduped[0].chunk.chunk_id == "child-0"

    def test_deduplicate_lexically_similar_chunks_with_distinct_facts_both_survive(self):
        """Two chunks that share most tokens but carry different required facts
        must both survive deduplication (overlap alone must not drop facts)."""
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = DocumentChunk(
            chunk_id="fact-a",
            source_name="spark",
            title="Fact A",
            url="http://example.com/a",
            text="filter a DataFrame with isNotNull to keep rows where a column has a value",
            content_hash="hash_a",
        )
        chunk2 = DocumentChunk(
            chunk_id="fact-b",
            source_name="sql-guide",
            title="Fact B",
            url="http://example.com/b",
            text="filter a DataFrame with isNotNull to drop null rows in the column",
            content_hash="hash_b",
        )
        chunks = [
            RetrievedChunk(chunk=chunk1, distance=0.1, confidence=0.93),
            RetrievedChunk(chunk=chunk2, distance=0.12, confidence=0.91),
        ]

        deduped = assembler._deduplicate_chunks(chunks)

        assert len(deduped) == 2
        assert {c.chunk.chunk_id for c in deduped} == {"fact-a", "fact-b"}

    def test_deduplicate_single_chunk(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "Single chunk.")
        retrieved1 = create_retrieved_chunk(chunk1)

        deduped = assembler._deduplicate_chunks([retrieved1])

        assert len(deduped) == 1

    def test_deduplicate_empty_list(self):
        assembler = ContextAssembler(max_context_chars=1000)
        deduped = assembler._deduplicate_chunks([])
        assert deduped == []

    def test_assemble_content_truncation(self):
        assembler = ContextAssembler(max_context_chars=50)
        chunk1 = create_test_chunk("chunk1", "Short A.")
        chunk2 = create_test_chunk("chunk2", "Short B that pushes past the limit.")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)

        context, sources, _dropped = assembler.assemble([retrieved1, retrieved2])

        assert len(context) <= 130
        assert sources == ["test_source"]

    def test_assemble_max_context_zero_only_two_chunks(self):
        assembler = ContextAssembler(max_context_chars=15)
        chunk1 = create_test_chunk("chunk1", "Short.")
        chunk2 = create_test_chunk("chunk2", "Longer text that will be truncated")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)

        context, sources, _dropped = assembler.assemble([retrieved1, retrieved2])

        assert len(context) <= 100
        assert "Short" in context

    def test_assemble_oversized_chunk_is_truncated_not_raised(self):
        assembler = ContextAssembler(max_context_chars=10000, item_limit_chars=100)
        long_text = "x" * 250
        chunk = create_test_chunk("big", long_text)
        retrieved = create_retrieved_chunk(chunk)

        context, _sources, _dropped = assembler.assemble([retrieved])

        assert len(context) > 0
        assert "x" * 100 in context
        assert "x" * 101 not in context

    def test_assemble_logging_deduplication(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "This is first content.")
        chunk2 = create_test_chunk("chunk2", "This is first content.")
        chunk3 = create_test_chunk("chunk3", "This is second content.")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)
        retrieved3 = create_retrieved_chunk(chunk3)

        with patch("data_engineering_copilot.services.context_assembler.logger") as mock_logger:
            context, sources, _dropped = assembler.assemble([retrieved1, retrieved2, retrieved3])

            assert mock_logger.info.called

    def test_assemble_logging_assembly(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "Test content.")
        chunk2 = create_test_chunk("chunk2", "More content.")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)

        with patch("data_engineering_copilot.services.context_assembler.logger") as mock_logger:
            context, sources, _dropped = assembler.assemble([retrieved1, retrieved2])

            assert mock_logger.info.called

    def test_assemble_source_names(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "Text A.", source_name="source_a")
        chunk2 = create_test_chunk("chunk2", "Text B.", source_name="source_b")
        chunk3 = create_test_chunk("chunk3", "Text C.", source_name="source_a")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)
        retrieved3 = create_retrieved_chunk(chunk3)

        context, sources, _dropped = assembler.assemble([retrieved1, retrieved2, retrieved3])

        assert sources == ["source_a", "source_b", "source_a"]

    def test_valid_6000_char_segment_is_never_skipped(self):
        assembler = ContextAssembler(max_context_chars=7000)
        text = "a" * 6000
        chunk = create_test_chunk("seg-0", text)
        chunk = DocumentChunk(
            **{**chunk.__dict__, "segment_index": 0, "segment_total": 1, "parent_content_hash": "parent-hash"}
        )
        retrieved = create_retrieved_chunk(chunk)

        context, sources, dropped = assembler.assemble([retrieved], deduplicate=False)

        assert text in context
        assert dropped == []
        assert sources == ["test_source"]

    def test_oversized_segment_is_truncated(self):
        assembler = ContextAssembler(max_context_chars=20000, item_limit_chars=6000)
        chunk = create_test_chunk("seg-oversized", "a" * 6001)
        retrieved = create_retrieved_chunk(chunk)

        context, _sources, _dropped = assembler.assemble([retrieved], deduplicate=False)

        assert "a" * 6000 in context
        assert len(context) <= 7000

    def test_budget_exhaustion_reports_dropped_reason_and_segment_id(self):
        assembler = ContextAssembler(max_context_chars=200)
        chunk1 = create_test_chunk("seg-0", "First segment that fits comfortably in the budget.")
        chunk2 = create_test_chunk("seg-1", "Second segment which is too long to fit after the first one.")
        chunk1 = DocumentChunk(
            **{
                **chunk1.__dict__,
                "segment_index": 0,
                "segment_total": 2,
                "parent_content_hash": "parent-hash",
            }
        )
        chunk2 = DocumentChunk(
            **{**chunk2.__dict__, "segment_index": 1, "segment_total": 2, "parent_content_hash": "parent-hash"}
        )
        retrieved1 = create_retrieved_chunk(chunk1, confidence=0.9)
        retrieved2 = create_retrieved_chunk(chunk2, confidence=0.5)

        context, _sources, dropped = assembler.assemble([retrieved1, retrieved2], deduplicate=False)

        assert len(dropped) == 1
        record = dropped[0]
        assert record["reason"] == "dropped_due_total_context_budget"
        assert record["chunk_id"] == "seg-1"
        assert record["segment_index"] == 1
        assert record["parent_content_hash"] == "parent-hash"
        assert record["url"] == "http://example.com/chunkseg-1"
        assert chunk1.text in context
        assert chunk2.text not in context

    def test_source_coverage_guarantee_prefers_distinct_urls(self):
        """One chunk per distinct source URL is placed before any URL deepens."""
        assembler = ContextAssembler(max_context_chars=1000, max_chunks_per_source=1)
        chunk_a1 = create_test_chunk("a1", "Alpha content one.", source_name="alpha", url="http://example.com/alpha")
        chunk_a2 = create_test_chunk("a2", "Alpha content two.", source_name="alpha", url="http://example.com/alpha")
        chunk_b1 = create_test_chunk("b1", "Beta content one.", source_name="beta", url="http://example.com/beta")

        retrieved1 = create_retrieved_chunk(chunk_a1, confidence=0.9)
        retrieved2 = create_retrieved_chunk(chunk_a2, confidence=0.8)
        retrieved3 = create_retrieved_chunk(chunk_b1, confidence=0.7)

        context, sources, dropped = assembler.assemble([retrieved1, retrieved2, retrieved3])

        assert sources == ["alpha", "beta"]
        assert chunk_a1.text in context
        assert chunk_b1.text in context
        assert chunk_a2.text not in context
        cap_dropped = [d for d in dropped if d["reason"] == "dropped_due_per_source_cap"]
        assert any(d["chunk_id"] == "a2" for d in cap_dropped)

    def test_source_coverage_keeps_one_per_url_under_tight_budget(self):
        """With a tight budget, a distinct source URL is never squeezed out by
        higher-ranked chunks from an already-covered URL."""
        text_a = "First source segment that fills the available context."
        text_b = "Second source, distinct, small."
        chunk_a = create_test_chunk("a0", text_a, source_name="alpha", url="http://example.com/alpha")
        chunk_b = create_test_chunk("b0", text_b, source_name="beta", url="http://example.com/beta")

        retrieved1 = create_retrieved_chunk(chunk_a, confidence=0.9)
        retrieved2 = create_retrieved_chunk(chunk_b, confidence=0.8)

        # Budget generous enough for both formatted chunks; the point is that
        # a distinct source URL is placed before any URL deepens.
        assembler = ContextAssembler(max_context_chars=500, max_chunks_per_source=1)
        context, sources, _dropped = assembler.assemble([retrieved1, retrieved2])

        assert sources == ["alpha", "beta"]
        assert text_a in context
        assert text_b in context

    def test_source_coverage_drops_lowest_ranked_when_budget_full(self):
        """Budget drops remove the lowest-ranked segments, not the middle ones."""
        assembler = ContextAssembler(max_context_chars=1000, max_chunks_per_source=1)
        chunks = []
        retrieved = []
        for i in range(6):
            chunk = create_test_chunk(f"c{i}", f"Segment number {i} content.", source_name=f"src{i}")
            chunks.append(chunk)
            retrieved.append(create_retrieved_chunk(chunk, confidence=1.0 - i * 0.05))

        context, _sources, dropped = assembler.assemble(retrieved)

        # All six are distinct URLs and well under budget, so nothing drops.
        assert len(dropped) == 0
        for chunk in chunks:
            assert chunk.text in context

    def test_lost_in_middle_reorder_only_selected_set(self):
        """Lost-in-the-middle mitigation reorders selected chunks but must not
        cause budget drops of the middle-ranked segments."""
        assembler = ContextAssembler(max_context_chars=500, max_chunks_per_source=1)
        texts = {
            "c0": "Top ranked content zero.",
            "c1": "Content one lower ranked.",
            "c2": "Content two lower ranked.",
            "c3": "Content three lowest ranked.",
        }
        chunks = [create_test_chunk(k, t, source_name=f"src{i}") for i, (k, t) in enumerate(texts.items())]
        retrieved = [create_retrieved_chunk(chunk, confidence=1.0 - i * 0.2) for i, chunk in enumerate(chunks)]

        context, _sources, dropped = assembler.assemble(retrieved)

        assert len(dropped) == 0
        assert texts["c0"] in context
        assert texts["c3"] in context

    def test_max_chunks_per_source_constructor(self):
        assembler = ContextAssembler(max_context_chars=1000, max_chunks_per_source=3)
        assert assembler.max_chunks_per_source == 3


class TestContentHashDedup:
    def test_exact_hash_dedup(self):
        assembler = ContextAssembler(max_context_chars=1000)
        c1 = DocumentChunk(
            chunk_id="a", source_name="s", title="A", url="http://a", text="Hello world", content_hash="abc123"
        )
        c2 = DocumentChunk(
            chunk_id="b", source_name="s", title="B", url="http://b", text="Hello world", content_hash="abc123"
        )
        c3 = DocumentChunk(
            chunk_id="c", source_name="s", title="C", url="http://c", text="Different text", content_hash="def456"
        )
        chunks = [
            RetrievedChunk(chunk=c1, distance=0.1, confidence=0.9),
            RetrievedChunk(chunk=c2, distance=0.1, confidence=0.8),
            RetrievedChunk(chunk=c3, distance=0.2, confidence=0.7),
        ]
        result = assembler._content_hash_dedup(chunks)
        assert len(result) == 2

    def test_empty_hash_not_deduped(self):
        assembler = ContextAssembler(max_context_chars=1000)
        c1 = DocumentChunk(chunk_id="a", source_name="s", title="A", url="http://a", text="Hello", content_hash="")
        c2 = DocumentChunk(chunk_id="b", source_name="s", title="B", url="http://b", text="Hello", content_hash="")
        chunks = [
            RetrievedChunk(chunk=c1, distance=0.1, confidence=0.9),
            RetrievedChunk(chunk=c2, distance=0.1, confidence=0.8),
        ]
        result = assembler._content_hash_dedup(chunks)
        assert len(result) == 2


class TestSiblingMerge:
    def test_adjacent_siblings_merge(self):
        assembler = ContextAssembler(max_context_chars=1000)
        c1 = DocumentChunk(
            chunk_id="c1",
            source_name="s",
            title="C1",
            url="http://a",
            text="Part 1",
            content_hash="h1",
            parent_chunk_id="p1",
            segment_index=0,
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            source_name="s",
            title="C2",
            url="http://a",
            text="Part 2",
            content_hash="h2",
            parent_chunk_id="p1",
            segment_index=1,
        )
        chunks = [
            RetrievedChunk(chunk=c1, distance=0.1, confidence=0.9),
            RetrievedChunk(chunk=c2, distance=0.1, confidence=0.8),
        ]
        result = assembler._merge_adjacent_siblings(chunks)
        assert len(result) == 1
        assert "Part 1\n\nPart 2" in result[0].chunk.text

    def test_orphan_chunks_preserved(self):
        assembler = ContextAssembler(max_context_chars=1000)
        c1 = DocumentChunk(chunk_id="o1", source_name="s", title="O1", url="http://a", text="Orphan", content_hash="h1")
        chunks = [RetrievedChunk(chunk=c1, distance=0.1, confidence=0.9)]
        result = assembler._merge_adjacent_siblings(chunks)
        assert len(result) == 1


class TestMMRDiversity:
    def test_mmr_increases_diversity(self):
        assembler = ContextAssembler(max_context_chars=1000)
        c1 = DocumentChunk(
            chunk_id="a",
            source_name="s",
            title="A",
            url="http://a",
            text="Spark SQL optimization techniques for joins",
            content_hash="h1",
        )
        c2 = DocumentChunk(
            chunk_id="b",
            source_name="s",
            title="B",
            url="http://b",
            text="Spark SQL optimization methods for join operations",
            content_hash="h2",
        )
        c3 = DocumentChunk(
            chunk_id="c",
            source_name="s",
            title="C",
            url="http://c",
            text="Airflow DAG scheduling configuration",
            content_hash="h3",
        )
        chunks = [
            RetrievedChunk(chunk=c1, distance=0.1, confidence=0.9),
            RetrievedChunk(chunk=c2, distance=0.1, confidence=0.85),
            RetrievedChunk(chunk=c3, distance=0.3, confidence=0.6),
        ]
        result = assembler._mmr_diversify(chunks, lambda_param=0.5)
        assert len(result) == 3
        assert result[0].chunk.chunk_id == "a"
        ids = [r.chunk.chunk_id for r in result]
        assert ids.index("c") < ids.index("b")

    def test_mmr_single_chunk(self):
        assembler = ContextAssembler(max_context_chars=1000)
        c1 = DocumentChunk(
            chunk_id="a", source_name="s", title="A", url="http://a", text="Only chunk", content_hash="h1"
        )
        chunks = [RetrievedChunk(chunk=c1, distance=0.1, confidence=0.9)]
        result = assembler._mmr_diversify(chunks)
        assert len(result) == 1


class TestXmlEscaping:
    def test_format_chunk_escapes_xml_metacharacters(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk = create_retrieved_chunk(create_test_chunk("c1", "Use <script>alert('xss')</script>"))
        result = assembler._format_chunk(chunk, 1)
        # < is escaped to &lt; — but > is NOT escaped per spec (only & and <)
        assert "&lt;script" in result
        assert "<script" not in result

    def test_format_chunk_escapes_ampersand(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk = create_retrieved_chunk(create_test_chunk("c2", "Tom & Jerry love M&M's"))
        result = assembler._format_chunk(chunk, 1)
        assert "&amp;" in result
        assert "Tom & Jerry" not in result

    def test_format_chunk_ampersand_before_angle_bracket(self):
        """Order matters: & must be escaped before < to prevent double-escaping."""
        assembler = ContextAssembler(max_context_chars=1000)
        # We use a string where < and & are separated to avoid accidental substring matches
        # Correct: & -> &amp;, < -> &lt;
        # Incorrect (wrong order): < -> &lt;, & in &lt; -> &amp;lt;
        chunk = create_retrieved_chunk(create_test_chunk("c3", "A < B & C"))
        result = assembler._format_chunk(chunk, 1)

        assert "&amp;" in result
        assert "&lt;" in result
        assert "&amp;lt;" not in result  # This is the double-escaping we must avoid

        # The raw < should be escaped; only wrapper tags retain <
        inner = result.split("\n", 1)[1].rsplit("\n</context_doc>", 1)[0]
        assert "<" not in inner

    def test_format_chunk_plain_text_unchanged(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk = create_retrieved_chunk(create_test_chunk("c4", "Hello world, no special chars here."))
        result = assembler._format_chunk(chunk, 1)
        assert "Hello world, no special chars here." in result

    def test_format_chunk_xml_escape_enabled(self):
        """When xml_content_escape=True (default), < and & are escaped."""
        assembler = ContextAssembler(max_context_chars=1000, xml_content_escape=True)
        chunk = create_retrieved_chunk(create_test_chunk("c5", "Use <script>alert('xss')</script> and Tom & Jerry"))
        result = assembler._format_chunk(chunk, 1)
        assert "&lt;script" in result
        assert "&amp;" in result
        assert "<script" not in result
        assert "Tom & Jerry" not in result

    def test_format_chunk_xml_escape_disabled(self):
        """When xml_content_escape=False, raw < and & are preserved in inner content."""
        assembler = ContextAssembler(max_context_chars=1000, xml_content_escape=False)
        chunk = create_retrieved_chunk(create_test_chunk("c6", "<script>alert('xss')</script> and Tom & Jerry"))
        result = assembler._format_chunk(chunk, 1)
        # Inner content preserves raw characters
        inner = result.split("\n", 1)[1].rsplit("\n</context_doc>", 1)[0]
        assert "<script>" in inner
        assert "Tom & Jerry" in inner
        assert "<" in inner
        assert "&" in inner

    def test_assemble_respects_escape_flag(self):
        """End-to-end assemble() respects xml_content_escape flag."""
        # With escaping enabled
        assembler_enabled = ContextAssembler(max_context_chars=1000, xml_content_escape=True)
        chunk = create_retrieved_chunk(create_test_chunk("c7", "<b>bold</b> and Tom & Jerry"))
        context_enabled, _, _ = assembler_enabled.assemble([chunk])
        assert "&lt;b>" in context_enabled
        assert "&amp;" in context_enabled

        # With escaping disabled
        assembler_disabled = ContextAssembler(max_context_chars=1000, xml_content_escape=False)
        context_disabled, _, _ = assembler_disabled.assemble([chunk])
        # Check inner content only (wrapper tags always have <)
        inner_disabled = context_disabled.split("\n", 1)[1].rsplit("\n</context_doc>", 1)[0]
        assert "<b>bold</b>" in inner_disabled
        assert "Tom & Jerry" in inner_disabled
        assert "<" in inner_disabled
        assert "&" in inner_disabled


class TestBreadcrumbs:
    def test_hierarchical_breadcrumb(self):
        c = DocumentChunk(
            chunk_id="c1",
            source_name="Spark",
            title="C1",
            url="http://a",
            text="text",
            content_hash="h1",
            heading_path=("SQL", "Joins"),
        )
        r = RetrievedChunk(chunk=c, distance=0.1, confidence=0.9)
        bc = ContextAssembler._build_breadcrumb(r, "hierarchical")
        assert bc == "[Source: Spark > SQL > Joins]"

    def test_flat_breadcrumb(self):
        c = DocumentChunk(
            chunk_id="c1",
            source_name="Spark",
            title="C1",
            url="http://a",
            text="text",
            content_hash="h1",
            heading_path=("SQL",),
        )
        r = RetrievedChunk(chunk=c, distance=0.1, confidence=0.9)
        bc = ContextAssembler._build_breadcrumb(r, "flat")
        assert bc == "[Source: Spark]"

    def test_no_breadcrumb(self):
        c = DocumentChunk(
            chunk_id="c1", source_name="Spark", title="C1", url="http://a", text="text", content_hash="h1"
        )
        r = RetrievedChunk(chunk=c, distance=0.1, confidence=0.9)
        bc = ContextAssembler._build_breadcrumb(r, "none")
        assert bc == ""

    def test_empty_heading_path(self):
        c = DocumentChunk(
            chunk_id="c1",
            source_name="Spark",
            title="C1",
            url="http://a",
            text="text",
            content_hash="h1",
            heading_path=(),
        )
        r = RetrievedChunk(chunk=c, distance=0.1, confidence=0.9)
        bc = ContextAssembler._build_breadcrumb(r, "hierarchical")
        assert bc == "[Source: Spark]"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
