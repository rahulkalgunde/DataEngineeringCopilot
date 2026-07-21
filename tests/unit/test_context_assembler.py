"""Tests for the ContextAssembler service."""

from __future__ import annotations

from unittest.mock import patch

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.context_assembler import ContextAssembler


def create_test_chunk(chunk_id, text, source_name="test_source"):
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name=source_name,
        title=f"Title {chunk_id}",
        url=f"http://example.com/chunk{chunk_id}",
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
        context, sources = assembler.assemble([])
        assert context == ""
        assert sources == []

    def test_assemble_single_chunk(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk = create_test_chunk("chunk1", "This is a single test chunk.")
        retrieved = create_retrieved_chunk(chunk)

        context, sources = assembler.assemble([retrieved])

        expected_context = "[test_source] This is a single test chunk."
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

        context, sources = assembler.assemble([retrieved1, retrieved2, retrieved3])

        expected_context = (
            "[test_source] First chunk of text.\n"
            "[test_source] Second chunk of text.\n"
            "[test_source] Third chunk of text."
        )
        assert context == expected_context
        assert sources == ["test_source", "test_source", "test_source"]

    def test_assemble_with_different_sources(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "First test chunk.", source_name="source_a")
        chunk2 = create_test_chunk("chunk2", "Second test chunk.", source_name="source_b")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)

        context, sources = assembler.assemble([retrieved1, retrieved2])

        expected_context = "[source_a] First test chunk.\n[source_b] Second test chunk."
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

        context, sources = assembler.assemble([retrieved1, retrieved2])

        assert len(context) <= 60
        assert sources == ["test_source"]

    def test_assemble_max_context_zero_only_two_chunks(self):
        assembler = ContextAssembler(max_context_chars=15)
        chunk1 = create_test_chunk("chunk1", "Short.")
        chunk2 = create_test_chunk("chunk2", "Longer text that will be truncated")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)

        context, sources = assembler.assemble([retrieved1, retrieved2])

        assert len(context) <= 30
        assert "Short" in context

    def test_assemble_logging_deduplication(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "This is first content.")
        chunk2 = create_test_chunk("chunk2", "This is first content.")
        chunk3 = create_test_chunk("chunk3", "This is second content.")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)
        retrieved3 = create_retrieved_chunk(chunk3)

        with patch("data_engineering_copilot.services.context_assembler.logger") as mock_logger:
            context, sources = assembler.assemble([retrieved1, retrieved2, retrieved3])

            assert mock_logger.info.called

    def test_assemble_logging_assembly(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "Test content.")
        chunk2 = create_test_chunk("chunk2", "More content.")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)

        with patch("data_engineering_copilot.services.context_assembler.logger") as mock_logger:
            context, sources = assembler.assemble([retrieved1, retrieved2])

            assert mock_logger.info.called

    def test_assemble_source_names(self):
        assembler = ContextAssembler(max_context_chars=1000)
        chunk1 = create_test_chunk("chunk1", "Text A.", source_name="source_a")
        chunk2 = create_test_chunk("chunk2", "Text B.", source_name="source_b")
        chunk3 = create_test_chunk("chunk3", "Text C.", source_name="source_a")

        retrieved1 = create_retrieved_chunk(chunk1)
        retrieved2 = create_retrieved_chunk(chunk2)
        retrieved3 = create_retrieved_chunk(chunk3)

        context, sources = assembler.assemble([retrieved1, retrieved2, retrieved3])

        assert sources == ["source_a", "source_b", "source_a"]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
