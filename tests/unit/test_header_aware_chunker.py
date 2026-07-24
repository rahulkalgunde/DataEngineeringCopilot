"""Unit tests for HeaderAwareChunker."""

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source_name="Spark", title="Test", url="http://x", text=text)


class TestHeaderAwareChunker:
    def test_splits_on_headers(self):
        md = (
            "# Intro\nSome intro text here with enough words.\n\n"
            "## Getting Started\nHere is how you get started with the framework.\n\n"
            "## Configuration\nConfigure your session with the right settings.\n"
        )
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker.chunk(_doc(md))
        # "Intro" (level 1) is separate from "Getting Started" + "Configuration" (level 2 under Intro)
        # Getting Started and Configuration share parent ("Intro") so they merge
        assert len(chunks) >= 2
        headers = [c.section_header for c in chunks]
        assert "Intro" in headers
        # The second chunk has the last merged section's header
        assert any(h in headers for h in ["Getting Started", "Configuration"])

    def test_heading_path_tracking(self):
        md = (
            "# Top\nIntro text.\n\n"
            "## Middle\nMiddle text.\n\n"
            "### Bottom\nBottom text.\n"
        )
        chunker = HeaderAwareChunker(chunk_size_words=100, overlap_words=10, min_chunk_words=3)
        chunks = chunker.chunk(_doc(md))
        # All three are nested: Top > Middle > Bottom — they share a path chain
        # so they may merge into fewer chunks. Check heading_path is populated.
        assert all(c.heading_path for c in chunks)

    def test_code_blocks_preserved(self):
        md = (
            "## Example\nText before code.\n\n"
            "```python\ndef foo():\n    pass\n```\n\n"
            "Text after code.\n"
        )
        chunker = HeaderAwareChunker(chunk_size_words=200, overlap_words=10, min_chunk_words=3)
        chunks = chunker.chunk(_doc(md))
        assert len(chunks) >= 1
        assert any("foo" in c.text for c in chunks)

    def test_chunk_type_set(self):
        md = "## Section\nSome text content here.\n"
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker.chunk(_doc(md))
        assert all(c.chunk_type == "text" for c in chunks)

    def test_word_count_populated(self):
        md = "## Section\nThis section has several words for testing.\n"
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker.chunk(_doc(md))
        assert all(c.word_count > 0 for c in chunks)

    def test_empty_document(self):
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker.chunk(_doc(""))
        assert chunks == []

    def test_no_headers_returns_empty(self):
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker.chunk(_doc("Just plain text without any headers."))
        assert chunks == []

    def test_min_chunk_words_filter(self):
        md = "## A\nShort.\n\n## B\nAnother short.\n"
        chunker = HeaderAwareChunker(chunk_size_words=100, overlap_words=10, min_chunk_words=50)
        chunks = chunker.chunk(_doc(md))
        # Both sections are too short individually and don't merge across parent boundaries
        # So we might get 0 or 1 chunk depending on merge behavior
        assert len(chunks) <= 1
