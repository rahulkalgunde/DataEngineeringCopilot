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
        chunks = chunker._sync_chunk(_doc(md))
        # "Intro" (level 1) is separate from "Getting Started" + "Configuration" (level 2 under Intro)
        # Getting Started and Configuration share parent ("Intro") so they merge
        assert len(chunks) >= 2
        headers = [c.section_header for c in chunks]
        assert "Intro" in headers
        # The second chunk has the last merged section's header
        assert any(h in headers for h in ["Getting Started", "Configuration"])

    def test_heading_path_tracking(self):
        md = "# Top\nIntro text.\n\n## Middle\nMiddle text.\n\n### Bottom\nBottom text.\n"
        chunker = HeaderAwareChunker(chunk_size_words=100, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(md))
        # All three are nested: Top > Middle > Bottom — they share a path chain
        # so they may merge into fewer chunks. Check heading_path is populated.
        assert all(c.heading_path for c in chunks)

    def test_code_blocks_preserved(self):
        md = "## Example\nText before code.\n\n```python\ndef foo():\n    pass\n```\n\nText after code.\n"
        chunker = HeaderAwareChunker(chunk_size_words=200, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(md))
        assert len(chunks) >= 1
        assert any("foo" in c.text for c in chunks)

    def test_chunk_type_set(self):
        md = "## Section\nSome text content here.\n"
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(md))
        assert all(c.chunk_type == "text" for c in chunks)

    def test_word_count_populated(self):
        md = "## Section\nThis section has several words for testing.\n"
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(md))
        assert all(c.word_count > 0 for c in chunks)

    def test_empty_document(self):
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(""))
        assert chunks == []

    def test_no_headers_returns_empty(self):
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(""))
        assert chunks == []

    def test_heading_less_content_is_chunked(self):
        # Regression: documents with no ``#`` markdown headings (table-heavy
        # reference pages, prose-only docs) were silently dropped. They must be
        # split into paragraph-sized chunks instead.
        text = (
            "Spark SQL supports operating on a variety of data sources through the DataFrame interface.\n\n"
            "Registering a DataFrame as a temporary view allows you to run SQL queries over its data."
        )
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(text))
        assert chunks, "heading-less content must produce chunks, not be dropped"
        joined = " ".join(c.text for c in chunks)
        assert "DataFrame interface" in joined
        assert "temporary view" in joined

    def test_redirect_stub_stays_no_content(self):
        # Nav-only stubs (moved / under-construction notices) keep yielding
        # zero chunks so coverage marks them ``no_content``.
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc("This document has moved [here](rdd-programming-guide.html)."))
        assert chunks == []

    def test_min_chunk_words_filter(self):
        md = "## A\nShort.\n\n## B\nAnother short.\n"
        chunker = HeaderAwareChunker(chunk_size_words=100, overlap_words=10, min_chunk_words=50)
        chunks = chunker._sync_chunk(_doc(md))
        # Both sections are too short individually and don't merge across parent boundaries
        # So we might get 0 or 1 chunk depending on merge behavior
        assert len(chunks) <= 1

    def test_metadata_propagated_from_document(self):
        md = "# Title\nSome intro text here with enough words to produce a chunk.\n"
        doc = ParsedDocument(
            source_name="Claude Platform Docs",
            title="Overview",
            url="https://platform.claude.com/docs/en/api/overview.md",
            text=md,
            doc_type="api_reference",
            file_path="api/overview.md",
        )
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(doc)
        assert chunks
        for chunk in chunks:
            assert chunk.doc_type == "api_reference"
            assert chunk.file_path == "api/overview.md"
            assert chunk.source_name == "Claude Platform Docs"

    def test_small_nested_sections_not_dropped(self):
        # Short API-reference pages use nested headings (## -> ### -> ####) with
        # a couple of lines under each. The parent-boundary flush used to discard
        # any accumulation below min_chunk_words, so a 77-word page produced zero
        # chunks. Sub-minimum content must be carried forward, not dropped.
        md = (
            "## Delete External Key\n"
            "Delete an external key config by its ID.\n\n"
            "### Path Parameters\n"
            "- `external_key_id: string`\n\n"
            "### Returns\n"
            "- `id: string`\n"
            '- `type: "external_key_deleted"`\n\n'
            "### Example\n"
            "```http\n"
            "curl -X DELETE ...\n"
            "```\n\n"
            "#### Response\n"
            "```json\n"
            '{ "id": "ekey_01AbCd" }\n'
            "```\n"
        )
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=20)
        chunks = chunker._sync_chunk(_doc(md))
        assert chunks, "sub-minimum nested sections must merge into a chunk, not be dropped"
        joined = " ".join(c.text for c in chunks)
        assert "Delete an external key config" in joined
        assert "ekey_01AbCd" in joined

    def test_parent_boundary_flush_still_respected_above_minimum(self):
        # A level-1 section followed by a level-2 section (parent/child
        # transition) with enough words must still flush at the boundary, so
        # the top-level section keeps its own chunk.
        md = (
            "# Intro\n"
            "Some intro text here with enough words to be a real section.\n\n"
            "## Config\n"
            "A child section with plenty of words of its own.\n"
        )
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(md))
        headers = {c.section_header for c in chunks}
        assert "Intro" in headers
        assert "Config" in headers
