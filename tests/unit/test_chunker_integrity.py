"""Chunker-integrity regression tests (Task 5 / 6.2 of the chunking plan).

Covers parent-recombination invariants (H2), front-matter offset bookkeeping
(H1), fence-atomicity + re-fencing (M12), and the broadened fence grammar
(tilde / indented / CRLF, H9).
"""

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.services.chunker import deduplicate_chunks
from data_engineering_copilot.services.header_aware_chunker import (
    _FENCE_RE,
    HeaderAwareChunker,
)


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source_name="Spark", title="Test", url="http://x", text=text)


def test_overflow_window_offsets_monotonic_in_bounds():
    text = "# Big\n\n" + "word " * 900
    chunker = HeaderAwareChunker(chunk_size_words=200, overlap_words=20, min_chunk_words=10)
    chunks = chunker._sync_chunk(_doc(text))
    assert len(chunks) > 1
    starts = [c.start_offset for c in chunks]
    assert starts == sorted(starts)
    for c in chunks:
        assert 0 <= c.start_offset < c.end_offset <= len(text)


def test_nested_sections_never_merge_across_parents_above_minimum():
    text = (
        "# A\n" + "alpha " * 40 + "\n\n"
        "## A1\n" + "one " * 40 + "\n\n"
        "## A2\n" + "two " * 40 + "\n\n"
        "# B\n" + "beta " * 40 + "\n\n"
        "## B1\n" + "bee " * 40
    )
    chunker = HeaderAwareChunker(chunk_size_words=55, overlap_words=0, min_chunk_words=10)
    chunks = chunker._sync_chunk(_doc(text))
    assert len(chunks) >= 5
    for ch in chunks:
        topics = {
            topic
            for word, topic in [("alpha", "A"), ("one", "A"), ("two", "A"), ("beta", "B"), ("bee", "B")]
            if word in ch.text
        }
        assert len(topics) <= 1, f"cross-parent merge: {ch.text[:60]!r}"


def test_parent_subtree_overflow_drops_no_child():
    text = (
        "# Cover\n" + "cover " * 20 + "\n\n"
        "## Child1\n" + "one " * 30 + "\n\n"
        "## Child2\n" + "two " * 30 + "\n\n"
        "## Child3\n" + "three " * 30
    )
    chunker = HeaderAwareChunker(chunk_size_words=25, overlap_words=0, min_chunk_words=5)
    chunks = chunker._sync_chunk(_doc(text))
    joined = " ".join(c.text for c in chunks)
    for word in ("one", "two", "three"):
        assert word in joined, f"child content dropped: {word}"


def test_sibling_sections_may_merge_under_budget():
    text = "# A\n" + "alpha " * 15 + "\n\n# B\n" + "beta " * 15
    chunker = HeaderAwareChunker(chunk_size_words=500, overlap_words=0, min_chunk_words=3)
    chunks = chunker._sync_chunk(_doc(text))
    assert chunks
    assert len(chunks) == 1


def test_frontmatter_offsets_relative_to_original():
    text = "---\ntitle: Spark Docs\n---\n# Title\nIntro text here."
    chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
    chunks = chunker._sync_chunk(_doc(text))
    assert chunks
    for c in chunks:
        assert 0 <= c.start_offset <= c.end_offset <= len(text)
        # chunks are exact substrings of the ORIGINAL doc (incl. front-matter)
        assert c.text == text[c.start_offset : c.end_offset], "front-matter shifted offsets"


def test_split_code_chunk_refences_balanced():
    defs = "".join(f"def f{i}(x):\n    return x * {i}\n\n" for i in range(30))
    text = "# Big code\n\n```python\n" + defs + "```"
    chunker = HeaderAwareChunker(chunk_size_words=30, overlap_words=0, min_chunk_words=3)
    chunks = chunker._sync_chunk(_doc(text))
    fenced = [c for c in chunks if "```" in c.text]
    assert fenced, "oversized fence should wind up in chunks"
    for c in fenced:
        assert c.text.count("```") % 2 == 0, "unbalanced fence (dangling opener)"
        assert c.text.lstrip().startswith("```python"), "fence opener lost"
        assert c.text.rstrip().endswith("```"), "fence closer lost"


def test_tilde_and_indented_fences_are_code_not_sections():
    text = (
        "~~~python\n"
        "# comment must NOT start a section\n"
        "def foo():\n"
        "    pass\n"
        "~~~\n\n"
        "  ```\n"
        "# indented fence comment must NOT start a section\n"
        "x = 1\n"
        "  ```\n\n"
        "## Real Section\n"
        "prose here.\n"
    )
    chunker = HeaderAwareChunker(chunk_size_words=200, overlap_words=0, min_chunk_words=1)
    sections = chunker._split_into_sections(text)
    headers = [s.header for s in sections]
    assert headers == ["", "Real Section"], f"in-fence # comments created sections: {headers}"
    preamble = sections[0].text
    assert "~~~python" in preamble and "# comment must NOT start a section" in preamble
    assert "  ```" in preamble and "x = 1" in preamble
    assert all(h for h in headers if h)  # no spurious in-fence headers


def test_crlf_fenced_block_detected():
    text = "## Section\r\n\r\n```python\r\ndef foo():\r\n    return 1\r\n```\r\n\r\nmore text.\r\n"
    chunker = HeaderAwareChunker(chunk_size_words=100, overlap_words=0, min_chunk_words=1)
    sections = chunker._split_into_sections(text)
    headers = [s.header for s in sections]
    assert headers == ["Section"]
    assert sections[0].code_blocks, "CRLF fenced block not recognized"


def test_overlap_suffix_in_both_neighbors():
    text = "# A\n" + "word " * 140 + "\n\n# B\n" + "beta " * 60
    chunker = HeaderAwareChunker(chunk_size_words=100, overlap_words=8, min_chunk_words=10)
    chunks = chunker._sync_chunk(_doc(text))
    assert len(chunks) >= 2
    prev, cur = chunks[0], chunks[1]
    tail_prev = " ".join(prev.text.split()[-8:])
    assert tail_prev in cur.text, "overlap tail missing from the next chunk"


def test_direct_fence_regex_matches_corpus_grammar():
    cases = [
        "```python\ncode\n```",
        "~~~\ncode\n~~~",
        "  ```js\ncode\n  ```",
        "```\r\ncode\r\n```\r\n",
    ]
    for case in cases:
        assert _FENCE_RE.search(case), f"fence grammar missed: {case!r}"


def _chunk(text: str, heading_path: tuple[str, ...] = (), header: str = "") -> DocumentChunk:
    return DocumentChunk(
        chunk_id="x",
        source_name="s",
        title="t",
        url="http://x",
        text=text,
        section_header=header,
        heading_path=heading_path,
    )


def test_dedup_keeps_same_body_under_different_headings():
    # H6: identical body under two headings is NOT a duplicate.
    a = _chunk("same body text under heading", heading_path=("A",), header="A")
    b = _chunk("same body text under heading", heading_path=("B",), header="B")
    c = _chunk("same body text under heading", heading_path=("A",), header="A")
    result = deduplicate_chunks([a, b, c])
    assert len(result) == 2


def test_dedup_collapses_true_duplicates():
    a = _chunk("the quick brown fox")
    b = _chunk("The    quick brown fox\n")
    result = deduplicate_chunks([a, b])
    assert len(result) == 1


def test_empty_heading_segment_is_skipped_in_path():
    # L2: heading-only lines must not smuggle empty breadcrumb segments.
    text = "# A\n\n## \n\nbody with several words here.\n"
    chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=0, min_chunk_words=3)
    chunks = chunker._sync_chunk(_doc(text))
    assert chunks
    for ch in chunks:
        assert not any(seg == "" for seg in ch.heading_path)
