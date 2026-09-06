"""Header-aware markdown chunker.

Splits documents along Markdown headers (#, ##, ###, etc.) to preserve
topical boundaries. Each chunk retains its heading hierarchy and any
embedded code blocks.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, replace

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument

logger = logging.getLogger(__name__)

# ``.+`` (not ``.*``) is deliberate, and the separator is ``[ \t]+`` rather
# than ``\s+``: heading lines may never cross into the next line. ``## `` on
# its own line otherwise lets ``\s+`` swallow the newline and the NEXT PARAGRAPH
# becomes the heading text. Requiring same-line space + at least one heading
# char keeps empty heading markers as prose.
_HEADER_RE = re.compile(r"^(#{1,6})[ \t]+(.+)$", re.MULTILINE)
# Fenced code: 0-3 leading spaces, `` ``` `` or ``~~~`` markers, optional
# language tag on the opening line. CRLF is tolerated on both the opening and
# closing lines. Only ``m.group(0)`` is consumed by callers.
_FENCE_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})[^\r\n]*\r?\n(.*?)^\1\2[^\r\n]*", re.MULTILINE | re.DOTALL)
# HTML tables are atomic split units: prose word windows must never cut a table
# mid-row, otherwise the markdown-to-HTML rendering (and downstream extraction)
# silently drops half the rows.
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r"<tr.*?</tr>", re.DOTALL | re.IGNORECASE)
# Leading YAML frontmatter block (Jekyll ``---\n...\n---``) stripped before
# sectioning so license/title boilerplate never leaks into chunk text.
_FRONTMATTER_RE = re.compile(r"^\ufeff?---\r?\n.*?\r?\n---\r?\n?", re.DOTALL)
# Navigation-only stub markers. Heading-less pages whose *entire* body matches
# one of these redirect/moved notices carry no retrieval content and stay
# ``no_content`` (matching the ``redirect_stub`` classification).
_NAV_STUB_MARKERS = (
    "has moved",
    "has been moved",
    "moved to",
    "is now archived",
    "under construction",
    "work in progress",
    "broken apart",
    "page is moved",
)


def _inside_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    """Return True when *pos* falls within any ``(start, end)`` span."""
    return any(start <= pos < end for start, end in spans)


def _verbatim_tail(text: str, n: int) -> str:
    """Return the last *n* whitespace-separated words of *text* verbatim.

    Unlike a naive ``" ".join(text.split()[-n:])``, the slice keeps the
    original line breaks/spacing between those words, so an overlap prefix
    reproduces the source tail byte-for-byte.
    """
    spans = [m for m in re.finditer(r"\S+", text)]
    if not spans or len(spans) <= n:
        return text
    return text[spans[-n].start() :]


@dataclass
class _RawSection:
    """Intermediate representation before merging."""

    header: str
    level: int
    heading_path: tuple[str, ...]
    text: str
    code_blocks: tuple[str, ...]
    start: int = 0
    end: int = 0


class HeaderAwareChunker:
    """Chunker that splits markdown along header boundaries.

    Parameters
    ----------
    chunk_size_words:
        Target chunk size in words.  Sections smaller than this are merged
        with subsequent sections under the same parent header.
    overlap_words:
        Overlap (in words) carried from the end of one chunk into the next
        when merging sections.  Preserves continuity across boundaries.
    min_chunk_words:
        Minimum words for a chunk to be included in the output.
    prepend_heading_path:
        If True, prepend the heading path (e.g. ``pyspark.sql.functions``)
        to each chunk's text for breadcrumb context in vector embeddings.
    """

    def __init__(
        self,
        chunk_size_words: int = 500,
        overlap_words: int = 120,
        min_chunk_words: int = 10,
        prepend_heading_path: bool = False,
    ) -> None:
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if overlap_words < 0 or overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be >= 0 and < chunk_size_words")
        if min_chunk_words < 0:
            raise ValueError("min_chunk_words must be non-negative")

        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.min_chunk_words = min_chunk_words
        self.prepend_heading_path = prepend_heading_path

    async def chunk(
        self, document: ParsedDocument, precomputed_embeddings: list[list[float]] | None = None
    ) -> list[DocumentChunk]:
        """Chunk *document* by splitting on Markdown headers."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_chunk, document)

    def extract_sentences(self, text: str) -> list[str] | None:
        """Sentence pre-extraction is not supported by this chunker."""
        return None

    def _sync_chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        sections = self._split_into_sections(document.text)
        if not sections:
            return []

        chunks = self._merge_sections(sections, document)
        logger.info(
            "Header-aware chunking: source=%s url=%s title=%r sections=%d chunks=%d",
            document.source_name,
            document.url,
            document.title,
            len(sections),
            len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Section splitting
    # ------------------------------------------------------------------

    def _split_heading_less_paragraphs(self, text: str, offset_base: int = 0) -> list[_RawSection]:
        """Split heading-less *text* into paragraph-sized level-0 sections.

        Returns an empty list for blank input. Paragraphs longer than
        ``chunk_size_words`` are further split into word windows so the
        downstream merge never accumulates an oversized chunk.
        ``offset_base`` shifts reported offsets when *text* is a sub-splice
        (e.g. after front-matter stripping) so offsets stay relative to the
        original document.
        """
        body = text.strip()
        if not body:
            return []

        # Redirect / moved / under-construction notices have no retrieval
        # value and must stay ``no_content``.
        lower = body.lower()
        if len(lower.split()) <= 40 and any(marker in lower for marker in _NAV_STUB_MARKERS):
            return []

        sections: list[_RawSection] = []
        cursor = 0
        for paragraph in re.split(r"\n\s*\n", body):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            words = paragraph.split()
            if len(words) <= self.chunk_size_words:
                idx = text.find(paragraph, cursor)
                start = idx if idx != -1 else cursor
                end = start + len(paragraph)
                cursor = end
                sections.append(
                    _RawSection(
                        header="",
                        level=0,
                        heading_path=(),
                        text=paragraph,
                        code_blocks=tuple(blk.group(0) for blk in _FENCE_RE.finditer(paragraph)),
                        start=offset_base + start,
                        end=offset_base + end,
                    )
                )
                continue
            # Oversized paragraph: use code-aware splitting to avoid breaking mid-function
            text_chunks = self._split_oversized_section(paragraph, self.chunk_size_words)
            # Calculate start/end positions for each chunk
            chunk_cursor = 0
            for text_chunk in text_chunks:
                idx = text.find(text_chunk, chunk_cursor)
                chunk_start = idx if idx != -1 else chunk_cursor
                chunk_end = chunk_start + len(text_chunk)
                chunk_cursor = chunk_end
                sections.append(
                    _RawSection(
                        header="",
                        level=0,
                        heading_path=(),
                        text=text_chunk,
                        code_blocks=tuple(blk.group(0) for blk in _FENCE_RE.finditer(text_chunk)),
                        start=offset_base + chunk_start,
                        end=offset_base + chunk_end,
                    )
                )
        return sections

    def _split_into_sections(self, text: str) -> list[_RawSection]:
        """Split markdown *text* into sections at header boundaries.

        Headings that appear *inside* a fenced code block are ignored so that
        a ``# comment`` line in code never starts a spurious section.
        """
        stripped = _FRONTMATTER_RE.sub("", text, count=1)
        # Offsets are reported relative to the ORIGINAL *text*: any front-matter
        # that was stripped shifts all subsequent section spans by its length.
        offset_base = len(text) - len(stripped)
        text = stripped
        fence_spans = [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]
        matches = [m for m in _HEADER_RE.finditer(text) if not _inside_spans(m.start(), fence_spans)]
        if not matches:
            # No markdown headings (table-heavy reference pages, TOC lists,
            # prose-only docs). Split on blank-line paragraphs so
            # ``_merge_sections`` can chunk by word budget instead of
            # silently dropping the page. Oversized single paragraphs are
            # further split into word windows.
            return self._split_heading_less_paragraphs(text, offset_base)

        sections: list[_RawSection] = []

        # Content before the first header is the preamble
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(
                _RawSection(
                    header="",
                    level=0,
                    heading_path=(),
                    text=preamble,
                    code_blocks=(),
                    start=offset_base,
                    end=offset_base + len(preamble),
                )
            )

        heading_stack: list[tuple[int, str]] = []

        for i, m in enumerate(matches):
            level = len(m.group(1))
            header_text = m.group(2).strip()
            start = m.end() + 1  # skip the newline after header
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw_body = text[start:end]

            # Extract code blocks from the body
            code_blocks = tuple(blk.group(0) for blk in _FENCE_RE.finditer(raw_body))

            # Build heading path. Pop entries at or below the new heading's
            # level so a sibling (same level) replaces the previous sibling
            # instead of being nested under it (which would fabricate a false
            # parent/child relation and flush boundary later).
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, header_text))
            path = tuple(t for _, t in heading_stack if t)

            sections.append(
                _RawSection(
                    header=header_text,
                    level=level,
                    heading_path=path,
                    text=raw_body.strip(),
                    code_blocks=code_blocks,
                    start=offset_base + start,
                    end=offset_base + end,
                )
            )

        # Flaw-pattern #7 guard (context fragmentation): a headed section whose
        # body alone exceeds the word budget must be split into word windows,
        # exactly like heading-less text is — otherwise it silently becomes one
        # oversized chunk and every retrieval inside it drags the whole section
        # into context. Windows keep the section's header/heading_path so no
        # fragment loses its parent context.
        windowed: list[_RawSection] = []
        for section in sections:
            if not section.text or len(section.text.split()) <= self.chunk_size_words:
                windowed.append(section)
                continue
            # Split oversized section at code-block and function boundaries
            text_chunks = self._split_oversized_section(section.text, self.chunk_size_words)
            cursor = section.start
            for text_chunk in text_chunks:
                windowed.append(
                    _RawSection(
                        header=section.header,
                        level=section.level,
                        heading_path=section.heading_path,
                        text=text_chunk,
                        code_blocks=tuple(blk.group(0) for blk in _FENCE_RE.finditer(text_chunk)),
                        start=cursor,
                        end=cursor + len(text_chunk),
                    )
                )
                cursor += len(text_chunk)
        sections = windowed

        return sections

    def _word_count_of_text(self, text: str) -> int:
        return len(text.split())

    def _split_oversized_section(self, text: str, max_words: int) -> list[str]:
        """Split *text* (which exceeds max_words) into chunks.

        Strategy: keep fenced code blocks and HTML tables intact as atomic
        spans; split prose by word windows; for oversized code blocks, split at
        ``def``/``class``/``async def`` boundaries first, then fall back to
        line-based windows; for oversized tables, pack whole rows greedily.
        """
        spans: list[tuple[int, int, str]] = [(m.start(), m.end(), "fence") for m in _FENCE_RE.finditer(text)]
        spans.extend((m.start(), m.end(), "table") for m in _TABLE_RE.finditer(text))
        spans.sort(key=lambda s: s[0])

        result: list[str] = []
        cursor = 0

        for start, end, kind in spans:
            if start < cursor:
                # Overlaps an already-processed span (e.g. table inside a
                # fenced block) — already emitted, skip.
                continue
            if cursor < start:
                result.extend(self._split_prose_chunk(text[cursor:start], max_words))
            block = text[start:end]
            if len(block.split()) > max_words:
                if kind == "table":
                    result.extend(self._split_table_chunk(block, max_words))
                else:
                    result.extend(self._split_code_chunk(block, max_words))
            else:
                result.append(block)
            cursor = end

        if cursor < len(text):
            result.extend(self._split_prose_chunk(text[cursor:], max_words))

        return result

    def _split_prose_chunk(self, text: str, max_words: int, tail_floor: int = 25) -> list[str]:
        """Split prose into word-window chunks, respecting blank-line boundaries.

        A trailing window smaller than ``tail_floor`` words is merged into the
        previous window so word-window splitting never emits 1–9 word nubs.
        """
        if not text.strip():
            return []
        chunks: list[str] = []
        words = text.split()
        if len(words) <= max_words:
            return [text]
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i : i + max_words]))
        if len(chunks) >= 2 and len(chunks[-1].split()) < tail_floor:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()
        return chunks

    def _split_table_chunk(self, table: str, max_words: int) -> list[str]:
        """Split an oversized HTML table into greedy ``<tr>`` row-unit chunks.

        Rows are packed up to ``max_words``; a row is never cut in half.  The
        ``<table>``/``</table>`` wrapper stays with the first/last chunk so the
        emitted fragments are still self-delimiting.
        """
        rows = [m.group(0) for m in _TR_RE.finditer(table)]
        if not rows:
            return [table]

        packed: list[list[str]] = []
        current: list[str] = []
        current_words = 0
        for row in rows:
            row_words = len(row.split())
            if current_words + row_words > max_words and current:
                packed.append(current)
                current = []
                current_words = 0
            current.append(row)
            current_words += row_words
        if current:
            packed.append(current)

        head = table[: table.find("<tr")]
        tail = table[table.rfind("</tr>") + len("</tr>") :] if "</tr>" in table else ""
        if head == table:
            head = ""

        chunks: list[str] = []
        for i, piece in enumerate(packed):
            body = "".join(piece)
            if i == 0 and len(packed) == 1:
                chunks.append(head + body + tail)
            elif i == 0:
                chunks.append(head + body)
            elif i == len(packed) - 1:
                chunks.append(body + tail)
            else:
                chunks.append(body)
        return chunks

    # Lines that start a new top-level function/class in common Spark languages.
    _FN_BOUNDARY_RE = re.compile(r"^(\s*)(def |class |async\s+def |@)")

    def _split_code_chunk(self, code: str, max_words: int) -> list[str]:
        """Split an oversized fenced code block at function/class boundaries.

        The block is first grouped into *units* at definition boundaries
        (``def``/``class``/``async def``/decorators), then whole units are packed
        greedily into chunks up to ``max_words``.  A unit is only sub-split by
        line when a single definition alone exceeds the word budget, so ordinary
        functions are never cut in half.  Every emitted chunk is re-wrapped in
        its own balanced fence (same markers/language tag), so a split fence
        never leaves a dangling opener in one chunk and its closer in another.
        """
        if not code.strip() or len(code.split()) <= max_words:
            return [code]

        # Split off the opening fence line and the closing marker.
        fence = re.match(r"^(\s{0,3})(`{3,}|~{3,})[^\r\n]*\r?\n", code)
        opener = fence.group(0) if fence else ""
        marker = fence.group(2) if fence else "```"
        body = code[len(opener) :]
        close_pat = re.compile(r"\s*" + re.escape(marker) + r"[^\r\n]*(?:\r?\n)?\Z")
        body = close_pat.sub("", body)

        lines = body.splitlines(keepends=True)
        units: list[list[str]] = []
        current_unit: list[str] = []
        for line in lines:
            if self._FN_BOUNDARY_RE.match(line) and current_unit:
                units.append(current_unit)
                current_unit = [line]
            else:
                current_unit.append(line)
        if current_unit:
            units.append(current_unit)

        # A single definition larger than the budget is sub-split by lines so
        # the packer never emits an oversized chunk.
        expanded: list[str] = []
        for unit in units:
            unit_text = "".join(unit)
            if len(unit_text.split()) > max_words and len(unit) > 1:
                expanded.extend(self._split_lines(unit, max_words))
            else:
                expanded.append(unit_text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_words = 0
        for unit_text in expanded:
            unit_words = len(unit_text.split())
            if current_words + unit_words > max_words and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_words = 0
            current_chunk.append(unit_text)
            current_words += unit_words
        if current_chunk:
            chunks.append("".join(current_chunk))

        # Re-fence each emitted chunk so fences stay balanced per chunk.
        return [opener + body_text + "\n" + marker for body_text in chunks]

    def _split_lines(self, lines: list[str], max_words: int) -> list[str]:
        """Sub-split an over-budget unit into line-bounded chunks."""
        chunks: list[str] = []
        current: list[str] = []
        current_words = 0
        for line in lines:
            line_words = len(line.split())
            if current_words + line_words > max_words and current:
                chunks.append("".join(current))
                current = []
                current_words = 0
            current.append(line)
            current_words += line_words
        if current:
            chunks.append("".join(current))
        return chunks

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge_sections(
        self,
        sections: list[_RawSection],
        document: ParsedDocument,
    ) -> list[DocumentChunk]:
        """Merge small sections into chunks respecting parent boundaries."""
        chunks: list[DocumentChunk] = []
        current_text_parts: list[str] = []
        current_code_parts: list[str] = []
        current_section_offsets: list[tuple[int, int]] = []
        current_heading: str = ""
        current_path: tuple[str, ...] = ()
        current_words = 0
        overlap_tail: str = ""

        def _flush() -> None:
            nonlocal current_text_parts, current_code_parts, current_words, current_section_offsets, overlap_tail
            if not current_text_parts:
                return
            body = "\n\n".join(current_text_parts).strip()
            if not body:
                overlap_tail = ""
                current_text_parts = []
                current_code_parts = []
                current_section_offsets = []
                current_words = 0
                return

            # min-chunk gate counts CONTENT words only: a breadcrumb prefix must
            # never inflate a tiny section past the minimum.
            content_wc = len(body.split())
            if self.prepend_heading_path and current_path:
                body = " ".join(current_path) + "\n\n" + body
            wc = len(body.split())
            if content_wc >= self.min_chunk_words:
                # Stash the verbatim tail of the last accumulated section so
                # the next chunk can open with it (overlap), preserving line
                # structure instead of a whitespace-flattened fragment.
                if self.overlap_words > 0:
                    overlap_tail = _verbatim_tail(current_text_parts[-1], self.overlap_words)
                chunk_id = self._chunk_id(document, len(chunks))
                # Determine chunk type. NOTE (M8): ``"code"`` is produced here
                # only when a section's body is empty while its code blocks are
                # non-empty — unreachable from the markdown paths, because
                # ``section.text`` always contains the fence body too. The
                # corpus ``code`` rows (2,664 in the pinned generation) come
                # exclusively from the Spark extractors, which create
                # ``DocumentChunk`` rows directly; this branch stays as a guard
                # for any future zero-prose code source.
                ct = "text"
                if current_code_parts and not current_text_parts:
                    ct = "code"
                elif current_code_parts:
                    ct = "mixed"
                start_offset = current_section_offsets[0][0] if current_section_offsets else 0
                end_offset = current_section_offsets[-1][1] if current_section_offsets else 0
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_name=document.source_name,
                        title=document.title,
                        url=document.url,
                        text=body,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        section_header=current_heading,
                        chunk_type=ct,
                        word_count=wc,
                        heading_path=current_path,
                        doc_type=document.doc_type,
                        file_path=document.file_path,
                        language=document.language,
                        spark_version=document.spark_version,
                        module=document.module,
                        source_commit=document.source_commit,
                        license=document.license,
                    )
                )
            elif chunks:
                # Sub-minimum trailing content (e.g. the tail of a short API
                # reference page split across nested headings) must not be
                # silently discarded — append it to the previous chunk so the
                # page keeps every word. Only a document whose *entire* body is
                # below the minimum is filtered out (matches the no-content
                # contract of ``min_chunk_words``).
                overlap_tail = ""
                last = chunks[-1]
                merged_body = body
                if self.prepend_heading_path and current_path:
                    merged_body = " ".join(current_path) + "\n\n" + merged_body
                merged_text = last.text + "\n\n" + merged_body
                merged_end = current_section_offsets[-1][1] if current_section_offsets else last.end_offset
                chunks[-1] = replace(
                    last,
                    text=merged_text,
                    word_count=len(merged_text.split()),
                    end_offset=merged_end,
                )

            current_text_parts = []
            current_code_parts = []
            current_section_offsets = []
            current_words = 0

        def _open_chunk() -> None:
            nonlocal overlap_tail, current_text_parts, current_words
            # Populate a freshly flushed accumulation with the verbatim tail of
            # the previous chunk's last section (overlap across ANY boundary).
            if self.overlap_words > 0 and overlap_tail:
                current_text_parts.append(overlap_tail)
                current_words = len(overlap_tail.split())
                overlap_tail = ""

        for section in sections:
            section_wc = len(section.text.split()) if section.text else 0

            # If this section is under a different parent than current accumulation,
            # flush first to preserve topical boundaries — but only when enough
            # words have accumulated to clear the minimum. A sub-minimum flush
            # would silently drop the whole (small) section, losing content from
            # short API-reference pages whose nested headings each own a couple
            # of lines. Carrying the content forward merges it with the next
            # section instead of discarding it. Overlap is NOT carried across a
            # parent transition: re-opening the previous topic's tail would mix
            # unrelated content under the new header.
            parent_current = current_path[:-1] if current_path else ()
            parent_new = section.heading_path[:-1] if section.heading_path else ()
            if parent_current != parent_new and current_text_parts and current_words >= self.min_chunk_words:
                _flush()

            # Would adding this section exceed the target?
            if current_words + section_wc > self.chunk_size_words and current_text_parts:
                _flush()
                _open_chunk()

            # Accumulate
            if section.text:
                current_text_parts.append(section.text)
                current_section_offsets.append((section.start, section.end))
            if section.code_blocks:
                current_code_parts.extend(section.code_blocks)
            current_words += section_wc
            current_heading = section.header
            current_path = section.heading_path

        _flush()
        return self._number_chunks(chunks)

    def _number_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Assign ``chunk_index`` / ``total_chunks`` to every chunk."""
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            if chunk.chunk_index != i or chunk.total_chunks != total:
                chunks[i] = replace(chunk, chunk_index=i, total_chunks=total)
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id(document: ParsedDocument, index: int) -> str:
        import uuid

        namespace = uuid.uuid5(uuid.NAMESPACE_DNS, document.url)
        return str(uuid.uuid5(namespace, f"{document.source_name}:hdr:{index:04d}"))
