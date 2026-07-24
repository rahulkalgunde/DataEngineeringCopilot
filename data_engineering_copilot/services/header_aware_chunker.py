"""Header-aware markdown chunker.

Splits documents along Markdown headers (#, ##, ###, etc.) to preserve
topical boundaries. Each chunk retains its heading hierarchy and any
embedded code blocks.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.utils.text import slugify

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_FENCE_RE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


@dataclass
class _RawSection:
    """Intermediate representation before merging."""

    header: str
    level: int
    heading_path: tuple[str, ...]
    text: str
    code_blocks: tuple[str, ...]


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
    """

    def __init__(
        self,
        chunk_size_words: int = 375,
        overlap_words: int = 90,
        min_chunk_words: int = 10,
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

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Chunk *document* by splitting on Markdown headers."""
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

    @staticmethod
    def _split_into_sections(text: str) -> list[_RawSection]:
        """Split markdown *text* into sections at header boundaries."""
        matches = list(_HEADER_RE.finditer(text))
        if not matches:
            return []

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
                )
            )

        heading_stack: list[str] = []

        for i, m in enumerate(matches):
            level = len(m.group(1))
            header_text = m.group(2).strip()
            start = m.end() + 1  # skip the newline after header
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw_body = text[start:end]

            # Extract code blocks from the body
            code_blocks = tuple(blk.group(0) for blk in _FENCE_RE.finditer(raw_body))

            # Build heading path
            while heading_stack and len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(header_text)
            path = tuple(heading_stack)

            sections.append(
                _RawSection(
                    header=header_text,
                    level=level,
                    heading_path=path,
                    text=raw_body.strip(),
                    code_blocks=code_blocks,
                )
            )

        return sections

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
        current_heading: str = ""
        current_path: tuple[str, ...] = ()
        current_words = 0

        def _flush() -> None:
            nonlocal current_text_parts, current_code_parts, current_words
            if not current_text_parts:
                return
            body = "\n\n".join(current_text_parts).strip()
            if not body:
                current_text_parts = []
                current_code_parts = []
                current_words = 0
                return

            wc = len(body.split())
            if wc >= self.min_chunk_words:
                chunk_id = self._chunk_id(document, len(chunks))
                enriched = f"[Source: {document.title}]\n{body}"
                # Determine chunk type
                ct = "text"
                if current_code_parts and not current_text_parts:
                    ct = "code"
                elif current_code_parts:
                    ct = "mixed"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_name=document.source_name,
                        title=document.title,
                        url=document.url,
                        text=enriched,
                        section_header=current_heading,
                        chunk_type=ct,
                        word_count=wc,
                        heading_path=current_path,
                    )
                )

            current_text_parts = []
            current_code_parts = []
            current_words = 0

        for section in sections:
            section_wc = len(section.text.split()) if section.text else 0

            # If this section is under a different parent than current accumulation,
            # flush first to preserve topical boundaries.
            parent_current = current_path[:-1] if current_path else ()
            parent_new = section.heading_path[:-1] if section.heading_path else ()
            if parent_current != parent_new and current_text_parts:
                _flush()

            # Would adding this section exceed the target?
            if current_words + section_wc > self.chunk_size_words and current_text_parts:
                _flush()
                # Start new chunk with overlap
                if self.overlap_words > 0 and chunks:
                    prev_text = current_text_parts[-1] if current_text_parts else ""
                    if not prev_text and chunks:
                        # Extract overlap from last flushed chunk text (strip prefix)
                        last_body = chunks[-1].text
                        prefix_end = last_body.find("\n")
                        if prefix_end != -1:
                            last_body = last_body[prefix_end + 1 :]
                        words = last_body.split()[-self.overlap_words :]
                        if words:
                            overlap_text = " ".join(words)
                            current_text_parts.append(overlap_text)
                            current_words = len(words)

            # Accumulate
            if section.text:
                current_text_parts.append(section.text)
            if section.code_blocks:
                current_code_parts.extend(section.code_blocks)
            current_words += section_wc
            current_heading = section.header
            current_path = section.heading_path

        _flush()
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id(document: ParsedDocument, index: int) -> str:
        digest = hashlib.sha1(document.url.encode("utf-8")).hexdigest()[:10]
        source = slugify(document.source_name)
        return f"{source}:{digest}:hdr:{index:04d}"
