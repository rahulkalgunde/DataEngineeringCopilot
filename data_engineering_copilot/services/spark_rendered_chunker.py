"""Chunking for locally rendered Spark documentation pages.

Chunks rendered HTML output by heading hierarchy while preserving code blocks
with their surrounding section. Falls back to paragraph/fixed-size splitting
when a page has no headings. Uses the same deterministic chunk ID scheme as
``SparkChunker`` with the representation included.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.infrastructure.spark_html_parser import RenderedParseResult
from data_engineering_copilot.services.spark_metadata import SparkMetadata

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^(```+|~~~+)(\w*)\n(.*?)^\1", re.MULTILINE | re.DOTALL)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")

_MIN_CHUNK_WORDS = 10


@dataclass(frozen=True)
class SparkRenderedChunker:
    """Chunk rendered Spark HTML main content deterministically."""

    async def chunk(self, parsed: ParsedDocument, metadata: SparkMetadata) -> list[DocumentChunk]:
        """Chunk a rendered page into heading-bounded, code-preserving chunks."""
        chunks = self._chunk_by_headings(parsed.text, parsed, metadata)
        if not chunks:
            chunks = self._chunk_fallback(parsed.text, parsed, metadata)
        return [self._number(chunk, index, len(chunks)) for index, chunk in enumerate(chunks)]

    # ------------------------------------------------------------------
    # Heading-bounded chunking
    # ------------------------------------------------------------------

    def _chunk_by_headings(
        self,
        text: str,
        parsed: ParsedDocument,
        metadata: SparkMetadata,
    ) -> list[DocumentChunk]:
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return []

        sections: list[tuple[tuple[str, ...], str, list[str], int, int]] = []

        # Preamble before the first heading.
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(((), preamble, [], 0, len(preamble)))

        heading_stack: list[tuple[str, int]] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            header = m.group(2).strip()
            start = m.end() + 1
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw_body = text[start:end].strip()

            while heading_stack and heading_stack[-1][1] >= level:
                heading_stack.pop()
            heading_stack.append((header, level))
            path = tuple(h for h, _ in heading_stack)
            sections.append((path, header, [raw_body], start, end))

        chunks: list[DocumentChunk] = []
        for path, header, body_parts, start, end in sections:
            body = "\n".join(p for p in body_parts if p.strip()).strip()
            if not body:
                continue
            code_blocks = tuple(blk.group(0) for blk in _FENCE_RE.finditer(body))
            chunk_type = self._classify(header, body, code_blocks)
            chunks.append(self._build_chunk(parsed, metadata, path, header, body, chunk_type, start, end))

        if not chunks:
            return []
        return chunks

    # ------------------------------------------------------------------
    # Fallback chunking (no headings)
    # ------------------------------------------------------------------

    def _chunk_fallback(
        self,
        text: str,
        parsed: ParsedDocument,
        metadata: SparkMetadata,
    ) -> list[DocumentChunk]:
        pieces = _split_paragraphs(text)
        if not pieces:
            pieces = [text]
        chunks: list[DocumentChunk] = []
        cursor = 0
        for piece in pieces:
            if not piece.strip():
                continue
            start_offset = text.find(piece, cursor)
            end_offset = start_offset + len(piece) if start_offset != -1 else cursor + len(piece)
            cursor = end_offset
            code_blocks = tuple(blk.group(0) for blk in _FENCE_RE.finditer(piece))
            chunk_type = self._classify("", piece, code_blocks)
            chunks.append(self._build_chunk(parsed, metadata, (), "", piece, chunk_type, start_offset, end_offset))
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(header: str, body: str, code_blocks: tuple[str, ...]) -> str:
        if code_blocks:
            return "code" if not body.replace("\n", " ").strip() or body.strip().startswith("```") else "mixed"
        lowered = header.lower()
        if any(token in lowered for token in ("api", "class ", "method ", "function ")):
            return "api"
        return "text"

    @staticmethod
    def _build_chunk(
        parsed: ParsedDocument,
        metadata: SparkMetadata,
        path: tuple[str, ...],
        header: str,
        body: str,
        chunk_type: str,
        start_offset: int = 0,
        end_offset: int = 0,
    ) -> DocumentChunk:
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        index = 0
        chunk_id = _rendered_chunk_id(metadata, index, body)
        return DocumentChunk(
            chunk_id=chunk_id,
            source_name=parsed.source_name,
            title=header or parsed.title,
            url=parsed.url,
            text=body,
            start_offset=start_offset,
            end_offset=end_offset,
            content_hash=content_hash,
            section_header=header,
            chunk_type=chunk_type,
            word_count=len(body.split()),
            heading_path=path,
            chunk_index=index,
            total_chunks=0,
            doc_type=metadata.doc_type,
            language=metadata.language,
            spark_version=metadata.spark_version,
            module=metadata.module,
            source_commit=metadata.source_commit,
            file_path=metadata.file_path,
            license=metadata.license,
            deployment_mode=metadata.deployment_mode,
        )

    @staticmethod
    def _number(chunk: DocumentChunk, index: int, total: int) -> DocumentChunk:
        return replace(chunk, chunk_index=index, total_chunks=total)


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text)]
    return [p for p in parts if p and len(p.split()) >= _MIN_CHUNK_WORDS]


def _rendered_chunk_id(metadata: SparkMetadata, index: int, text: str) -> str:
    """Deterministic chunk ID for rendered content.

    Mirrors ``SparkChunker._deterministic_chunk_id`` but includes the
    ``rendered`` representation so native and rendered chunks never collide.
    """
    digest = hashlib.sha256(
        f"rendered|{metadata.source_commit}|{metadata.file_path}|{index}|{text}".encode()
    ).hexdigest()
    return f"spark-rendered-{metadata.source_commit[:8]}-{index}-{digest[:12]}"


def chunk_rendered_document(
    result: RenderedParseResult,
    metadata: SparkMetadata,
    source_name: str,
    doc_type: str,
    language: str,
) -> list[DocumentChunk]:
    """Synchronous convenience wrapper for offline rendering pipelines."""
    parsed = ParsedDocument(
        source_name=source_name,
        title=result.title,
        url=result.canonical_url,
        text=result.text,
        doc_type=doc_type,
        language=language,
        spark_version=metadata.spark_version,
        module=metadata.module,
        source_commit=metadata.source_commit,
        file_path=result.source_path,
        license=metadata.license,
    )
    import asyncio

    chunker = SparkRenderedChunker()
    return asyncio.run(chunker.chunk(parsed, metadata))
