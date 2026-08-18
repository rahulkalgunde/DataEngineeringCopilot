"""JSON structured-data chunker.

Emits one chunk per logical JSON unit (a bounded object, a key-value pair, or
an array row), each annotated with its JSON path (``$.users[0].name``) as
``section_header``. Values are serialized with ``json.dumps`` so their content
is preserved verbatim; oversized values are split losslessly with the shared
token-budget utility. Malformed or non-JSON text is never silently discarded —
it falls back to lossless text chunks so ``"".join(chunk_texts)`` still
reconstructs the source.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.infrastructure.token_budget import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    count_tokens,
    split_text_losslessly,
)

logger = logging.getLogger(__name__)


class StructuredDataChunker:
    """Chunker that splits JSON documents along logical record boundaries.

    Parameters
    ----------
    max_tokens:
        Hard token budget per emitted chunk (matches the embedding input
        budget used elsewhere).
    max_chars:
        Hard character budget per emitted chunk. Oversized serialized values
        are split losslessly into bounded segments.
    """

    chunker_version = "structured-json-v1"

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_tokens = max_tokens
        self.max_chars = max_chars

    async def chunk(
        self, document: ParsedDocument, precomputed_embeddings: list[list[float]] | None = None
    ) -> list[DocumentChunk]:
        """Chunk *document* (whose ``text`` is JSON) into structured units."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_chunk, document)

    def extract_sentences(self, text: str) -> list[str] | None:
        """Sentence pre-extraction is not supported for structured data."""
        return None

    def _sync_chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = document.text.strip()
        if not text:
            return []

        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return self._fallback_text_chunks(document, text)

        units = self._emit(value, "$")

        chunks: list[DocumentChunk] = []
        total_chunks = len(units)
        for index, (path, unit_text) in enumerate(units):
            chunks.append(
                DocumentChunk(
                    chunk_id=self._chunk_id(document, path),
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=unit_text,
                    content_hash=hashlib.sha256(unit_text.encode("utf-8")).hexdigest(),
                    section_header=path,
                    chunk_type="structured",
                    word_count=len(unit_text.split()),
                    chunk_index=index,
                    total_chunks=total_chunks,
                    doc_type=document.doc_type,
                    language=document.language,
                    file_path=document.file_path,
                    chunker_version=self.chunker_version,
                    token_count=count_tokens(unit_text),
                    character_count=len(unit_text),
                )
            )

        logger.info(
            "Structured chunking: source=%s url=%s units=%d chunks=%d",
            document.source_name,
            document.url,
            len(units),
            len(chunks),
        )
        return chunks

    def _emit(self, value: object, path: str) -> list[tuple[str, str]]:
        """Recursively emit ``(path, json.dumps(value))`` units for *value*.

        A unit that fits within both budgets is emitted whole. A unit that does
        not fit is decomposed: dicts by key, lists by element, and oversized
        scalar serializations via the lossless token-budget splitter.
        """
        serialized = json.dumps(value, ensure_ascii=False)
        if count_tokens(serialized) <= self.max_tokens and len(serialized) <= self.max_chars:
            return [(path, serialized)]

        if isinstance(value, dict):
            units: list[tuple[str, str]] = []
            for key, child in value.items():
                units.extend(self._emit(child, f"{path}.{key}"))
            return units

        if isinstance(value, list):
            units = []
            for index, child in enumerate(value):
                units.extend(self._emit(child, f"{path}[{index}]"))
            return units

        return self._split_scalar(value, path)

    def _split_scalar(self, value: object, path: str) -> list[tuple[str, str]]:
        """Split an oversized scalar serialization losslessly.

        The shared token-budget utility splits along its boundary heuristics;
        a single whitespace-free atomic token longer than ``max_chars`` raises
        instead of dropping characters, so string values fall back to
        char-bounded windows (each still emitted as a valid ``json.dumps``
        string whose parsed content reconstructs the original).
        """
        serialized = json.dumps(value, ensure_ascii=False)
        try:
            segments = split_text_losslessly(serialized, max_tokens=self.max_tokens, max_chars=self.max_chars)
        except ValueError:
            if not isinstance(value, str):
                raise
            return [(path, json.dumps(piece, ensure_ascii=False)) for piece in self._split_string_value(value)]
        return [(path, segment) for segment in segments]

    def _split_string_value(self, value: str) -> list[str]:
        """Char-bounded windows of *value* whose serialization fits ``max_chars``.

        Reserves two characters for the JSON quotes. Tries the lossless
        utility first so internal boundaries (newlines, paragraphs) are
        honoured when possible; atomic runs fall back to fixed windows.
        """
        piece_budget = max(self.max_chars - 2, 1)
        try:
            return split_text_losslessly(value, max_tokens=self.max_tokens, max_chars=piece_budget)
        except ValueError:
            return [value[i : i + piece_budget] for i in range(0, len(value), piece_budget)]

    def _fallback_text_chunks(self, document: ParsedDocument, text: str) -> list[DocumentChunk]:
        """Losslessly split non-JSON text so malformed input is never dropped."""
        segments = split_text_losslessly(text, max_tokens=self.max_tokens, max_chars=self.max_chars)
        chunks: list[DocumentChunk] = []
        total_chunks = len(segments)
        for index, segment in enumerate(segments):
            chunks.append(
                DocumentChunk(
                    chunk_id=self._chunk_id(document, f"$[{index}]"),
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=segment,
                    content_hash=hashlib.sha256(segment.encode("utf-8")).hexdigest(),
                    section_header="$",
                    chunk_type="text",
                    word_count=len(segment.split()),
                    chunk_index=index,
                    total_chunks=total_chunks,
                    doc_type=document.doc_type,
                    language=document.language,
                    file_path=document.file_path,
                    chunker_version=self.chunker_version,
                    token_count=count_tokens(segment),
                    character_count=len(segment),
                )
            )
        return chunks

    @staticmethod
    def _chunk_id(document: ParsedDocument, path: str) -> str:
        namespace = uuid.uuid5(uuid.NAMESPACE_DNS, document.url)
        return str(uuid.uuid5(namespace, f"{document.source_name}:json:{path}"))
