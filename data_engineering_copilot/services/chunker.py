from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument

if TYPE_CHECKING:
    from langchain_text_splitters import Language

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Syntax-aware document chunker using langchain RecursiveCharacterTextSplitter.

    Detects programming language from the document URL and uses
    language-appropriate splitters (e.g. splitting Python on class/def
    boundaries, SQL on statement boundaries, etc.). Falls back to a
    generic text splitter for non-code documents.
    """

    def __init__(
        self,
        chunk_size_chars: int = 1000,
        chunk_overlap_chars: int = 100,
    ) -> None:
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be positive")
        if chunk_overlap_chars < 0 or chunk_overlap_chars >= chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be >= 0 and less than chunk_size_chars")

        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars

        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

        self._default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_chars,
            chunk_overlap=chunk_overlap_chars,
            separators=["\n\n", "\n", " ", ""],
        )
        self._language_splitters = {
            Language.PYTHON: RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON, chunk_size=chunk_size_chars, chunk_overlap=chunk_overlap_chars
            ),
            Language.SCALA: RecursiveCharacterTextSplitter.from_language(
                Language.SCALA, chunk_size=chunk_size_chars, chunk_overlap=chunk_overlap_chars
            ),
            Language.JAVA: RecursiveCharacterTextSplitter.from_language(
                Language.JAVA, chunk_size=chunk_size_chars, chunk_overlap=chunk_overlap_chars
            ),
            Language.R: RecursiveCharacterTextSplitter.from_language(
                Language.R, chunk_size=chunk_size_chars, chunk_overlap=chunk_overlap_chars
            ),
        }

    def _detect_language(self, url: str) -> Language | None:
        from langchain_text_splitters import Language

        url_lower = url.lower()
        if "/api/python/" in url_lower or "/pyspark" in url_lower:
            return Language.PYTHON
        if "/api/scala/" in url_lower:
            return Language.SCALA
        if "/api/java/" in url_lower:
            return Language.JAVA
        if "/api/r/" in url_lower:
            return Language.R
        return None

    async def chunk(
        self, document: ParsedDocument, precomputed_embeddings: list[list[float]] | None = None
    ) -> list[DocumentChunk]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_chunk, document)

    def extract_sentences(self, text: str) -> list[str] | None:
        """Sentence pre-extraction is not supported by this chunker."""
        return None

    def _sync_chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        lang = self._detect_language(document.url)
        splitter = self._language_splitters[lang] if lang is not None else self._default_splitter

        texts = splitter.split_text(document.text)

        chunks: list[DocumentChunk] = []
        total_chunks = len(texts)
        cursor = 0
        for i, text in enumerate(texts):
            # The splitter may drop separator characters, so pieces are not
            # guaranteed to tile the source contiguously. Locate each piece by
            # searching from the running cursor; fall back to best-effort when
            # the exact text is not found.
            start_offset = document.text.find(text, cursor)
            if start_offset == -1:
                start_offset = cursor
            end_offset = start_offset + len(text)
            cursor = end_offset
            chunk = DocumentChunk(
                chunk_id=self._chunk_id(document, i),
                source_name=document.source_name,
                title=document.title,
                url=document.url,
                text=text,
                start_offset=start_offset,
                end_offset=end_offset,
                chunk_index=i,
                total_chunks=total_chunks,
            )
            chunks.append(chunk)

        logger.info(
            "Chunked document source=%s url=%s title=%r lang=%s chunks=%s",
            document.source_name,
            document.url,
            document.title,
            lang.name if lang else "text",
            len(chunks),
        )
        return chunks

    def _is_valid_chunk(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        return any(c.isalnum() for c in text)

    def _chunk_id(self, document: ParsedDocument, index: int) -> str:
        import uuid

        namespace = uuid.uuid5(uuid.NAMESPACE_DNS, document.url)
        return str(uuid.uuid5(namespace, f"{document.source_name}:{index:04d}"))
