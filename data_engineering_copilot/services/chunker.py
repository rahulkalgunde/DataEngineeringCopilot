from __future__ import annotations

import hashlib
import logging

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.utils.text import slugify

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
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self._language_splitters = {
            Language.PYTHON: RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ),
            Language.SCALA: RecursiveCharacterTextSplitter.from_language(
                Language.SCALA, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ),
            Language.JAVA: RecursiveCharacterTextSplitter.from_language(
                Language.JAVA, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ),
            Language.R: RecursiveCharacterTextSplitter.from_language(
                Language.R, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ),
        }

    def _detect_language(self, url: str) -> Language | None:
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

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        lang = self._detect_language(document.url)
        splitter = self._language_splitters.get(lang, self._default_splitter)

        texts = splitter.split_text(document.text)

        chunks: list[DocumentChunk] = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                chunk_id=self._chunk_id(document, i),
                source_name=document.source_name,
                title=document.title,
                url=document.url,
                text=text,
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
        digest = hashlib.sha1(document.url.encode("utf-8")).hexdigest()[:10]
        source = slugify(document.source_name)
        return f"{source}:{digest}:{index:04d}"
