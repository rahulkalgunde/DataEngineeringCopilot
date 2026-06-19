from __future__ import annotations

import hashlib
import logging

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.utils.text import slugify


logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, chunk_size_words: int, overlap_words: int) -> None:
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if overlap_words < 0 or overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be >= 0 and less than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        words = document.text.split()
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        step = self.chunk_size_words - self.overlap_words

        while start < len(words):
            end = min(start + self.chunk_size_words, len(words))
            text = " ".join(words[start:end])
            chunk_id = self._chunk_id(document, index)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=text,
                )
            )
            if end == len(words):
                break
            start += step
            index += 1

        logger.info(
            "Chunked document source=%s url=%s title=%r words=%s chunks=%s",
            document.source_name,
            document.url,
            document.title,
            len(words),
            len(chunks),
        )
        return chunks

    def _chunk_id(self, document: ParsedDocument, index: int) -> str:
        digest = hashlib.sha1(document.url.encode("utf-8")).hexdigest()[:10]
        source = slugify(document.source_name)
        return f"{source}:{digest}:{index:04d}"
