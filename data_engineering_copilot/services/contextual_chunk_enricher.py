"""Anthropic-style contextual chunk enrichment.

Injects a short document summary before each chunk so isolated chunks
carry document-level meaning during similarity search.
"""

from __future__ import annotations

import logging
from typing import Protocol

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument

logger = logging.getLogger(__name__)


class ContextSummarizer(Protocol):
    async def summarize(self, document: ParsedDocument) -> str: ...


class LLMContextSummarizer:
    """Generates a concise document summary for contextual chunk enrichment."""

    def __init__(self, llm_client, max_summary_words: int = 50) -> None:
        self._llm_client = llm_client
        self._max_summary_words = max_summary_words

    async def summarize(self, document: ParsedDocument) -> str:
        prompt = (
            f"Summarise the following documentation page in under {self._max_summary_words} words.\n"
            f"Title: {document.title}\n"
            f"Content:\n{document.text[:2000]}\n\nSummary:"
        )
        try:
            result = await self._llm_client.generate(prompt)
            return result.strip().split("\n")[0][:500]
        except Exception as exc:
            logger.warning("Context summarisation failed: %s", exc)
            return ""


class ContextualChunkEnricher:
    """Enriches each chunk with document-level context (Anthropic-style).

    Prepends a ``context`` field to each chunk's payload containing a short
    document summary so that isolated chunks carry document-level meaning
    during similarity search.
    """

    def __init__(
        self,
        summarizer: ContextSummarizer | None = None,
        enabled: bool = False,
    ) -> None:
        self._summarizer = summarizer
        self._enabled = enabled

    async def enrich(
        self,
        document: ParsedDocument,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Enrich chunks with document-level context summary (per-document)."""
        if not self._enabled or self._summarizer is None:
            return chunks

        summary = await self._summarizer.summarize(document)
        if not summary:
            return chunks

        enriched: list[DocumentChunk] = []
        for chunk in chunks:
            context_text = f"[Context: {summary}]\n{chunk.text}"
            enriched.append(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    source_name=chunk.source_name,
                    title=chunk.title,
                    url=chunk.url,
                    text=context_text,
                    content_hash=chunk.content_hash,
                    section_header=chunk.section_header,
                    chunk_type=chunk.chunk_type,
                    word_count=chunk.word_count,
                    heading_path=chunk.heading_path,
                )
            )
        logger.info(
            "contextual_enrichment source=%s url=%s chunks=%d summary_len=%d",
            document.source_name,
            document.url,
            len(chunks),
            len(summary),
        )
        return enriched

    async def enrich_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Batch-level enrichment interface (used by AsyncIngestionService).

        When called at batch level without document context, groups chunks
        by (source_name, title) and enriches each group with a title-based summary.
        """
        if not self._enabled or self._summarizer is None:
            return chunks

        from collections import defaultdict
        groups: dict[tuple[str, str], list[DocumentChunk]] = defaultdict(list)
        for chunk in chunks:
            groups[(chunk.source_name, chunk.title)].append(chunk)

        enriched: list[DocumentChunk] = []
        for (source_name, title), group_chunks in groups.items():
            fake_doc = ParsedDocument(
                source_name=source_name,
                title=title,
                url=group_chunks[0].url if group_chunks else "",
                text=title,
            )
            enriched.extend(await self.enrich(fake_doc, group_chunks))

        return enriched
