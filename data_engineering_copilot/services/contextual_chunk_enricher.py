"""Anthropic-style contextual chunk enrichment.

Injects a short document summary before each chunk so isolated chunks
carry document-level meaning during similarity search.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument

logger = logging.getLogger(__name__)

_INDEX_URL_BLACKLIST = frozenset(
    {
        "index-all.html",
        "deprecated-list.html",
        "package-summary.html",
        "allclasses-index.html",
        "allpackages-index.html",
        "constant-values.html",
        "serialized-form.html",
        "overview-tree.html",
        "help-doc.html",
    }
)

_MIN_CONTENT_WORDS = 40


class ContextSummarizer(Protocol):
    async def summarize(self, document: ParsedDocument) -> str: ...


class LLMContextSummarizer:
    """Generates a concise document summary for contextual chunk enrichment."""

    def __init__(self, llm_client, max_summary_words: int = 50) -> None:
        self._llm_client = llm_client
        self._max_summary_words = max_summary_words

    async def summarize(self, document: ParsedDocument) -> str:
        prompt = (
            "You are a technical documentation indexer.\n"
            f"Provide a direct 1-2 sentence overview (under {self._max_summary_words} words) "
            "of the documentation page below.\n"
            "State ONLY what main concepts, components, or procedures are documented.\n"
            "INTERNAL STYLE: flat, factual, no introductory fluff.\n"
            "If the page lacks substantive content beyond navigation links, headers, "
            "or index listings, return exactly: NO_CONTENT_TO_SUMMARIZE\n\n"
            f"Title: {document.title}\n"
            f"Content:\n{document.text[:3000]}\n\n"
            "Summary:"
        )
        try:
            result = await self._llm_client.generate(prompt)
            return self._clean_summary(result)
        except Exception as exc:
            logger.warning("Context summarisation failed for title '%s': %s", document.title, exc)
            return ""

    def _clean_summary(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        text = raw_text.replace("```", "").strip(' \t\n\r"')
        if text.upper() == "NO_CONTENT_TO_SUMMARIZE":
            return ""
        text = re.sub(
            r"^(here is (a|the) summary|summary|this page|this document|overview|"
            r"the documentation page|the following page|the provided page):?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = " ".join(lines)
        return clean_text[:400].strip()


class ContextualChunkEnricher:
    """Enriches each chunk with document-level context (Anthropic-style).

    Prepends a short document summary before each chunk so isolated chunks
    carry document-level meaning during similarity search.
    """

    def __init__(
        self,
        summarizer: ContextSummarizer | None = None,
        enabled: bool = False,
        batch_size: int = 20,
    ) -> None:
        self._summarizer = summarizer
        self._enabled = enabled
        self._batch_size = batch_size

    def _is_blacklisted_url(self, url: str) -> bool:
        path = url.rsplit("/", 1)[-1] if "/" in url else url
        return path in _INDEX_URL_BLACKLIST

    async def enrich(
        self,
        document: ParsedDocument,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        if not self._enabled or self._summarizer is None or not chunks:
            return chunks

        if self._is_blacklisted_url(document.url):
            logger.info(
                "enrichment_skipped_blacklist url=%s title='%s'",
                document.url,
                document.title,
            )
            return chunks

        word_count = len(document.text.split())
        if word_count < _MIN_CONTENT_WORDS:
            logger.info(
                "enrichment_skipped_short_doc url=%s title='%s' words=%d",
                document.url,
                document.title,
                word_count,
            )
            return chunks

        summary = await self._summarizer.summarize(document)
        if not summary:
            return chunks

        enriched: list[DocumentChunk] = []
        for chunk in chunks:
            context_text = f"[Document Context: {summary}]\n{chunk.text}"
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
                    word_count=len(context_text.split()),
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
