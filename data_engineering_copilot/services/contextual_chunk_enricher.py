"""Anthropic-style contextual chunk enrichment.

Injects a short document summary before each chunk so isolated chunks
carry document-level meaning during similarity search.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Protocol

from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.infrastructure.llm_client import SYSTEM_BLOCK_SEPARATOR
from data_engineering_copilot.observability.langfuse_prompts import get_langfuse_prompt, register_fallback

logger = logging.getLogger(__name__)

# Offline fallback for the Langfuse-managed ``chunk-enrichment-summary`` prompt.
_SUMMARY_PROMPT = (
    "You are a technical documentation indexer.\n"
    "Provide a direct 1-2 sentence overview (under {max_summary_words} words) "
    "of the documentation page below.\n"
    "State ONLY what main concepts, components, or procedures are documented.\n"
    "INTERNAL STYLE: flat, factual, no introductory fluff.\n"
    "If the page lacks substantive content beyond navigation links, headers, "
    "or index listings, return exactly: NO_CONTENT_TO_SUMMARIZE\n\n" + SYSTEM_BLOCK_SEPARATOR + "Title: {title}\n"
    "Content:\n{text}\n\n"
    "Summary:"
)

register_fallback("chunk-enrichment-summary", _SUMMARY_PROMPT)

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

_TRANSIENT_CATEGORIES = frozenset(
    {
        ProviderErrorCategory.RETRYABLE,
        ProviderErrorCategory.RATE_LIMITED,
        ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
        ProviderErrorCategory.QUOTA_EXCEEDED,
    }
)


class ContextSummarizer(Protocol):
    async def summarize(self, document: ParsedDocument) -> str: ...


class LLMContextSummarizer:
    """Generates a concise document summary for contextual chunk enrichment.

    Failures are fail-open: a page that cannot be summarised is still indexed
    without context.  Transient provider errors are retried a bounded number
    of times; permanent errors skip straight to the failure path.  After the
    retry budget is exhausted the URL is handed to ``failure_recorder`` (when
    provided) so a later re-enrichment pass can pick it up.
    """

    def __init__(
        self,
        llm_client,
        max_summary_words: int = 50,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
        failure_recorder: Callable[[ParsedDocument], Awaitable[None]] | None = None,
        telemetry=None,
    ) -> None:
        self._llm_client = llm_client
        self._max_summary_words = max_summary_words
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._failure_recorder = failure_recorder
        self._telemetry = telemetry

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        category = getattr(exc, "category", None)
        if category is not None:
            return category in _TRANSIENT_CATEGORIES
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return status_code in (408, 429) or status_code >= 500
        return True

    async def summarize(self, document: ParsedDocument) -> str:
        prompt = get_langfuse_prompt("chunk-enrichment-summary").compile(
            max_summary_words=self._max_summary_words,
            title=document.title,
            text=document.text[:3000],
        )
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                gen = None
                if self._telemetry:
                    gen = self._telemetry.start_observation(
                        name="enrichment-summarize",
                        as_type="generation",
                        model=getattr(self._llm_client, "model", None),
                        input=prompt,
                    )
                result = await self._llm_client.generate(prompt)
                if gen:
                    gen.update(output=result)
                    gen.end()
                return self._clean_summary(result)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries and self._is_transient(exc):
                    await asyncio.sleep(self._retry_backoff_seconds * (attempt + 1))
                    continue
                break
        logger.warning(
            "Context summarisation failed for title '%s': %s",
            document.title,
            last_error,
        )
        if self._failure_recorder is not None:
            try:
                await self._failure_recorder(document)
            except Exception:
                logger.exception("Context summarisation failure recorder error for url=%s", document.url)
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
        telemetry=None,
    ) -> None:
        self._summarizer = summarizer
        self._enabled = enabled
        self._batch_size = batch_size
        self._telemetry = telemetry

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
