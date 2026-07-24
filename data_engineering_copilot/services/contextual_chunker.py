"""Contextual chunk enrichment — prepend document context to each chunk.

Based on Anthropic's Contextual Retrieval approach:
https://www.anthropic.com/news/contextual-retrieval

Each chunk gets a brief contextual prefix that helps the retriever
understand what document the chunk came from, even when the chunk
is retrieved in isolation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_engineering_copilot.domain.models import DocumentChunk
    from data_engineering_copilot.domain.protocols import LLMClientProtocol

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = (
    "Here is a chunk from a document titled '{title}' (source: {source_name}):\n\n"
    "{chunk_text}\n\n"
    "Provide a 1-2 sentence summary of what this chunk is about, "
    "including the document title and any relevant context. "
    "This summary will be prepended to the chunk for retrieval purposes."
)


class ContextualChunkEnricher:
    """Enriches chunks with document-level context for better retrieval.

    Each chunk gets a brief contextual prefix that helps the retriever
    understand what document the chunk came from, even when the chunk
    is retrieved in isolation.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        enabled: bool = True,
        batch_size: int = 10,
    ) -> None:
        self._llm = llm_client
        self._enabled = enabled
        self._batch_size = batch_size

    async def enrich_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Add contextual prefixes to chunks using the LLM.

        Args:
            chunks: List of DocumentChunk objects to enrich.

        Returns:
            New list of chunks with enriched text (original + context prefix).
        """
        if not self._enabled or not chunks:
            return chunks

        enriched: list[DocumentChunk] = []
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            tasks = [self._enrich_single(chunk) for chunk in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for chunk, result in zip(batch, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Context enrichment failed for chunk %s: %s", chunk.chunk_id, result)
                    enriched.append(chunk)
                else:
                    enriched.append(result)

        logger.info("Enriched %d/%d chunks with contextual prefixes", len(enriched), len(chunks))
        return enriched

    async def _enrich_single(self, chunk: DocumentChunk) -> DocumentChunk:
        """Enrich a single chunk with document context."""
        prompt = _CONTEXT_PROMPT.format(
            title=chunk.title,
            source_name=chunk.source_name,
            chunk_text=chunk.text[:500],
        )
        context = await self._llm.generate(prompt)
        context = context.strip().rstrip(".")
        enriched_text = f"[Context: {context}]\n\n{chunk.text}"
        return replace(chunk, text=enriched_text)
