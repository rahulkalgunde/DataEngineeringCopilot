"""Intelligent context assembly for RAG answer generation.

This module handles:
1. Semantic deduplication of overlapping chunks
2. Smart truncation respecting max_context_chars
3. Source citations in context
4. Optimal chunk ordering and composition
"""

import logging

from data_engineering_copilot.domain.models import RetrievedChunk

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assembles high-quality context from retrieved chunks for LLM prompting.

    Handles deduplication, truncation, and source attribution.
    """

    def __init__(self, max_context_chars: int):
        """Initialize context assembler.

        Args:
            max_context_chars: Maximum characters allowed in final context
        """
        self.max_context_chars = max_context_chars

    def assemble(self, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        """Assemble context from chunks with deduplication and truncation.

        Args:
            chunks: List of retrieved chunks, already sorted by confidence

        Returns:
            Tuple of (context_string, list_of_source_names)
        """
        if not chunks:
            return "", []

        # Step 1: Deduplicate semantically similar chunks
        deduped = self._deduplicate_chunks(chunks)
        logger.info("Deduplication: %d chunks → %d chunks", len(chunks), len(deduped))

        # Step 2: Build context with truncation
        context_lines = []
        source_names = []
        current_length = 0

        for chunk in deduped:
            source = chunk.chunk.source_name
            text = chunk.chunk.text
            section_header = chunk.chunk.section_header

            # Format: "Source: [Title > Section] [Text]" when section header exists
            formatted = f"[{source} > {section_header}] {text}" if section_header else f"[{source}] {text}"

            # Check if adding this chunk would exceed limit
            new_length = current_length + len(formatted) + 2  # +2 for newlines

            if new_length > self.max_context_chars and context_lines:
                # We've reached the limit; truncate here
                logger.info(
                    "Context truncated at chunk boundary: %.0f/%.0f chars", current_length, self.max_context_chars
                )
                break

            context_lines.append(formatted)
            source_names.append(source)
            current_length = new_length

        context = "\n".join(context_lines)

        logger.info(
            "Context assembled: %d chunks, %d chars, sources=%s",
            len(context_lines),
            len(context),
            list(set(source_names)),
        )

        return context, source_names

    def _deduplicate_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove semantically similar chunks using overlap detection.

        A simple heuristic: if two chunks share >70% of their words, deduplicate.
        Keeps the first (highest confidence) version.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Deduplicated list of chunks
        """
        if len(chunks) <= 1:
            return chunks

        deduped = [chunks[0]]

        for current_chunk in chunks[1:]:
            # Check if current chunk is similar to any in deduped
            is_duplicate = False

            for existing_chunk in deduped:
                similarity = self._text_overlap_ratio(existing_chunk.chunk.text, current_chunk.chunk.text)

                if similarity > 0.70:
                    is_duplicate = True
                    logger.debug("Deduplication: removed chunk (%.0f%% overlap with existing)", similarity * 100)
                    break

            if not is_duplicate:
                deduped.append(current_chunk)

        return deduped

    def _text_overlap_ratio(self, text1: str, text2: str) -> float:
        """Compute overlap ratio between two texts using word overlap.

        Returns ratio in [0, 1] representing shared content.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Overlap ratio (0 = no overlap, 1 = identical)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Remove common filler words to avoid false positives
        filler = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "is", "are"}
        words1 -= filler
        words2 -= filler

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
