"""Intelligent context assembly for RAG answer generation.

This module handles:
1. Semantic deduplication of overlapping chunks
2. Smart truncation respecting max_context_chars
3. Source citations in context
4. Optimal chunk ordering and composition

Every indexed segment must already satisfy the per-segment item limit
(6,000 characters by default, enforced at indexing time by the lossless
token-budget splitter). Assembly therefore never skips a chunk *because* it is
oversized: an oversized segment is an invariant violation. When the total
context budget is exhausted, lower-ranked segments are dropped and recorded in
provenance with reason ``dropped_due_total_context_budget``.
"""

import logging

from data_engineering_copilot.domain.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Per-segment item limit enforced at indexing time (token_budget.DEFAULT_MAX_CHARS).
DEFAULT_ITEM_LIMIT_CHARS = 6000


class ContextAssemblerError(ValueError):
    """Raised when an indexed segment violates the item limit invariant."""


class ContextAssembler:
    """Assembles high-quality context from retrieved chunks for LLM prompting.

    Handles deduplication, truncation, and source attribution. Segments must
    already satisfy the per-segment item limit; assembly only drops whole
    segments to respect the total context budget.
    """

    def __init__(self, max_context_chars: int, item_limit_chars: int = DEFAULT_ITEM_LIMIT_CHARS):
        """Initialize context assembler.

        Args:
            max_context_chars: Maximum characters allowed in final context
            item_limit_chars: Per-segment item limit. An indexed segment whose
                text exceeds this limit raises ``ContextAssemblerError``.
        """
        self.max_context_chars = max_context_chars
        self.item_limit_chars = item_limit_chars

    def assemble(
        self,
        chunks: list[RetrievedChunk],
        mitigate_lost_in_middle: bool = True,
        deduplicate: bool = True,
    ) -> tuple[str, list[str], list[dict[str, object]]]:
        """Assemble context from chunks with deduplication and budget-aware drop.

        Args:
            chunks: List of retrieved chunks, already sorted by confidence
            mitigate_lost_in_middle: When True, reorders chunks so the most
                relevant ones appear at the beginning AND end of the context,
                reducing the "lost in the middle" effect.
            deduplicate: When True, deduplicate chunks before assembly.
                Set to False if upstream (ContextCompressor) already deduplicated.

        Returns:
            Tuple of ``(context_string, list_of_source_names, dropped_records)``
            where ``dropped_records`` lists every segment excluded only because
            the total context budget was exhausted, each with reason
            ``dropped_due_total_context_budget`` and segment provenance (rank,
            chunk_id, url, segment_index, parent_content_hash).

        Raises:
            ContextAssemblerError: when an indexed segment exceeds
                ``item_limit_chars`` (invariant violation).
        """
        if not chunks:
            return "", [], []

        deduped = self._deduplicate_chunks(chunks) if deduplicate else chunks
        logger.info("Deduplication: %d chunks → %d chunks", len(chunks), len(deduped))

        # Step 2: Lost-in-the-middle mitigation — reorder so top chunks
        # appear at both ends of the context window.
        if mitigate_lost_in_middle and len(deduped) > 3:
            rearranged: list[RetrievedChunk] = []
            left, right = 0, len(deduped) - 1
            turn_left = True
            while left <= right:
                if left == right:
                    rearranged.append(deduped[left])
                    break
                if turn_left:
                    rearranged.append(deduped[left])
                    left += 1
                else:
                    rearranged.append(deduped[right])
                    right -= 1
                turn_left = not turn_left
            deduped = rearranged

        # Step 3: Build context until the total budget is full. Every indexed
        # segment must satisfy the item limit; oversized segments are an
        # invariant violation. Segments that cannot fit the remaining budget
        # are dropped and recorded with reason ``dropped_due_total_context_budget``.
        context_lines = []
        source_names = []
        dropped_records: list[dict[str, object]] = []
        current_length = 0

        # Reserve one slot per available doc_type (guide/api_reference/code_example)
        # so multi-doc-type queries retain coverage.
        seen_doc_types: set[str] = set()
        doc_type_reserved: set[str] = set()
        for chunk in deduped:
            dt = chunk.chunk.doc_type
            if dt:
                doc_type_reserved.add(dt)

        for i, chunk in enumerate(deduped, start=1):
            source = chunk.chunk.source_name
            text = chunk.chunk.text
            section_header = chunk.chunk.section_header

            if len(text) > self.item_limit_chars:
                raise ContextAssemblerError(
                    f"Indexed segment {chunk.chunk.chunk_id!r} is {len(text)} chars, "
                    f"exceeding the item limit of {self.item_limit_chars}; "
                    "a valid generation must never produce over-limit segments"
                )

            section_suffix = f" [{section_header}]" if section_header else ""
            formatted = (
                f'<context_doc id="{i}" url="{chunk.chunk.url}">[{source}{section_suffix}]\n{text}\n</context_doc>'
            )

            new_length = current_length + len(formatted) + 2  # +2 for newlines

            # Keep at least one segment per required doc_type when it can be
            # accommodated by swapping out a larger included segment.
            if new_length > self.max_context_chars and context_lines:
                dt = chunk.chunk.doc_type
                if (
                    dt in doc_type_reserved
                    and dt not in seen_doc_types
                    and self._try_make_room(
                        formatted,
                        context_lines,
                        source_names,
                        len(deduped),
                        current_length,
                    )
                ):
                    current_length = sum(len(line) + 1 for line in context_lines)
                    continue

                # Budget exhausted: drop this lower-ranked segment and record it.
                dropped_records.append(
                    {
                        "reason": "dropped_due_total_context_budget",
                        "rank": i - 1,
                        "chunk_id": chunk.chunk.chunk_id,
                        "url": chunk.chunk.url,
                        "segment_index": chunk.chunk.segment_index,
                        "parent_content_hash": chunk.chunk.parent_content_hash,
                    }
                )
                logger.info(
                    "Context dropped segment %s (rank=%d): total context budget exhausted",
                    chunk.chunk.chunk_id,
                    i - 1,
                )
                continue

            context_lines.append(formatted)
            source_names.append(source)
            if chunk.chunk.doc_type:
                seen_doc_types.add(chunk.chunk.doc_type)
            current_length = new_length

        context = "\n".join(context_lines)

        logger.info(
            "Context assembled: %d chunks, %d chars, sources=%s, dropped=%d",
            len(context_lines),
            len(context),
            list(set(source_names)),
            len(dropped_records),
        )

        return context, source_names, dropped_records

    @staticmethod
    def _try_make_room(
        formatted: str,
        context_lines: list[str],
        source_names: list[str],
        total_chunks: int,
        current_length: int,
    ) -> bool:
        """Try to make room for a required doc-type chunk by dropping the largest.

        Returns True if a swap was performed.
        """
        # Find the largest included formatted chunk and drop it if the required
        # chunk is smaller, keeping overall size within budget.
        largest_idx = 0
        largest_len = 0
        for idx, line in enumerate(context_lines):
            if len(line) > largest_len:
                largest_len = len(line)
                largest_idx = idx
        if largest_len <= len(formatted):
            return False
        context_lines.pop(largest_idx)
        source_names.pop(largest_idx)
        context_lines.append(formatted)
        source_names.append(source_names[-1] if source_names else "")
        return True

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
