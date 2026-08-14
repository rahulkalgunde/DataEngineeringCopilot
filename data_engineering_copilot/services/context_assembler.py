"""Intelligent context assembly for RAG answer generation.

This module handles:
1. Semantic deduplication of overlapping chunks
2. Source-coverage packing: budget-aware selection that guarantees at least one
   chunk per distinct source URL before filling the remaining budget by rank,
   capped at ``max_chunks_per_source`` chunks per URL (reference architecture:
   cross-encoder rerank, then "at most N chunks per document", then select the
   final evidence set)
3. Source citations in context
4. Lost-in-the-middle mitigation applied only to the *selected* set

Every indexed segment must already satisfy the per-segment item limit
(6,000 characters by default, enforced at indexing time by the lossless
token-budget splitter). Assembly therefore never skips a chunk *because* it is
oversized: an oversized segment is an invariant violation. When the total
context budget is exhausted, lower-ranked segments are dropped and recorded in
provenance with reason ``dropped_due_total_context_budget``. Segments excluded
because their source URL already contributed ``max_chunks_per_source`` chunks
are recorded with reason ``dropped_due_per_source_cap``.
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

    def __init__(
        self,
        max_context_chars: int,
        item_limit_chars: int = DEFAULT_ITEM_LIMIT_CHARS,
        max_chunks_per_source: int = 2,
    ):
        """Initialize context assembler.

        Args:
            max_context_chars: Maximum characters allowed in final context
            item_limit_chars: Per-segment item limit. An indexed segment whose
                text exceeds this limit raises ``ContextAssemblerError``.
            max_chunks_per_source: Maximum number of chunks kept per distinct
                source URL after the coverage guarantee. Reference architecture
                recommends "at most N chunks per document" as a diversity cap.
        """
        self.max_context_chars = max_context_chars
        self.item_limit_chars = item_limit_chars
        self.max_chunks_per_source = max_chunks_per_source

    def assemble(
        self,
        chunks: list[RetrievedChunk],
        mitigate_lost_in_middle: bool = True,
        deduplicate: bool = True,
    ) -> tuple[str, list[str], list[dict[str, object]]]:
        """Assemble context with source-coverage packing and budget-aware drop.

        Selection runs in rank order (chunks are assumed sorted by confidence):

        1. Coverage pass — the first (highest-ranked) chunk of every distinct
           source URL is guaranteed a slot, as long as the budget allows.
        2. Depth pass — remaining chunks are added by rank, capped at
           ``max_chunks_per_source`` chunks per URL, until the total budget is
           exhausted.

        The lost-in-the-middle mitigation is applied only to the *selected*
        set, so budget drops always remove the lowest-ranked segments.

        Args:
            chunks: List of retrieved chunks, already sorted by confidence
            mitigate_lost_in_middle: When True, reorders the selected chunks so
                the most relevant ones appear at the beginning AND end of the
                context, reducing the "lost in the middle" effect.
            deduplicate: When True, deduplicate chunks before assembly.
                Set to False if upstream (ContextCompressor) already deduplicated.

        Returns:
            Tuple of ``(context_string, list_of_source_names, dropped_records)``
            where ``dropped_records`` lists every segment excluded from the
            final context, each with reason ``dropped_due_total_context_budget``
            (budget exhausted) or ``dropped_due_per_source_cap`` (source URL
            already at ``max_chunks_per_source``) and segment provenance (rank,
            chunk_id, url, segment_index, parent_content_hash).

        Raises:
            ContextAssemblerError: when an indexed segment exceeds
                ``item_limit_chars`` (invariant violation).
        """
        if not chunks:
            return "", [], []

        deduped = self._deduplicate_chunks(chunks) if deduplicate else chunks
        logger.info("Deduplication: %d chunks → %d chunks", len(chunks), len(deduped))

        # Step 1: Source-coverage budget selection in rank order.
        selected, dropped_records = self._select_with_source_coverage(deduped)

        # Step 2: Lost-in-the-middle mitigation on the selected set only.
        if mitigate_lost_in_middle and len(selected) > 3:
            selected = self._reorder_lost_in_middle(selected)

        # Step 3: Build the final context strings with sequential ids.
        context_lines = []
        source_names = []
        for i, chunk in enumerate(selected, start=1):
            formatted = self._format_chunk(chunk, i)
            context_lines.append(formatted)
            source_names.append(chunk.chunk.source_name)

        context = "\n".join(context_lines)

        logger.info(
            "Context assembled: %d chunks, %d chars, sources=%s, dropped=%d",
            len(context_lines),
            len(context),
            list(set(source_names)),
            len(dropped_records),
        )

        return context, source_names, dropped_records

    def _select_with_source_coverage(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], list[dict[str, object]]]:
        """Select chunks under budget with a per-source coverage guarantee.

        Two passes over the rank-ordered chunks:

        1. Coverage pass — the highest-ranked chunk of every distinct source
           URL is placed first (evidence-set selection), guaranteeing maximal
           cross-source coverage before any source can be deepened.
        2. Depth pass — remaining chunks are added by rank, capped at
           ``max_chunks_per_source`` chunks per URL, until the total budget is
           exhausted.

        Returns:
            Tuple of ``(selected_chunks, dropped_records)``.
        """
        selected: list[RetrievedChunk] = []
        dropped_records: list[dict[str, object]] = []
        url_counts: dict[str, int] = {}
        selected_ids: set[str] = set()
        current_length = 0

        def try_place(chunk: RetrievedChunk, rank: int, reason: str) -> bool:
            nonlocal current_length
            formatted = self._format_chunk(chunk, len(selected) + 1)
            new_length = current_length + len(formatted) + 2  # +2 for newlines
            # The very first chunk is always placed even when it exceeds the
            # total budget, so a single over-budget segment still produces a
            # usable context.
            if new_length > self.max_context_chars and selected:
                dropped_records.append(self._drop_record(chunk, rank, reason))
                return False
            selected.append(chunk)
            selected_ids.add(chunk.chunk.chunk_id)
            url_counts[chunk.chunk.url] = url_counts.get(chunk.chunk.url, 0) + 1
            current_length = new_length
            return True

        # Pass 1: coverage — one slot per distinct source URL.
        covered_urls: set[str] = set()
        for rank, chunk in enumerate(chunks):
            url = chunk.chunk.url
            if url in covered_urls:
                continue
            if try_place(chunk, rank, "dropped_due_total_context_budget"):
                covered_urls.add(url)

        # Pass 2: depth — fill remaining budget by rank, capped per URL.
        for rank, chunk in enumerate(chunks):
            if chunk.chunk.chunk_id in selected_ids:
                continue
            url = chunk.chunk.url
            if url not in covered_urls:
                continue
            if url_counts[url] >= self.max_chunks_per_source:
                dropped_records.append(self._drop_record(chunk, rank, "dropped_due_per_source_cap"))
                continue
            try_place(chunk, rank, "dropped_due_total_context_budget")

        return selected, dropped_records

    def _format_chunk(self, chunk: RetrievedChunk, doc_id: int) -> str:
        """Format a chunk as a context_doc block, enforcing the item limit."""
        text = chunk.chunk.text
        if len(text) > self.item_limit_chars:
            raise ContextAssemblerError(
                f"Indexed segment {chunk.chunk.chunk_id!r} is {len(text)} chars, "
                f"exceeding the item limit of {self.item_limit_chars}; "
                "a valid generation must never produce over-limit segments"
            )

        source = chunk.chunk.source_name
        section_header = chunk.chunk.section_header
        section_suffix = f" [{section_header}]" if section_header else ""
        return f'<context_doc id="{doc_id}" url="{chunk.chunk.url}">[{source}{section_suffix}]\n{text}\n</context_doc>'

    @staticmethod
    def _drop_record(chunk: RetrievedChunk, rank: int, reason: str) -> dict[str, object]:
        """Build a provenance record for a dropped segment."""
        return {
            "reason": reason,
            "rank": rank,
            "chunk_id": chunk.chunk.chunk_id,
            "url": chunk.chunk.url,
            "segment_index": chunk.chunk.segment_index,
            "parent_content_hash": chunk.chunk.parent_content_hash,
        }

    @staticmethod
    def _reorder_lost_in_middle(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Reorder chunks so the most relevant appear at both ends."""
        rearranged: list[RetrievedChunk] = []
        left, right = 0, len(chunks) - 1
        turn_left = True
        while left <= right:
            if left == right:
                rearranged.append(chunks[left])
                break
            if turn_left:
                rearranged.append(chunks[left])
                left += 1
            else:
                rearranged.append(chunks[right])
                right -= 1
            turn_left = not turn_left
        return rearranged

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
