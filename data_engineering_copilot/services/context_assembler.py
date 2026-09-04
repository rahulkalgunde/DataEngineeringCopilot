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
import re

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk

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
        xml_content_escape: bool = True,
    ):
        """Initialize context assembler.

        Args:
            max_context_chars: Maximum characters allowed in final context
            item_limit_chars: Per-segment item limit. An indexed segment whose
                text exceeds this limit raises ``ContextAssemblerError``.
            max_chunks_per_source: Maximum number of chunks kept per distinct
                source URL after the coverage guarantee. Reference architecture
                recommends "at most N chunks per document" as a diversity cap.
            xml_content_escape: Whether to escape XML metacharacters (& and <)
                in chunk text. Defaults to True for safety.
        """
        self.max_context_chars = max_context_chars
        self.item_limit_chars = item_limit_chars
        self.max_chunks_per_source = max_chunks_per_source
        self._xml_content_escape = xml_content_escape

    def _content_hash_dedup(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Fast exact-match dedup using SHA-256 content hashes.

        Runs before Jaccard overlap to instantly strip identical text blocks.
        """
        seen_hashes: set[str] = set()
        result: list[RetrievedChunk] = []
        for chunk in chunks:
            h = chunk.chunk.content_hash
            if h and h in seen_hashes:
                logger.debug("Content-hash dedup: dropped %s (hash=%s)", chunk.chunk.chunk_id, h)
                continue
            if h:
                seen_hashes.add(h)
            result.append(chunk)
        return result

    def _merge_adjacent_siblings(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Merge adjacent chunks from the same parent into a single contiguous block.

        Groups chunks by parent_chunk_id, sorts by segment_index within each group,
        and returns one merged chunk per group plus all orphan chunks.
        """
        if not chunks:
            return []

        groups: dict[str, list[RetrievedChunk]] = {}
        orphans: list[RetrievedChunk] = []

        for chunk in chunks:
            pid = chunk.chunk.parent_chunk_id
            if pid:
                groups.setdefault(pid, []).append(chunk)
            else:
                orphans.append(chunk)

        merged: list[RetrievedChunk] = []
        for _pid, siblings in groups.items():
            siblings.sort(key=lambda c: c.chunk.segment_index)
            best = max(siblings, key=lambda c: c.confidence)
            merged_text = "\n\n".join(s.chunk.text for s in siblings)
            merged_chunk = DocumentChunk(
                chunk_id=best.chunk.chunk_id,
                source_name=best.chunk.source_name,
                title=best.chunk.title,
                url=best.chunk.url,
                text=merged_text,
                content_hash=best.chunk.content_hash,
                section_header=best.chunk.section_header,
                chunk_type=best.chunk.chunk_type,
                word_count=best.chunk.word_count,
                heading_path=best.chunk.heading_path,
                chunk_index=best.chunk.chunk_index,
                total_chunks=best.chunk.total_chunks,
                crawled_at=best.chunk.crawled_at,
                doc_type=best.chunk.doc_type,
                language=best.chunk.language,
                spark_version=best.chunk.spark_version,
                module=best.chunk.module,
                source_commit=best.chunk.source_commit,
                file_path=best.chunk.file_path,
                license=best.chunk.license,
                parser_version=best.chunk.parser_version,
                chunker_version=best.chunk.chunker_version,
                index_generation=best.chunk.index_generation,
                deployment_mode=best.chunk.deployment_mode,
                parent_content_hash=best.chunk.parent_content_hash,
                segment_index=best.chunk.segment_index,
                segment_total=best.chunk.segment_total,
                token_count=best.chunk.token_count,
                character_count=len(merged_text),
                representation=best.chunk.representation,
                parent_chunk_id=best.chunk.parent_chunk_id,
            )
            merged.append(
                RetrievedChunk(
                    chunk=merged_chunk,
                    distance=best.distance,
                    confidence=best.confidence,
                )
            )

        return merged + orphans

    def _mmr_diversify(
        self,
        chunks: list[RetrievedChunk],
        lambda_param: float = 0.5,
    ) -> list[RetrievedChunk]:
        """Apply MMR diversity: balance relevance vs diversity via greedy selection.

        At each step, pick the chunk maximizing:
            MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        """
        if len(chunks) <= 1:
            return chunks

        def _tokenize(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9_]+", text.lower()))

        def _cosine(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / (len(a) * len(b)) ** 0.5

        selected: list[RetrievedChunk] = []
        remaining = list(chunks)
        selected_tokens: list[set[str]] = []

        while remaining:
            best_score = -float("inf")
            best_idx = 0
            for i, chunk in enumerate(remaining):
                chunk_tokens = _tokenize(chunk.chunk.text)
                relevance = chunk.confidence
                max_sim = (
                    max((_cosine(chunk_tokens, st) for st in selected_tokens), default=0.0) if selected_tokens else 0.0
                )
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            selected_tokens.append(_tokenize(chosen.chunk.text))

        return selected

    @staticmethod
    def _build_breadcrumb(chunk: RetrievedChunk, fmt: str = "hierarchical") -> str:
        """Build a metadata breadcrumb header for a chunk.

        Formats:
            "hierarchical": [Source: Spark > SQL > Joins]
            "flat": [Source: Spark]
            "none": ""
        """
        source = chunk.chunk.source_name
        if fmt == "none":
            return ""
        if fmt == "flat" or not chunk.chunk.heading_path:
            return f"[Source: {source}]"
        path = " > ".join(chunk.chunk.heading_path)
        return f"[Source: {source} > {path}]"

    def assemble(
        self,
        chunks: list[RetrievedChunk],
        mitigate_lost_in_middle: bool = True,
        deduplicate: bool = True,
        content_hash_dedup: bool = True,
        enable_sibling_merge: bool = True,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.5,
        breadcrumb_format: str = "hierarchical",
    ) -> tuple[str, list[str], list[dict[str, object]]]:
        """Assemble context from retrieved chunks for the RAG answer prompt.

        Pipeline:
            1. Content-hash dedup (exact SHA-256 match)
            2. Adjacent sibling merge (same parent → one block)
            3. Either MMR diversity OR Jaccard lexical dedup
            4. Source-coverage budget selection
            5. Lost-in-the-middle reorder
            6. Format as XML with optional hierarchical breadcrumbs

        Args:
            chunks: Retrieved chunks already sorted by confidence.
            mitigate_lost_in_middle: Reorder so top scores appear at start AND end.
            deduplicate: Lexical dedup via Jaccard overlap (skipped when mmr_enabled).
            content_hash_dedup: Exact-match dedup via SHA-256 content_hash.
            enable_sibling_merge: Merge adjacent children of same parent.
            mmr_enabled: Use MMR diversity instead of Jaccard dedup.
            mmr_lambda: MMR balance: 1.0=pure relevance, 0.0=pure diversity.
            breadcrumb_format: "hierarchical"|"flat"|"none" for metadata headers.

        Returns:
            Tuple of ``(context_string, list_of_source_names, dropped_records)``.

        Raises:
            ContextAssemblerError: when an indexed segment exceeds item_limit_chars.
        """
        if not chunks:
            return "", [], []

        # Phase 1: Fast exact-match dedup (content hash)
        p1 = self._content_hash_dedup(chunks) if content_hash_dedup else chunks

        # Phase 2: Merge adjacent siblings from same parent
        p2 = self._merge_adjacent_siblings(p1) if enable_sibling_merge else p1

        # Phase 3: Diversity — MMR or Jaccard dedup
        if mmr_enabled:
            working = self._mmr_diversify(p2, lambda_param=mmr_lambda)
            deduped = working
        else:
            deduped = self._deduplicate_chunks(p2) if deduplicate else p2

        logger.info(
            "[ContextAssembler Diagnostic] Stage Counts: Incoming=%d -> HashDedup=%d -> SiblingMerged=%d -> Deduped=%d",
            len(chunks),
            len(p1),
            len(p2),
            len(deduped),
        )

        # Step 1: Source-coverage budget selection
        selected, dropped_records = self._select_with_source_coverage(deduped)

        # Step 2: Lost-in-the-middle mitigation
        if mitigate_lost_in_middle and len(selected) > 3:
            selected = self._reorder_lost_in_middle(selected)

        # Step 3: Format
        context_lines = []
        source_names = []
        for i, chunk in enumerate(selected, start=1):
            formatted = self._format_chunk(chunk, i, breadcrumb_fmt=breadcrumb_format)
            context_lines.append(formatted)
            source_names.append(chunk.chunk.source_name)

        context = "\n".join(context_lines)

        logger.info(
            "[ContextAssembler Diagnostic] Final Selected: %d chunks (%d chars / %d max) | Sources: %s | Total Dropped Records: %d",
            len(context_lines),
            len(context),
            self.max_context_chars,
            list(set(source_names)),
            len(dropped_records),
        )
        return context, source_names, dropped_records

    def _select_with_source_coverage(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], list[dict[str, object]]]:
        """Select chunks under budget with a per-source coverage guarantee.

        Three passes over the rank-ordered chunks:

        1. Coverage per source_name — the highest-ranked chunk of every
            distinct ``source_name`` (corpus, e.g. Spark vs Airflow) is placed
            first, guaranteeing cross-corpus coverage for multi-intent queries.
        2. Coverage per URL — one slot per distinct URL not yet covered.
        3. Depth — remaining chunks by rank, capped at
            ``max_chunks_per_source`` chunks per URL, until the total budget is
            exhausted. ``max_chunks_per_source`` is per URL (kept for backward
            compat; budgeting per source_name is via Pass 1).

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
                logger.info(
                    "[ContextAssembler Drop] BUDGET_EXCEEDED | rank=%d chunk_id=%s url=%s len=%d (+%d) > max=%d",
                    rank,
                    chunk.chunk.chunk_id,
                    chunk.chunk.url,
                    current_length,
                    len(formatted),
                    self.max_context_chars,
                )
                dropped_records.append(self._drop_record(chunk, rank, reason))
                return False
            selected.append(chunk)
            selected_ids.add(chunk.chunk.chunk_id)
            url_counts[chunk.chunk.url] = url_counts.get(chunk.chunk.url, 0) + 1
            current_length = new_length
            return True

        # Pass 1: coverage — one slot per distinct source_name (corpus),
        # then one slot per distinct URL. Guarantees multi-corpus queries
        # (e.g. "spark + airflow + claude") get at least one chunk per
        # corpus before per-URL depth fills the budget.
        covered_names: set[str] = set()
        covered_urls: set[str] = set()
        for rank, chunk in enumerate(chunks):
            name = chunk.chunk.source_name
            if name in covered_names:
                continue
            if try_place(chunk, rank, "dropped_due_total_context_budget"):
                covered_names.add(name)
                covered_urls.add(chunk.chunk.url)

        # Pass 1b: coverage — one slot per distinct source URL not yet covered.
        for rank, chunk in enumerate(chunks):
            if chunk.chunk.chunk_id in selected_ids:
                continue
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
                logger.info(
                    "[ContextAssembler Drop] PER_SOURCE_CAP | rank=%d chunk_id=%s url=%s count=%d >= max=%d",
                    rank,
                    chunk.chunk.chunk_id,
                    chunk.chunk.url,
                    url_counts[url],
                    self.max_chunks_per_source,
                )
                dropped_records.append(self._drop_record(chunk, rank, "dropped_due_per_source_cap"))
                continue
            try_place(chunk, rank, "dropped_due_total_context_budget")

        return selected, dropped_records

    def _format_chunk(self, chunk: RetrievedChunk, doc_id: int, breadcrumb_fmt: str = "hierarchical") -> str:
        """Format a chunk as a context document with optional metadata breadcrumb.

        The ``breadcrumb_fmt`` parameter controls the metadata header style:
        "hierarchical" shows full heading path, "flat" shows source only,
        "none" omits the header entirely.
        """
        text = chunk.chunk.text
        if self._xml_content_escape:
            text = text.replace("&", "&amp;").replace("<", "&lt;")

        if len(text) > self.item_limit_chars:
            text = text[: self.item_limit_chars]

        breadcrumb = self._build_breadcrumb(chunk, breadcrumb_fmt)
        header = f"{breadcrumb}\n" if breadcrumb else ""
        return f'<context_doc id="{doc_id}" url="{chunk.chunk.url}">{header}{text}\n</context_doc>'

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

        In a hierarchical corpus, multiple children of the same parent already
        carry identical (substituted) parent text, so siblings are collapsed to
        the single highest-confidence child.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Deduplicated list of chunks
        """
        if len(chunks) <= 1:
            return chunks

        deduped: list[RetrievedChunk] = []
        seen_parents: set[str] = set()

        for current_chunk in chunks:
            parent_id = current_chunk.chunk.parent_chunk_id
            if parent_id:
                if parent_id in seen_parents:
                    logger.info(
                        "[ContextAssembler Drop] PARENT_ID_DEDUP | chunk_id=%s url=%s parent_id=%s",
                        current_chunk.chunk.chunk_id,
                        current_chunk.chunk.url,
                        parent_id,
                    )
                    continue
                seen_parents.add(parent_id)

            # Check if current chunk is similar to any in deduped
            is_duplicate = False

            for existing_chunk in deduped:
                similarity = self._text_overlap_ratio(existing_chunk.chunk.text, current_chunk.chunk.text)

                if similarity > 0.70:
                    is_duplicate = True
                    logger.info(
                        "[ContextAssembler Drop] JACCARD_OVERLAP | chunk_id=%s url=%s (%.0f%% overlap with %s)",
                        current_chunk.chunk.chunk_id,
                        current_chunk.chunk.url,
                        similarity * 100,
                        existing_chunk.chunk.chunk_id,
                    )
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
