"""Hierarchical (parent-child) chunking.

Parent chunks carry full section context; child chunks are smaller sub-splits
embedded for precise retrieval. At query time the store substitutes a matched
child's text with its parent's text so the LLM receives broader context.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.infrastructure.token_budget import (
    DEFAULT_MAX_TOKENS,
    count_tokens,
    split_text_losslessly,
)


def _split_children(
    text: str,
    child_max_tokens: int,
    parent_max_tokens: int,
) -> list[str]:
    """Split *text* into child sub-splits, falling back to larger budgets.

    The child token budget is a retrieval-quality target, not a hard
    embedding limit. Rare atomic pieces (a single long code line or URL
    longer than ``child_max_tokens * 4`` characters) cannot be split
    further without losing characters; failing the whole build over them
    would drop otherwise-good documents. Retry at the parent budget, then
    at the provider hard cap, before giving up.
    """
    for budget in (child_max_tokens, parent_max_tokens, DEFAULT_MAX_TOKENS):
        try:
            return split_text_losslessly(text, max_tokens=budget, max_chars=budget * 4)
        except ValueError:
            continue
    return split_text_losslessly(text, max_tokens=parent_max_tokens, max_chars=parent_max_tokens * 4)


def hierarchical_chunk(
    chunk: DocumentChunk,
    parent_max_tokens: int = 1024,
    child_max_tokens: int = 256,
) -> list[DocumentChunk]:
    """Split *chunk* into a parent chunk plus child sub-splits.

    Returns a list where the first element is the parent chunk and the rest are
    its children. Each child carries ``parent_chunk_id`` set to the parent's
    ``chunk_id``; the parent carries ``parent_chunk_id=""``. Chunks already
    within the child budget are returned unchanged (they are their own parent).

    Children are produced with ``split_text_losslessly``, so
    ``"".join(child_texts)`` reconstructs the parent text exactly and every
    child satisfies the child token budget. For segment-budget validation
    compatibility the parent is a complete unit (empty ``parent_content_hash``,
    ``segment_index=0``, ``segment_total=1``) while the children share the
    parent text hash as ``parent_content_hash`` and contiguous indices.
    """
    if count_tokens(chunk.text) <= child_max_tokens:
        # Within the child budget: the chunk is its own parent. Reset segment
        # metadata so it validates as a complete single-segment unit and never
        # collides with hierarchical sibling groups.
        return [
            replace(
                chunk,
                parent_content_hash="",
                segment_index=0,
                segment_total=1,
                parent_chunk_id="",
            )
        ]

    parent_texts = split_text_losslessly(
        chunk.text,
        max_tokens=parent_max_tokens,
        max_chars=parent_max_tokens * 4,
    )

    result: list[DocumentChunk] = []
    for p_idx, parent_text in enumerate(parent_texts):
        parent_id = f"{chunk.chunk_id}:p{p_idx}"
        # Hash the stripped text so children's shared parent_content_hash matches
        # _validate_segment_budgets' lossless-reconstruction check (which strips).
        parent_hash = hashlib.sha256(parent_text.strip().encode("utf-8")).hexdigest()
        result.append(
            replace(
                chunk,
                chunk_id=parent_id,
                text=parent_text,
                content_hash=parent_hash,
                word_count=len(parent_text.split()),
                parent_content_hash="",
                segment_index=0,
                segment_total=1,
                token_count=count_tokens(parent_text),
                character_count=len(parent_text),
                parent_chunk_id="",
            )
        )

        child_texts = _split_children(parent_text, child_max_tokens, parent_max_tokens)
        for c_idx, child_text in enumerate(child_texts):
            result.append(
                replace(
                    chunk,
                    chunk_id=f"{parent_id}:c{c_idx}",
                    text=child_text,
                    content_hash=hashlib.sha256(child_text.encode("utf-8")).hexdigest(),
                    word_count=len(child_text.split()),
                    parent_content_hash=parent_hash,
                    segment_index=c_idx,
                    segment_total=len(child_texts),
                    token_count=count_tokens(child_text),
                    character_count=len(child_text),
                    parent_chunk_id=parent_id,
                )
            )
    return result
