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


def _merge_blank_pieces(pieces: list[str]) -> list[str]:
    """Fold whitespace-only pieces into a content neighbor.

    Lossless splitting can emit a piece that is only the whitespace between
    two blocks (e.g. the ``"\\n\\n"`` paragraph separator preceding an atomic
    code fence). Embedding providers reject blank input strings with HTTP 400
    (``input must not be blank or empty``), and a blank chunk carries no
    retrieval value, so fold such pieces into an adjacent content piece.

    Every character is preserved and ordering is unchanged, so
    ``"".join(merged) == "".join(pieces)``: lossless reconstruction still
    holds and segment-budget validation is unaffected.
    """
    merged: list[str] = []
    pending_blank = ""
    for piece in pieces:
        if not piece.strip():
            pending_blank += piece
            continue
        if pending_blank:
            merged.append(pending_blank + piece)
            pending_blank = ""
        else:
            merged.append(piece)
    if pending_blank:
        if merged:
            merged[-1] += pending_blank
        else:
            merged.append(pending_blank)
    return merged


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
            pieces = split_text_losslessly(text, max_tokens=budget, max_chars=budget * 4)
        except ValueError:
            continue
        return _merge_blank_pieces(pieces)
    pieces = split_text_losslessly(text, max_tokens=parent_max_tokens, max_chars=parent_max_tokens * 4)
    return _merge_blank_pieces(pieces)


def hierarchical_chunk(
    chunk: DocumentChunk,
    parent_max_tokens: int = 1024,
    child_max_tokens: int = 256,
    parent_offset_start: int = 0,
    parent_offset_end: int = 0,
) -> list[DocumentChunk]:
    """Split *chunk* into a parent chunk plus child sub-splits.

    Returns a list where the first element is the parent chunk and the rest are
    its children. Each child carries ``parent_chunk_id`` set to the parent's
    ``chunk_id``; the parent carries ``parent_chunk_id=""``. Chunks already
    within the child budget are returned unchanged (they are their own parent).

    Children are produced with ``split_text_with_offsets``, so
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
                start_offset=parent_offset_start,
                end_offset=parent_offset_end,
                parent_content_hash="",
                segment_index=0,
                segment_total=1,
                parent_chunk_id="",
            )
        ]

    parent_texts = _merge_blank_pieces(
        split_text_losslessly(
            chunk.text,
            max_tokens=parent_max_tokens,
            max_chars=parent_max_tokens * 4,
        )
    )

    result: list[DocumentChunk] = []
    cursor = 0
    for p_idx, parent_text in enumerate(parent_texts):
        start = chunk.text.find(parent_text, cursor)
        if start == -1:
            start = cursor
        end = start + len(parent_text)
        cursor = end
        abs_parent_start = parent_offset_start + start
        abs_parent_end = parent_offset_start + end
        parent_id = f"{chunk.chunk_id}:p{p_idx}"
        # Hash the stripped text so children's shared parent_content_hash matches
        # _validate_segment_budgets' lossless-reconstruction check (which strips).
        parent_hash = hashlib.sha256(parent_text.strip().encode("utf-8")).hexdigest()
        result.append(
            replace(
                chunk,
                chunk_id=parent_id,
                text=parent_text,
                start_offset=abs_parent_start,
                end_offset=abs_parent_end,
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

        # Mirror _split_children: try the child budget, fall back to larger
        # budgets so rare atomic pieces (a single long line) don't break
        # splitting. Returns lossless, blank-merged child pieces.
        child_pieces: list[str] = []
        for budget in (child_max_tokens, parent_max_tokens, DEFAULT_MAX_TOKENS):
            try:
                child_pieces = _merge_blank_pieces(
                    split_text_losslessly(parent_text, max_tokens=budget, max_chars=budget * 4)
                )
                break
            except ValueError:
                continue

        child_segments: list[tuple[str, int, int]] = []
        cursor = 0
        for piece in child_pieces:
            start = parent_text.find(piece, cursor)
            if start == -1:
                start = cursor
            end = start + len(piece)
            cursor = end
            child_segments.append((piece, start, end))

        for c_idx, (child_text, rel_start, rel_end) in enumerate(child_segments):
            result.append(
                replace(
                    chunk,
                    chunk_id=f"{parent_id}:c{c_idx}",
                    text=child_text,
                    start_offset=abs_parent_start + rel_start,
                    end_offset=abs_parent_start + rel_end,
                    content_hash=hashlib.sha256(child_text.encode("utf-8")).hexdigest(),
                    word_count=len(child_text.split()),
                    parent_content_hash=parent_hash,
                    segment_index=c_idx,
                    segment_total=len(child_segments),
                    token_count=count_tokens(child_text),
                    character_count=len(child_text),
                    parent_chunk_id=parent_id,
                )
            )
    return result
