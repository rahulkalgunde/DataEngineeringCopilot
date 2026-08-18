"""Lossless token-budget splitting and validation.

The embedding provider has a hard input limit (3,800 tokens by default) and the
context assembler has a per-segment character limit (6,000 characters). Neither
limit may be satisfied by truncating content — any loss of characters would
silently break retrieval. This module splits text into deterministic segments
that collectively reproduce the source character-for-character.

Fenced code blocks are treated as atomic units: a fence is never split when it
fits within a segment by itself. Only a fence that alone exceeds the budget is
split (losslessly, by lines).
"""

from __future__ import annotations

import re
from typing import Protocol

import tiktoken

# Default embedding input budget. Conservative below the OpenAI-compatible
# provider limit and configurable per provider/model by callers.
DEFAULT_MAX_TOKENS = 3800
# Default per-segment character budget leaving room for context XML, URL,
# title, and prompt formatting inside the 8,000-character context budget.
DEFAULT_MAX_CHARS = 6000

# The tokenizer must match the provider's encoder. ``cl100k_base`` matches the
# common OpenAI-compatible tokenizer ratio and is the same encoder the embedder
# uses for its pre-flight counting.
_ENCODER = tiktoken.get_encoding("cl100k_base")


class TokenEncoder(Protocol):
    """Anything whose ``.encode(text)`` yields the token sequence for counting.

    Both ``tiktoken`` encodings and model-specific tokenizers (e.g. a
    ``transformers`` tokenizer adapter) satisfy this shape.
    """

    def encode(self, text: str) -> list[int]: ...


# Boundaries tried in order for non-fence text, each keeping its delimiter
# attached to the part it terminates so that ``"".join(segments)``
# reconstructs the source exactly.
_HEADING_BOUNDARY_RE = re.compile(r"^\s*#{1,6}\s+.*$|^\s*-{3,}\s*$|^\s*={3,}\s*$", re.MULTILINE)
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n{2,}")
_LIST_BOUNDARY_RE = re.compile(r"\n(?=[*-] |\d+\. )")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])")

# A fenced code block: opening line through closing line (same fence char).
_FENCE_RE = re.compile(r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*$")


def count_tokens(text: str, encoder: TokenEncoder = _ENCODER) -> int:
    """Return the number of tokens in *text* using the shared encoder."""
    if not text:
        return 0
    return len(encoder.encode(text))


def split_text_losslessly(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_chars: int = DEFAULT_MAX_CHARS,
    separators: tuple[str, ...] = (),
    encoder: TokenEncoder = _ENCODER,
) -> list[str]:
    """Split *text* into segments that fit within *max_tokens* and *max_chars*.

    Every non-whitespace character of the source is preserved in exactly one
    segment; ``"".join(segments)`` reconstructs the normalized source.
    Raises ``ValueError`` when a single token exceeds the budget.

    Parameters
    ----------
    text:
        Source text to split. Empty or whitespace-only text returns ``[]``.
    max_tokens:
        Hard token budget per segment.
    max_chars:
        Hard character budget per segment.
    separators:
        Optional explicit separator strings tried before the default
        boundaries (in order).
    encoder:
        Token encoder whose ``.encode(text)`` yields the token sequence used
        for budget counting. Defaults to the shared cl100k encoder; pass a
        model-specific encoder so the split budget matches the embedder's own
        pre-flight counting.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    normalized = text.strip()
    if not normalized:
        return []
    if _fits(normalized, max_tokens, max_chars, encoder):
        return [normalized]

    segments = _pack_atoms(normalized, max_tokens, max_chars, separators, encoder)
    validate_segments(normalized, segments)
    return segments


def validate_segments(original: str, segments: list[str]) -> None:
    """Raise ``ValueError`` when *segments* do not reconstruct *original*."""
    reconstructed = "".join(segments)
    if reconstructed.strip() != original.strip():
        raise ValueError("Segment reconstruction does not match the normalized source")


# ---------------------------------------------------------------------------
# Internal splitting
# ---------------------------------------------------------------------------


def _fits(text: str, max_tokens: int, max_chars: int, encoder: TokenEncoder = _ENCODER) -> bool:
    return len(text) <= max_chars and count_tokens(text, encoder) <= max_tokens


def _atoms(text: str) -> list[tuple[str, str]]:
    """Split *text* into (kind, content) atoms: ``fence`` or ``text``.

    A ``fence`` atom is a complete fenced code block. A ``text`` atom is the
    run between fences. Delimiters (the blank lines and surrounding prose)
    remain attached to the following text atom so reconstruction is exact.
    """
    atoms: list[tuple[str, str]] = []
    last = 0
    for match in _FENCE_RE.finditer(text):
        head = text[last : match.start()]
        if head:
            atoms.append(("text", head))
        atoms.append(("fence", match.group(0)))
        last = match.end()
    tail = text[last:]
    if tail:
        atoms.append(("text", tail))
    return atoms


def _pack_atoms(
    text: str,
    max_tokens: int,
    max_chars: int,
    separators: tuple[str, ...],
    encoder: TokenEncoder = _ENCODER,
) -> list[str]:
    atoms = _atoms(text)
    boundaries = _boundary_patterns(separators)
    segments: list[str] = []
    current = ""
    for kind, content in atoms:
        if _fits(content, max_tokens, max_chars, encoder):
            candidate = current + content
            if _fits(candidate, max_tokens, max_chars, encoder):
                current = candidate
            else:
                if current:
                    segments.append(current)
                current = content
            continue
        # A single atom exceeds the budget.
        if current:
            segments.append(current)
            current = ""
        if kind == "fence":
            # Split a too-large fence losslessly by lines, then re-pack so
            # continuation markers land in their own segments.
            for piece in _split_fence_losslessly(content, max_tokens, max_chars, encoder):
                if _fits(piece, max_tokens, max_chars, encoder):
                    segments.append(piece)
                else:
                    raise ValueError(f"Fence continuation exceeds budget: {len(piece)} chars")
        else:
            pieces = _split_recursive(content, max_tokens, max_chars, boundaries, depth=0, encoder=encoder)
            segments.extend(pieces)
    if current:
        segments.append(current)
    return segments


def _split_fence_losslessly(fence: str, max_tokens: int, max_chars: int, encoder: TokenEncoder = _ENCODER) -> list[str]:
    """Split an oversized fenced block by lines without dropping characters.

    The opener stays on the first piece and the closer on the last piece;
    ``"".join(pieces)`` reconstructs the original fence exactly. Continuation
    markers are the caller's per-segment ``segment_index``/``segment_total``
    metadata, not in-band fence text.

    Every emitted piece must individually satisfy the budget. Intermediate
    pieces carry a trailing ``"\\n"`` (so the line break between pieces
    survives reconstruction) and the last piece carries the closer, so the
    fit check reserves one char plus the closer length to avoid an
    off-by-one where a piece of exactly ``max_chars`` is rejected once the
    newline/closer is attached.
    """
    lines = fence.splitlines()
    if not lines:
        return [fence]
    opener = lines[0]
    closer = lines[-1] if len(lines) > 1 and lines[-1].strip().startswith(("```", "~~~")) else ""
    body = lines[1:-1] if closer else lines[1:]

    pieces: list[str] = []
    current: list[str] = [opener]
    for line in body:
        # Reserve the trailing newline plus the closer so every flushed piece
        # (with its "\n" or the final "\n{closer}") still fits the budget.
        candidate = "\n".join(current + [line]) + "\n" + closer
        if _fits(candidate, max_tokens, max_chars, encoder):
            current.append(line)
        else:
            # End this piece with a newline so the next piece reconstructs it.
            pieces.append("\n".join(current) + "\n")
            current = [line]
    if closer:
        pieces.append("\n".join(current) + f"\n{closer}")
    else:
        pieces.append("\n".join(current))
    return [p for p in pieces if p]


def _boundary_patterns(separators: tuple[str, ...]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for sep in separators:
        if sep:
            patterns.append(re.compile(re.escape(sep)))
    patterns.extend(
        [
            _HEADING_BOUNDARY_RE,
            _PARAGRAPH_BOUNDARY_RE,
            _LIST_BOUNDARY_RE,
            _SENTENCE_BOUNDARY_RE,
        ]
    )
    return patterns


def _split_recursive(
    text: str,
    max_tokens: int,
    max_chars: int,
    boundaries: list[re.Pattern[str]],
    depth: int,
    encoder: TokenEncoder = _ENCODER,
) -> list[str]:
    if _fits(text, max_tokens, max_chars, encoder):
        return [text]
    if depth >= len(boundaries):
        return _split_by_chars(text, max_chars, max_tokens, encoder)

    pattern = boundaries[depth]
    parts = _split_keep_delimiter(text, pattern)
    if len(parts) <= 1:
        return _split_recursive(text, max_tokens, max_chars, boundaries, depth + 1, encoder)

    segments: list[str] = []
    current = ""
    for part in parts:
        candidate = current + part
        if _fits(candidate, max_tokens, max_chars, encoder):
            current = candidate
            continue
        if current:
            segments.append(current)
            current = ""
        if _fits(part, max_tokens, max_chars, encoder):
            current = part
        else:
            sub = _split_recursive(part, max_tokens, max_chars, boundaries, depth + 1, encoder)
            segments.extend(sub[:-1])
            current = sub[-1]
    if current:
        segments.append(current)
    return segments


def _split_keep_delimiter(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split *text* on *pattern* keeping each delimiter on the part before it."""
    parts: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        parts.append(text[last : match.end()])
        last = match.end()
    parts.append(text[last:])
    return [p for p in parts if p]


def _split_by_chars(text: str, max_chars: int, max_tokens: int, encoder: TokenEncoder = _ENCODER) -> list[str]:
    """Last-resort split at whitespace boundaries preserving all characters."""
    parts = re.split(r"(\s+)", text)
    segments: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if len(part) > max_chars:
            raise ValueError(f"Single token exceeds budget ({len(part)} > {max_chars} chars)")
        candidate = current + part
        if _fits(candidate, max_tokens, max_chars, encoder):
            current = candidate
            continue
        if current:
            segments.append(current)
            current = part
        else:
            current = part
    if current:
        segments.append(current)
    return segments
