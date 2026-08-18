"""Code-span masking for sentence-preserving tokenization.

Replaces fenced and inline code spans with document-scoped, collision-safe
sentinels so NLTK sentence tokenization never splits inside code. The sentinel
replacement is reversible: ``unmask_code_spans(mask_code_spans(text).text, ...)``
returns the original text unchanged. A malformed (unclosed) fence is left
untouched rather than partially masked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A complete fenced code block: opening line through closing line (same fence).
_FENCE_RE = re.compile(r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*$")
# Any fence opener line, for detecting malformed (unclosed) fences.
_FENCE_OPENER_RE = re.compile(r"(?m)^(`{3,}|~{3,})")
# Inline code: a run of backticks, non-greedy content without that same run, close.
_INLINE_CODE_RE = re.compile(r"(`+)(?:(?!\1).)+?\1")

_SENTINEL_PREFIX = "\u0000CODE"


@dataclass(frozen=True)
class MaskedText:
    """Masked *text* plus the sentinel-to-original span mapping.

    ``spans`` is a tuple of ``(sentinel, original)`` pairs in masking order.
    Sentinels contain NUL bytes and are verified absent from the input, so they
    cannot collide with placeholder-like user text.
    """

    text: str
    spans: tuple[tuple[str, str], ...]


def mask_code_spans(text: str) -> MaskedText:
    """Replace code spans in *text* with collision-safe sentinels.

    Returns ``MaskedText(text=text, spans=())`` unchanged when *text* is empty,
    contains no code spans, or contains a malformed (unclosed) fence.
    """
    if not text:
        return MaskedText(text=text, spans=())

    fences = list(_FENCE_RE.finditer(text))
    remainder = _FENCE_RE.sub("", text)
    if _FENCE_OPENER_RE.search(remainder):
        # A fence opener without a matching closer: masking the rest would
        # silently reorder code into the surrounding prose. Leave unchanged.
        return MaskedText(text=text, spans=())

    fence_spans = [(m.start(), m.end()) for m in fences]
    spans: list[tuple[int, int]] = list(fence_spans)

    for match in _INLINE_CODE_RE.finditer(text):
        start, end = match.start(), match.end()
        if any(start < fence_end and end > fence_start for fence_start, fence_end in fence_spans):
            continue
        spans.append((start, end))

    if not spans:
        return MaskedText(text=text, spans=())

    # Replace spans with sentinels in reverse order so earlier offsets stay valid.
    sentinel_pairs: list[tuple[str, str]] = []
    masked = text
    for index, (start, end) in reversed(list(enumerate(spans))):
        original = text[start:end]
        sentinel = _collision_safe_sentinel(text, index)
        masked = masked[:start] + sentinel + masked[end:]
        sentinel_pairs.append((sentinel, original))
    sentinel_pairs.reverse()
    return MaskedText(text=masked, spans=tuple(sentinel_pairs))


def unmask_code_spans(text: str, masked: MaskedText) -> str:
    """Restore every sentinel in *text* to its original code span."""
    result = text
    for sentinel, original in masked.spans:
        result = result.replace(sentinel, original)
    return result


def _collision_safe_sentinel(text: str, index: int) -> str:
    """A NUL-delimited sentinel guaranteed not to appear in *text*."""
    candidate = f"{_SENTINEL_PREFIX}{index}\u0000"
    while candidate in text:
        candidate = "\u0000" + candidate + "\u0000"
    return candidate
