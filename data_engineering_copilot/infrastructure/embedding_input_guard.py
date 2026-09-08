"""Pattern-level guard against content-routing embedding provider rejections.

Some OpenAI-compatible gateways (NVIDIA's) route the *content* of an embedding
request, not just its shape. Any input containing certain tokens is rejected
deterministically with a 5xx that looks like an outage but is actually a
content incompatibility (e.g. ``data:image/`` → 503 "image inputs require VLM
serving to be enabled on this server"). Because the rejection is content
anchored, every batch containing the token fails forever, no matter how long
the backoff — the offline wait controller treats it as a transient outage and
spins.

The guard neutralizes the trigger *in embedding input only* by splicing a
zero-width interspace into the token, breaking the content-route match while
leaving the text semantically identical. Stored chunk text / BM25 index are
never touched.

Verified against the live NVIDIA endpoint (2026-09-08): only the exact
lowercase token ``data:image/`` trips the router; ``data:image``, ``data:/``,
``DATA:IMAGE/``, and any interspaced variant all pass (200). Replacement is
idempotent: a neutralized token no longer contains the raw trigger, so
re-running the guard is a no-op.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Trigger tokens rejected by a content-routing embedding gateway. Neutralized
# (zero-width interspace spliced in) in embedding input by the guard.
DEFAULT_EMBEDDING_TRIGGER_PATTERNS: tuple[str, ...] = ("data:image/",)


def _neutralize(text: str, pattern: str) -> str:
    """Splice a zero-width interspace into every occurrence of *pattern*.

    The interspace goes at the pattern midpoint so the token no longer exists
    contiguously for the content router, while remaining one typographic
    character short of any other reader. Idempotent (a spliced token cannot
    re-match the raw trigger).
    """
    if not pattern:
        return text
    mid = len(pattern) // 2
    neutralized = pattern[:mid] + "\u200b" + pattern[mid:]
    if neutralized == pattern:
        return text
    return text.replace(pattern, neutralized)


def neutralize_embedding_input(
    texts: list[str],
    triggers: tuple[str, ...] = DEFAULT_EMBEDDING_TRIGGER_PATTERNS,
) -> tuple[list[str], int]:
    """Neutralize trigger tokens in embedding *input* only.

    Returns ``(sanitized_texts, n_modified)``. When ``triggers`` is empty the
    input is returned untouched. Never modifies ids/titles/urls — pure text.
    """
    if not triggers:
        return texts, 0
    modified = 0
    out: list[str] = []
    for text in texts:
        orig = text
        for pattern in triggers:
            text = _neutralize(text, pattern)
        if text != orig:
            modified += 1
        out.append(text)
    return out, modified
