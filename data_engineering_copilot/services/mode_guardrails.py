"""Verified deployment-mode fact guardrails for high-risk queries.

Some questions (resource management, cluster mode comparisons) tempt LLMs to
smooth over differences between execution modes and hallucinate cross-mode
behaviour.  This module injects a small, corpus-verified fact block into the
system prompt for such queries.  Every fact string is a byte-exact substring
of the pinned Spark documentation corpus (``docs/running-on-yarn.md`` /
``docs/running-on-kubernetes.md``); the unit tests assert that invariant so the
facts can never silently drift from the documentation.
"""

from __future__ import annotations

import re

# Mode keywords: a question mentioning one of these triggers the guardrail.
_MODE_PATTERNS: dict[str, re.Pattern[str]] = {
    "yarn": re.compile(r"\byarn\b", re.IGNORECASE),
    "kubernetes": re.compile(r"\bkubernetes\b|\bk8s\b", re.IGNORECASE),
}

# Corpus-verified facts.  Each value must be a byte-exact substring of the
# pinned doc file listed next to it (enforced by tests/unit/test_mode_guardrails.py).
# Only facts verified against the corpus are shipped — never model memory.
_MODE_FACTS: dict[str, tuple[str, str]] = {
    # (fact, corpus file it must appear in)
    "yarn": (
        "To start the Spark Shuffle Service on each `NodeManager` in your YARN cluster",
        "docs/running-on-yarn.md",
    ),
    "kubernetes": (
        "Kubernetes doesn't support an external shuffle service at this time",
        "docs/running-on-kubernetes.md",
    ),
}

_GUARDRAIL_HEADER = "## VERIFIED DOCUMENTATION FACTS\n"
_GUARDRAIL_INTRO = (
    "The following facts are verified against the Apache Spark documentation and "
    "describe the specific execution mode(s) named in the question. Do not transfer "
    "these facts to other modes, and do not contradict them using general knowledge.\n"
)


def detect_modes(question: str) -> list[str]:
    """Return the execution modes mentioned in ``question``, in registry order."""
    return [mode for mode, pattern in _MODE_PATTERNS.items() if pattern.search(question)]


def build_mode_guardrail_block(question: str) -> str | None:
    """Build a ``VERIFIED DOCUMENTATION FACTS`` prompt block, or ``None``.

    Returns ``None`` when the question names no known execution mode, so the
    system prompt stays byte-identical to the baseline for ordinary queries.
    """
    modes = detect_modes(question)
    if not modes:
        return None
    facts = _MODE_FACTS.get(modes[0])
    if facts is None:
        return None
    lines = [_GUARDRAIL_HEADER, _GUARDRAIL_INTRO]
    for mode in modes:
        fact = _MODE_FACTS.get(mode)
        if fact is not None:
            lines.append(f"- [{mode}] {fact[0]}")
    return "\n".join(lines) + "\n"
