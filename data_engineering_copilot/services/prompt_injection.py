"""Shared prompt-injection detection used by API middleware and input guardrails.

Centralizes the injection pattern list so defense-in-depth layers (request
middleware, retrieved-document scanning) agree on what counts as an injection
attempt.

Patterns are split into two classes:

- **Directive patterns** (``INJECTION_PATTERNS``): imperative overrides —
  "ignore previous instructions", "you are now", "reveal the system prompt",
  chat-template markers. A single directive hit is itself suspicious.
- **Descriptive patterns** (``DESCRIPTIVE_INJECTION_PATTERNS``) and the
  structural header rule: legitimate documentation vocabulary about these
  features ("system prompt", "developer mode", "## System", "## Output format").
  These only *corroborate* a directive; alone they never trigger rejection,
  otherwise real documentation about prompt engineering would be dropped.
"""

from __future__ import annotations

import re

# Directive patterns: a single hit is enough to be suspicious. These are
# imperative instructions that try to override or extract the model.
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|above|prior|the\s+above)\s+(instructions|prompts|directions)", re.IGNORECASE
    ),
    re.compile(r"you\s+are\s+(now|free|an?\s+AI\s+named|DAN)", re.IGNORECASE),
    re.compile(r"(REVEAL|LEAK|DUMP|DISPLAY|OUTPUT)\s+(ALL\s+)?(INSTRUCTIONS|PROMPT|SYSTEM|CONSTRAINTS)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(DAN|an?\s+AI\s+without|if\s+you\s+are)", re.IGNORECASE),
    re.compile(r"bypass|jailbreak|breach|compromise", re.IGNORECASE),
    re.compile(r"new\s+(system|developer)\s*(role|prompt|instructions)\s*:", re.IGNORECASE),
    re.compile(
        r"override\s+((all\s+)?(previous|prior|above|system))\s+(instructions|prompts|rules|constraints)", re.IGNORECASE
    ),
    re.compile(
        r"(forget|disregard|ignore)\s+(everything|all|your)\s+(previous|prior|instructions|rules)", re.IGNORECASE
    ),
    re.compile(r"you\s+(are|must|should)\s+now\s+(act|behave|respond)\s+as", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE),
    re.compile(r"###\s*(system|human|assistant)\s*:", re.IGNORECASE),
]

# Descriptive vocabulary that legitimately appears in documentation *about*
# these features. Never sufficient on its own — only adds weight when a
# directive pattern already fired, so a doc explaining "system prompts" is not
# mistaken for an injection attempt.
DESCRIPTIVE_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"system\s+prompt|developer\s+mode|prompt\s+injection", re.IGNORECASE),
]

# Structural markers: markdown headers that could mimic prompt sections. Like
# the descriptive patterns, these only corroborate an actual directive; a
# legitimate "## Output format" or "## System" documentation heading is not an
# injection on its own.
STRUCTURAL_INJECTION_RE = re.compile(r"^#{1,6}\s*(system|constraints|instructions|output format)", re.MULTILINE)

INJECTION_THRESHOLD = 0.3

# Each directive hit contributes this weight.
_DIRECTIVE_WEIGHT = 0.3
# Structural header corroboration weight.
_STRUCTURAL_WEIGHT = 0.4


def detect_prompt_injection(text: str) -> float:
    """Return 0.0–1.0 injection likelihood. ``>= INJECTION_THRESHOLD`` suggests rejection.

    Descriptive feature vocabulary ("system prompt", "developer mode", section
    headings) is only counted when a directive pattern has already fired, so
    legitimate documentation is not flagged.
    """
    text_lower = text.lower()
    score = 0.0
    directive_fired = False
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text_lower):
            score += _DIRECTIVE_WEIGHT
            directive_fired = True

    if not directive_fired:
        # No imperative override present — descriptive mentions are not evidence.
        return 0.0

    for pattern in DESCRIPTIVE_INJECTION_PATTERNS:
        if pattern.search(text_lower):
            score += _DIRECTIVE_WEIGHT
    if STRUCTURAL_INJECTION_RE.search(text_lower):
        score += _STRUCTURAL_WEIGHT
    return min(1.0, score)
