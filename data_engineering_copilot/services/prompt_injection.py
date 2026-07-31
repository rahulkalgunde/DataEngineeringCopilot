"""Shared prompt-injection detection used by API middleware and input guardrails.

Centralizes the injection pattern list so defense-in-depth layers (request
middleware, retrieved-document scanning) agree on what counts as an injection
attempt.
"""

from __future__ import annotations

import re

# Prompt injection detection patterns
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|above|prior|the\s+above)\s+(instructions|prompts|directions)", re.IGNORECASE
    ),
    re.compile(r"you\s+are\s+(now|free|an?\s+AI\s+named|DAN)", re.IGNORECASE),
    re.compile(r"system\s+prompt|developer\s+mode|prompt\s+injection", re.IGNORECASE),
    re.compile(r"(REVEAL|LEAK|DUMP|DISPLAY|OUTPUT)\s+(ALL\s+)?(INSTRUCTIONS|PROMPT|SYSTEM|CONSTRAINTS)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(DAN|an?\s+AI\s+without|if\s+you\s+are)", re.IGNORECASE),
    re.compile(r"bypass|jailbreak|breach|compromise", re.IGNORECASE),
    re.compile(r"new\s+(system|developer)\s*(role|prompt|instructions)\s*:", re.IGNORECASE),
    re.compile(r"override\s+(previous|all|system)\s+(instructions|prompts|rules|constraints)", re.IGNORECASE),
    re.compile(
        r"(forget|disregard|ignore)\s+(everything|all|your)\s+(previous|prior|instructions|rules)", re.IGNORECASE
    ),
    re.compile(r"you\s+(are|must|should)\s+now\s+(act|behave|respond)\s+as", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE),
    re.compile(r"###\s*(system|human|assistant)\s*:", re.IGNORECASE),
]

# Structural injection markers: markdown headers that could mimic prompt sections
STRUCTURAL_INJECTION_RE = re.compile(r"^#{1,6}\s*(system|constraints|instructions|output format)", re.MULTILINE)

INJECTION_THRESHOLD = 0.3


def detect_prompt_injection(text: str) -> float:
    """Return 0.0–1.0 injection likelihood. ``>= INJECTION_THRESHOLD`` suggests rejection."""
    text_lower = text.lower()
    score = 0.0
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text_lower):
            score += 0.3
    if STRUCTURAL_INJECTION_RE.search(text_lower):
        score += 0.4
    return min(1.0, score)
