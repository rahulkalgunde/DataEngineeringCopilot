"""Context assembly evaluation metrics.

Measures duplicate rate, source coverage, compression ratio,
and needle-loss for assembled context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def duplicate_candidate_rate(assembled_text: str) -> float:
    """Fraction of tokens that are duplicates: 1 - (unique_tokens / total_tokens)."""
    tokens = re.findall(r"[a-z0-9_]+", assembled_text.lower())
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def source_coverage_rate(source_names: list[str], total_sources_in_pool: int) -> float:
    """Fraction of unique source documents retained post-deduplication."""
    if total_sources_in_pool == 0:
        return 0.0
    return len(set(source_names)) / total_sources_in_pool


def context_compression_ratio(assembled_chars: int, initial_candidate_chars: int) -> float:
    """Ratio of final assembled chars to initial candidate chars."""
    if initial_candidate_chars == 0:
        return 0.0
    return assembled_chars / initial_candidate_chars


def needle_loss_rate(
    assembled_text: str,
    gold_facts: list[str],
) -> float:
    """Fraction of gold-standard facts missing from the assembled context.

    A "needle" is present if any 4-token window of the fact appears in the text.
    """
    if not gold_facts:
        return 0.0
    assembled_lower = assembled_text.lower()
    missing = 0
    for fact in gold_facts:
        tokens = re.findall(r"[a-z0-9_]+", fact.lower())
        found = False
        for i in range(len(tokens) - 3):
            window = " ".join(tokens[i : i + 4])
            if window in assembled_lower:
                found = True
                break
        if not found:
            missing += 1
    return missing / len(gold_facts)


@dataclass
class AssemblyEvalReport:
    duplicate_rate: float = 0.0
    source_coverage: float = 0.0
    compression_ratio: float = 0.0
    needle_loss: float = 0.0

    def summary(self) -> str:
        return (
            f"Duplicate rate: {self.duplicate_rate:.4f}\n"
            f"Source coverage: {self.source_coverage:.4f}\n"
            f"Compression ratio: {self.compression_ratio:.4f}\n"
            f"Needle loss: {self.needle_loss:.4f}"
        )
