"""TRACe-style context utilization/completeness diagnostics (offline, $0).

RAGBench/TRACe definitions operationalized over captured chunks:
utilization = used_chars / retrieved_chars; completeness = |relevant ∩ used| /
|relevant|. Informational only — never gates.
"""

from __future__ import annotations

_OVERLAP_THRESHOLD = 0.3


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


def _token_overlap(chunk_text: str, answer: str) -> float:
    ct, at = _tokens(chunk_text), _tokens(answer)
    if not ct:
        return 0.0
    return len(ct & at) / len(ct)


def chunk_is_used(chunk: dict, answer: str) -> bool:
    if chunk.get("cited"):
        return True
    return _token_overlap(chunk.get("text", ""), answer) >= _OVERLAP_THRESHOLD


def trace_utilization(chunks: list[dict], answer: str) -> float:
    """Fraction of retrieved characters that were actually used."""
    if not chunks or not answer.strip():
        return 0.0
    total = sum(len(c.get("text", "")) for c in chunks)
    if total == 0:
        return 0.0
    used = sum(len(c.get("text", "")) for c in chunks if chunk_is_used(c, answer))
    return used / total


def trace_completeness(chunks: list[dict]) -> float:
    """Fraction of relevant chunks that were used ('used' precomputed)."""
    relevant = [c for c in chunks if c.get("relevant")]
    if not relevant:
        return 0.0
    return sum(1 for c in relevant if c.get("used")) / len(relevant)


def percentile(values: list[float], q: float) -> float:
    """Deterministic percentile (linear interpolation)."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]
