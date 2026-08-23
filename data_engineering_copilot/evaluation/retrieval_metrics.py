"""Deterministic IR metrics for retrieval evals (binary relevance)."""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for e in expected if e in top) / len(expected)


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    exp_set = set(expected)
    dcg = sum(1.0 / math.log2(i + 1) for i, url in enumerate(retrieved[:k], start=1) if url in exp_set)
    ideal_hits = min(len(exp_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]. Empty input returns 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac
