"""Deterministic IR metrics for retrieval evals (binary relevance)."""

from __future__ import annotations

import math

from data_engineering_copilot.evaluation.url_normalization import url_content_key


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    exp_norm = [url_content_key(e) for e in expected]
    top = {url_content_key(u) for u in retrieved[:k]}
    return sum(1 for e in exp_norm if e in top) / len(exp_norm)


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """nDCG@K with binary relevance, deduped — duplicate URLs from multi-chunk
    pages must not earn DCG credit twice (they pushed nDCG past 1.0)."""
    if not expected:
        return 0.0
    exp_set = {url_content_key(e) for e in expected}
    seen: set[str] = set()
    dcg = 0.0
    for i, url in enumerate(retrieved[:k], start=1):
        n = url_content_key(url)
        if n in exp_set and n not in seen:
            dcg += 1.0 / math.log2(i + 1)
            seen.add(n)
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
