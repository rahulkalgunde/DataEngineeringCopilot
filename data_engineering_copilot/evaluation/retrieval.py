"""Ablation harness helpers for dense / sparse / hybrid evaluation.

Provides deterministic holdout split and bootstrap CI for hybrid vs best-single.
Used by ``dec eval-retrieval --ablation`` (cli.py) and the scripts gate.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence

from data_engineering_copilot.evaluation.stats import _percentile_rank


def split_queries(
    queries: Sequence[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Deterministic 50/50 shuffle split (seed=42).

    Mirrors plan Task 3: 220q baseline_inscope → 110 train / 110 held-out.
    When n is odd, train gets the larger half.
    """
    qs = list(queries)
    rnd = random.Random(seed)
    rnd.shuffle(qs)
    mid = (len(qs) + 1) // 2
    # Plan expects exactly 110/110 for 220; keep generic for other sizes.
    if len(qs) == 220:
        return qs[:110], qs[110:]
    return qs[:mid], qs[mid:]


def bootstrap_delta_ci(
    hybrid_scores: Sequence[float],
    best_single_scores: Sequence[float],
    n_boot: int = 1000,
    seed: int = 13,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap 95% CI for mean(hybrid - best_single).

    Paired bootstrap: same resampled indices applied to both lists.
    Returns (mean_delta, (ci_low, ci_high)).
    """
    if not hybrid_scores or not best_single_scores:
        return 0.0, (0.0, 0.0)
    n = min(len(hybrid_scores), len(best_single_scores))
    h = list(hybrid_scores)[:n]
    b = list(best_single_scores)[:n]
    deltas = [hv - bv for hv, bv in zip(h, b, strict=False)]
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    # bootstrap mean deltas
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.fmean(sample))
    lo = _percentile_rank(boot_means, 0.025)
    hi = _percentile_rank(boot_means, 0.975)
    return mean_delta, (lo, hi)


def per_query_best(recalls_dense: Sequence[float], recalls_sparse: Sequence[float]) -> list[float]:
    """Per-query best-single recall (max of dense / sparse)."""
    return [max(d, s) for d, s in zip(recalls_dense, recalls_sparse, strict=False)]


# Search mode loop descriptor for cli ablation (keeps cli import-light).
ABLATION_MODES = ("dense", "sparse", "hybrid_rrf_k60", "hybrid_rrf_k5", "dbsf")
