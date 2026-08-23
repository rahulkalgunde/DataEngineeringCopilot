"""Bootstrap confidence intervals for eval metric comparisons.

Paired resampling: identical index sequences are applied to both series so the
delta distribution reflects per-query pairing. Deterministic under seed.
"""

from __future__ import annotations

import random
import statistics


def _percentile_rank(xs: list[float], q: float) -> float:
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def bootstrap_ci(
    values: list[float], *, n_boot: int = 2000, confidence: float = 0.95, seed: int = 13
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic under ``seed``."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)]
    alpha = (1 - confidence) / 2
    return (_percentile_rank(means, alpha), _percentile_rank(means, 1 - alpha))


def regression_verdict(
    current: list[float],
    baseline: list[float],
    *,
    tolerance: float = 0.02,
    seed: int = 13,
) -> tuple[bool, float, tuple[float, float]]:
    """Paired-bootstrap regression verdict.

    Returns (pass, mean_delta, (ci_low, ci_high)). Pass iff ci_low > -tolerance:
    we only declare a regression when even the optimistic end of the delta
    distribution is below the tolerance.
    """
    if not current or not baseline:
        return (True, 0.0, (0.0, 0.0))
    n = min(len(current), len(baseline))
    cur, base = current[:n], baseline[:n]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(statistics.fmean(cur[i] for i in idx) - statistics.fmean(base[i] for i in idx))
    lo = _percentile_rank(deltas, 0.025)
    hi = _percentile_rank(deltas, 0.975)
    mean_delta = statistics.fmean(cur) - statistics.fmean(base)
    return (lo > -tolerance, mean_delta, (lo, hi))
