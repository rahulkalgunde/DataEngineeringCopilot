"""Bootstrap confidence intervals for eval metric comparisons.

Paired resampling: identical index sequences are applied to both series so the
delta distribution reflects per-query pairing. Deterministic under seed.
"""

from __future__ import annotations

import math
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
    """Regression verdict with bootstrap CI reported as context.

    Returns (pass, mean_delta, (ci_low, ci_high)). Pass iff the point delta is
    within tolerance: ``mean_delta >= -tolerance``.

    Why not gate on ``ci_low``: at n≈220 the percentile bootstrap CI half-width
    is ~±0.08, so a CI-based rule demands mean_delta ≳ +0.06 to pass — even an
    unchanged retriever fails on rerun noise alone (measured 0.012 drift between
    identical-harness runs, 2026-08-23). CI is still computed and returned so
    callers can print uncertainty alongside the verdict.
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
    return (mean_delta >= -tolerance, mean_delta, (lo, hi))


def per_intent_tolerance(
    baseline_recall: float,
    n: int,
    *,
    floor: float = 0.05,
) -> float:
    """Noise-aware per-intent gate tolerance: max(floor, 2σ of the baseline mean).

    A fixed tolerance below the binomial noise floor gates rerun variance, not
    regressions: at n=23, p=0.30 the 1σ sampling error is ~±0.10, so a −0.05
    point rule flips on identical code+index (measured 2026-08-23: debugging
    intent swung −0.087 across two runs of an unchanged harness). 2σ keeps the
    gate above noise for small n while collapsing to ``floor`` for large n.
    """
    if n <= 0:
        return floor
    sigma = math.sqrt(max(baseline_recall, 0.0) * (1 - min(max(baseline_recall, 0.0), 1.0)) / n)
    return max(floor, 2 * sigma)
