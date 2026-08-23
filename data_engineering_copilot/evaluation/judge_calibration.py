"""Judge-vs-human calibration: raw agreement + Cohen's kappa.

Gate (industry floor): raw >= 0.80 AND kappa >= 0.60. Exit codes:
0 pass · 1 below gate · 2 unlabeled/incomplete rows present.
"""

from __future__ import annotations

from dataclasses import dataclass

KAPPA_GATE = 0.60
RAW_GATE = 0.80


def agreement(y_true: list[int], y_pred: list[int]) -> tuple[float, float]:
    """(raw_agreement, cohen_kappa). kappa=0.0 when degenerate (single class)."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        raise ValueError("agreement requires equal-length non-empty label lists")
    n = len(y_true)
    raw = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / n
    p_yes_t = sum(y_true) / n
    p_yes_p = sum(y_pred) / n
    pe = p_yes_t * p_yes_p + (1 - p_yes_t) * (1 - p_yes_p)
    if pe == 1.0:
        return raw, 0.0
    return raw, (raw - pe) / (1 - pe)


@dataclass
class CalibrationReport:
    metric: str
    raw: float
    kappa: float
    passed: bool


def verdict_for(raw: float, kappa: float) -> bool:
    return raw >= RAW_GATE and kappa >= KAPPA_GATE
