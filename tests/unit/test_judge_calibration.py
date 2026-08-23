import pytest

from data_engineering_copilot.evaluation.judge_calibration import agreement


def test_perfect_agreement():
    assert agreement([1, 0, 1, 1], [1, 0, 1, 1]) == (1.0, 1.0)


def test_known_kappa_value():
    y_true = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0, 0, 1]
    raw, kappa = agreement(y_true, y_pred)
    assert abs(raw - 0.7) < 1e-9
    n = len(y_true)
    p_yes_t = sum(y_true) / n
    p_yes_p = sum(y_pred) / n
    pe = p_yes_t * p_yes_p + (1 - p_yes_t) * (1 - p_yes_p)
    expected_kappa = (raw - pe) / (1 - pe)
    assert abs(kappa - expected_kappa) < 1e-9


def test_degenerate_single_class_returns_zero_kappa():
    raw, kappa = agreement([1, 1, 1], [1, 1, 1])
    assert raw == 1.0 and kappa == 0.0


def test_empty_inputs():
    with pytest.raises(ValueError):
        agreement([], [])
