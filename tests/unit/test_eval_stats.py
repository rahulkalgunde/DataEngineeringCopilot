from data_engineering_copilot.evaluation.stats import bootstrap_ci, regression_verdict


def test_ci_deterministic_under_seed():
    v = [0.5, 0.6, 0.55, 0.65, 0.5, 0.6]
    assert bootstrap_ci(v) == bootstrap_ci(v)


def test_identical_distributions_pass():
    a = [0.7, 0.72, 0.69, 0.71, 0.7, 0.73]
    ok, delta, (lo, hi) = regression_verdict(a, a)
    assert ok is True
    assert abs(delta) < 1e-9
    assert lo <= 0.0 <= hi


def test_clear_regression_fails():
    cur = [0.5, 0.52, 0.49, 0.51, 0.5, 0.52]
    base = [0.8, 0.82, 0.79, 0.81, 0.8, 0.83]
    ok, delta, (lo, hi) = regression_verdict(cur, base, tolerance=0.02)
    assert ok is False
    assert hi < -0.02


def test_small_improvement_within_tolerance_passes():
    cur = [0.70] * 10
    base = [0.715] * 10
    ok, _, _ = regression_verdict(cur, base, tolerance=0.02)
    assert ok is True


def test_empty_inputs_pass_trivially():
    ok, delta, (lo, hi) = regression_verdict([], [])
    assert ok is True and delta == 0.0 and (lo, hi) == (0.0, 0.0)
