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


def test_rerun_noise_within_tolerance_passes_despite_wide_ci():
    # Real-world case (2026-08-23): identical-harness rerun drifted -0.012
    # (0.247 vs baseline 0.259, n=219/220); bootstrap CI half-width ~±0.08 so a
    # ci_low-based rule breached -0.02 even though the retriever was unchanged.
    # Verdict must gate on the point delta, not the CI low.
    base = [1.0] * 57 + [0.0] * 163  # mean 0.259
    cur = [0.0] * 165 + [1.0] * 54  # mean 0.247, decorrelated from base
    ok, delta, (lo, _hi) = regression_verdict(cur, base, tolerance=0.02)
    assert -0.02 <= delta < 0
    assert ok is True
    assert lo < -0.02  # CI alone would have failed the gate


def test_empty_inputs_pass_trivially():
    ok, delta, (lo, hi) = regression_verdict([], [])
    assert ok is True and delta == 0.0 and (lo, hi) == (0.0, 0.0)
