from data_engineering_copilot.cli import _deterministic_sample_indices


def test_sampler_deterministic_and_bounded():
    a = _deterministic_sample_indices(n_total=100, n_sample=30, seed=7)
    b = _deterministic_sample_indices(n_total=100, n_sample=30, seed=7)
    assert a == b
    assert len(a) == 30
    assert len(set(a)) == 30
    assert all(0 <= i < 100 for i in a)


def test_sampler_caps_at_total():
    a = _deterministic_sample_indices(n_total=10, n_sample=30, seed=1)
    assert len(a) == 10
