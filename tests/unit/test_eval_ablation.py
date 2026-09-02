"""Ablation harness: dense / sparse / hybrid with holdout split."""

from __future__ import annotations

import subprocess


def test_eval_retrieval_ablation_flag_exists() -> None:
    r = subprocess.run(
        ["dec_venv/bin/dec", "eval-retrieval", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--ablation" in r.stdout, r.stdout


def test_eval_retrieval_split_flag_exists() -> None:
    r = subprocess.run(
        ["dec_venv/bin/dec", "eval-retrieval", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--split" in r.stdout, r.stdout


def test_ablation_split_deterministic() -> None:
    from data_engineering_copilot.evaluation.retrieval import split_queries

    qs = [{"id": f"q{i}"} for i in range(220)]
    train_a, held_a = split_queries(qs, seed=42)
    train_b, held_b = split_queries(qs, seed=42)
    assert len(train_a) == 110
    assert len(held_a) == 110
    assert train_a == train_b
    assert held_a == held_b
    # ensure partition cover all
    ids = {q["id"] for q in train_a + held_a}
    assert len(ids) == 220


def test_bootstrap_delta_ci_includes_zero_when_no_difference() -> None:
    from data_engineering_copilot.evaluation.retrieval import bootstrap_delta_ci

    hybrid = [1.0, 0.0, 1.0, 0.0] * 25
    best = [1.0, 0.0, 1.0, 0.0] * 25
    mean_delta, (lo, hi) = bootstrap_delta_ci(hybrid, best, n_boot=200, seed=42)
    assert abs(mean_delta) < 1e-9
    assert lo <= 0 <= hi


def test_bootstrap_delta_ci_positive_when_hybrid_wins() -> None:
    from data_engineering_copilot.evaluation.retrieval import bootstrap_delta_ci

    hybrid = [1.0] * 100
    best = [0.0] * 100
    mean_delta, (lo, hi) = bootstrap_delta_ci(hybrid, best, n_boot=200, seed=42)
    assert mean_delta == 1.0
    assert lo > 0
