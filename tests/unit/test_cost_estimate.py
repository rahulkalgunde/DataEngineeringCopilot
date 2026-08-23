import pytest

from data_engineering_copilot.evaluation.cost_estimate import enforce_cost_gate, estimate_calls


def test_generation_formula():
    assert estimate_calls("eval-generation", 12, n_trials=3) == 12 * 7


def test_evaluate_formula_with_and_without_ragas():
    assert estimate_calls("evaluate", 12, ragas=False) == 12 * 2
    assert estimate_calls("evaluate", 12, ragas=True) == 12 * 21


def test_spark_recall_rows_are_free():
    assert estimate_calls("evaluate", 51, spark=True) == 0


def test_force_gate_exits_non_interactive(monkeypatch):
    monkeypatch.setenv("FORCE", "")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit) as ei:
        enforce_cost_gate("evaluate", 240)
    assert ei.value.code == 2


def test_force_gate_passes_with_env(monkeypatch):
    monkeypatch.setenv("FORCE", "1")
    enforce_cost_gate("evaluate", 240)  # no raise


def test_zero_estimate_never_gates(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    enforce_cost_gate("evaluate", 0)  # no raise, no FORCE needed
