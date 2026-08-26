"""Tests for evaluation/cost_estimate.py."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from data_engineering_copilot.evaluation.cost_estimate import (
    enforce_cost_gate,
    estimate_calls,
)


class TestEstimateCalls:
    def test_eval_generation(self) -> None:
        assert estimate_calls("eval-generation", 10) == 10 * (1 + 3 * 2)

    def test_evaluate_no_ragas(self) -> None:
        assert estimate_calls("evaluate", 10) == 10 * 2

    def test_evaluate_with_ragas(self) -> None:
        assert estimate_calls("evaluate", 10, ragas=True) == 10 * (2 + 19)

    def test_evaluate_spark_short_circuits(self) -> None:
        assert estimate_calls("evaluate", 10, spark=True) == 0

    def test_unknown_command(self) -> None:
        assert estimate_calls("unknown", 10) == 0


class TestEnforceCostGate:
    def test_zero_estimate_passes(self) -> None:
        enforce_cost_gate("test", 0)  # should not raise

    def test_tty_passes(self) -> None:
        with patch("sys.stdin.isatty", return_value=True):
            enforce_cost_gate("test", 100)

    def test_force_set_passes(self) -> None:
        with patch("sys.stdin.isatty", return_value=False), patch.dict(os.environ, {"FORCE": "1"}):
            enforce_cost_gate("test", 100)

    def test_non_tty_no_force_raises(self) -> None:
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(SystemExit),
        ):
            enforce_cost_gate("test", 100)
