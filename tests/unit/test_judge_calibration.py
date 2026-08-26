"""Tests for evaluation/judge_calibration.py."""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.judge_calibration import (
    KAPPA_GATE,
    RAW_GATE,
    agreement,
    verdict_for,
)


class TestAgreement:
    def test_perfect_agreement(self) -> None:
        raw, kappa = agreement([1, 0, 1], [1, 0, 1])
        assert raw == 1.0
        assert kappa == 1.0

    def test_no_agreement(self) -> None:
        raw, kappa = agreement([1, 1, 0], [0, 0, 1])
        assert raw == 0.0

    def test_partial(self) -> None:
        raw, kappa = agreement([1, 0, 1, 0], [1, 1, 1, 0])
        assert 0 < raw < 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            agreement([], [])

    def test_mismatched_length_raises(self) -> None:
        with pytest.raises(ValueError):
            agreement([1, 0], [1])


class TestVerdictFor:
    def test_passes_both_gates(self) -> None:
        assert verdict_for(0.9, 0.7) is True

    def test_fails_raw(self) -> None:
        assert verdict_for(0.7, 0.7) is False

    def test_fails_kappa(self) -> None:
        assert verdict_for(0.9, 0.5) is False

    def test_exact_gate_passes(self) -> None:
        assert verdict_for(RAW_GATE, KAPPA_GATE) is True
