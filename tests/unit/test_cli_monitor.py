"""Tests for cli_monitor.py pure functions."""

from __future__ import annotations

from data_engineering_copilot.cli_monitor import (
    STATUS_ICONS,
    TERMINAL_STATES,
    _fmt_delta,
    _fmt_elapsed,
)


class TestFmtDelta:
    def test_zero(self) -> None:
        assert _fmt_delta(0) == "0"

    def test_positive(self) -> None:
        assert _fmt_delta(5) == "+5"

    def test_negative(self) -> None:
        assert _fmt_delta(-3) == "-3"


class TestFmtElapsed:
    def test_seconds(self) -> None:
        assert _fmt_elapsed(30) == "00:00:30"

    def test_minutes(self) -> None:
        assert _fmt_elapsed(90) == "00:01:30"

    def test_hours(self) -> None:
        assert _fmt_elapsed(3661) == "01:01:01"


class TestConstants:
    def test_terminal_states(self) -> None:
        assert "COMPLETED" in TERMINAL_STATES
        assert "FAILED" in TERMINAL_STATES

    def test_status_icons(self) -> None:
        assert STATUS_ICONS["COMPLETED"] == "[OK]"
        assert STATUS_ICONS["FAILED"] == "[!!]"
