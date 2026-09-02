"""Tests for logger utilities: safe_pct guard against % in stdlib logging."""

from __future__ import annotations

import logging

import pytest

from data_engineering_copilot.logger import safe_pct


class TestSafePct:
    """safe_pct escapes literal '%' to '%%' so stdlib logging does not raise
    TypeError when the logged string contains percent signs."""

    def test_escapes_single_percent(self) -> None:
        assert safe_pct("50%") == "50%%"

    def test_escapes_multiple_percents(self) -> None:
        assert safe_pct("100% complete 50% done") == "100%% complete 50%% done"

    def test_no_percent_unchanged(self) -> None:
        assert safe_pct("no percent here") == "no percent here"

    def test_empty_string(self) -> None:
        assert safe_pct("") == ""

    def test_only_percents(self) -> None:
        assert safe_pct("%%%") == "%%%%%%"

    def test_with_logger_format_string(self) -> None:
        """Simulate stdlib logging call with a string that originally had '%'.

        Before safe_pct: logging.warning("Progress: %s", "50%") raises TypeError.
        After safe_pct: logging.warning("Progress: %s", safe_pct("50%")) works.
        """
        logger = logging.getLogger("test_safe_pct")
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

        # This must NOT raise TypeError
        try:
            logger.warning("Progress: %s", safe_pct("50%"))
        except TypeError as exc:
            pytest.fail(f"safe_pct did not prevent TypeError: {exc}")

    def test_preserves_format_specifiers_in_format_string(self) -> None:
        """safe_pct only escapes '%' in the value, not the format string."""
        result = safe_pct("50%")
        assert result == "50%%"

        # The format string itself can still use % formatting
        formatted = f"Progress: {result}"
        assert formatted == "Progress: 50%%"
