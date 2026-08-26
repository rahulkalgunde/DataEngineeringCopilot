"""Tests for utils/text.py."""

from __future__ import annotations

from data_engineering_copilot.utils.text import normalize_whitespace, slugify


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self) -> None:
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_tabs(self) -> None:
        assert normalize_whitespace("hello\tworld") == "hello world"

    def test_collapses_newlines(self) -> None:
        assert normalize_whitespace("hello\nworld") == "hello world"

    def test_strips_leading_trailing(self) -> None:
        assert normalize_whitespace("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert normalize_whitespace("") == ""


class TestSlugify:
    def test_lowercases(self) -> None:
        assert slugify("Hello World") == "hello-world"

    def test_replaces_special_chars(self) -> None:
        assert slugify("hello@world!") == "hello-world"

    def test_strips_trailing_dashes(self) -> None:
        assert slugify("hello-") == "hello"

    def test_empty_returns_document(self) -> None:
        assert slugify("") == "document"

    def test_only_special_returns_document(self) -> None:
        assert slugify("!!!") == "document"
