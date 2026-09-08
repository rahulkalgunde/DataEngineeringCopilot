"""Tests for the pattern-level embedding input guard.

Covers the pure neutralizer (idempotent, trigger-splice semantics) and the
contract of the shared ``neutralize_embedding_input`` entry point.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.infrastructure.embedding_input_guard import (
    DEFAULT_EMBEDDING_TRIGGER_PATTERNS,
    neutralize_embedding_input,
)


@pytest.mark.parametrize(
    ("text", "expected_has_break"),
    [
        ("data:image/", True),
        ("padding data:image/ prefix", True),
        ("no `data:image/...;base64,` prefix", True),
        ("data:/", False),
        ("DATA:IMAGE/", False),
        ("data:image", False),
        ("plain text without triggers", False),
    ],
)
def test_neutralize_inserts_zero_width(text: str, expected_has_break: bool) -> None:
    sanitized, n_modified = neutralize_embedding_input([text])
    if expected_has_break:
        assert "\u200b" in sanitized[0]
        assert n_modified == 1
        assert "data:image/" not in sanitized[0]
    else:
        assert sanitized[0] == text
        assert n_modified == 0


def test_default_registry_is_pattern_level() -> None:
    """The registry must carry the trigger token (pattern), not per-doc fixes."""
    assert DEFAULT_EMBEDDING_TRIGGER_PATTERNS == ("data:image/",)


@pytest.mark.parametrize(
    "text",
    [
        "data:\u200bimage/",
        "already data:image\u200b/ sanitized",
        "data:image\u200c/",
    ],
)
def test_idempotent_on_already_sanitized(text: str) -> None:
    sanitized, n_modified = neutralize_embedding_input([text])
    assert sanitized[0] == text
    assert n_modified == 0


def test_multiple_occurrences_all_neutralized() -> None:
    sanitized, n_modified = neutralize_embedding_input(["data:image/ and data:image/"])
    assert n_modified == 1
    assert sanitized[0].count("data:image/") == 0
    assert sanitized[0].count("\u200b") == 2


def test_empty_triggers_returns_input_untouched() -> None:
    texts = ["data:image/"]
    sanitized, n_modified = neutralize_embedding_input(texts, triggers=())
    assert sanitized is texts
    assert n_modified == 0


def test_length_preserved() -> None:
    texts = ["a", "data:image/", "zzzz"]
    sanitized, n_modified = neutralize_embedding_input(texts)
    assert len(sanitized) == len(texts)
    assert n_modified == 1


def test_custom_trigger_registry() -> None:
    sanitized, n_modified = neutralize_embedding_input(
        ["foo:bar/ zzz"],
        triggers=("foo:bar/",),
    )
    assert "\u200b" in sanitized[0]
    assert n_modified == 1
    assert "foo:bar/" not in sanitized[0]
