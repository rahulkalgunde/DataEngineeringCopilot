"""Tests for the tokenizer registry: known input limits and encoder resolution.

Hermetic: never triggers a HuggingFace download. The real-model tokenizer path
is exercised by monkeypatching the loader; production resolution is covered by
the live-probe verification (see conversation notes), not unit tests.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.infrastructure.tokenizer_registry import (
    KNOWN_INPUT_LIMITS,
    declared_input_limit,
    reset_tokenizer_cache,
    token_counter_for,
)


def test_known_input_limits_declared() -> None:
    assert declared_input_limit("nvidia/nemotron-3-embed-1b") == ("chars", 65536)
    assert declared_input_limit("nvidia/nemotron-3-embed-1b:free") == ("tokens", 4096)


def test_unknown_model_has_no_declared_limit() -> None:
    assert declared_input_limit("some/other-model") is None


def test_known_limits_table_entries() -> None:
    # Every known model must declare an explicit unit+limit.
    assert KNOWN_INPUT_LIMITS
    for _model, (unit, limit) in KNOWN_INPUT_LIMITS.items():
        assert unit in ("chars", "tokens")
        assert limit > 0


def test_unknown_model_counter_falls_back_to_cl100k() -> None:
    counter = token_counter_for("some/unknown-model")
    # cl100k tokenizes "hello world" as 2 tokens.
    assert counter("hello world") == 2


def test_fallback_counter_empty_text() -> None:
    counter = token_counter_for("some/unknown-model")
    assert counter("") == 0


def test_registry_cache_reset_seam() -> None:
    reset_tokenizer_cache()
    counter = token_counter_for("some/unknown-model")
    assert counter("a b") == 2


def test_model_tokenizer_load_failure_falls_back_to_cl100k(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the HF tokenizer cannot be loaded, the resolver degrades to cl100k."""
    from data_engineering_copilot.infrastructure import tokenizer_registry

    # Point the known model at a bogus repo so the real loader's internal
    # try/except trips and falls back to cl100k (no network, deterministic).
    monkeypatch.setitem(
        tokenizer_registry._MODEL_TOKENIZER_REPO,
        "nvidia/nemotron-3-embed-1b:free",
        "definitely-not-a-real-repo-xyz",
    )
    reset_tokenizer_cache()
    counter = token_counter_for("nvidia/nemotron-3-embed-1b:free")
    assert counter("hello world") == 2
