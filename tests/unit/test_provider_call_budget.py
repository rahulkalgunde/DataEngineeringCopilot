"""Tests for the external-provider call budget guardrail."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from data_engineering_copilot.infrastructure.provider_fallback import (
    _CallBudgetTracker,
    _estimate_request_tokens,
)


def test_budget_tracks_calls_within_limit() -> None:
    budget = _CallBudgetTracker(max_calls=3, max_cost_usd=1000.0)
    budget.check_and_track("openrouter", 100)
    budget.check_and_track("openrouter", 100)
    budget.check_and_track("openrouter", 100)


def test_budget_raises_when_call_limit_exceeded() -> None:
    budget = _CallBudgetTracker(max_calls=2, max_cost_usd=1000.0)
    budget.check_and_track("openrouter", 100)
    budget.check_and_track("openrouter", 100)
    with pytest.raises(RuntimeError, match="call budget exceeded"):
        budget.check_and_track("openrouter", 100)


def test_budget_raises_when_cost_limit_exceeded() -> None:
    budget = _CallBudgetTracker(max_calls=1000, max_cost_usd=0.00001)
    with pytest.raises(RuntimeError, match="cost budget exceeded"):
        budget.check_and_track("nvidia", 100_000_000)


def test_estimate_request_tokens_from_embedding_request_texts() -> None:
    tokens = _estimate_request_tokens(SimpleNamespace(texts=["a" * 400, "b" * 400]))
    assert tokens == 200


def test_estimate_request_tokens_from_string_prompt() -> None:
    assert _estimate_request_tokens(SimpleNamespace(prompt="x" * 400)) == 100


def test_estimate_request_tokens_from_messages() -> None:
    class _Msg:
        content: str

    msg = _Msg()
    msg.content = "y" * 800
    assert _estimate_request_tokens(SimpleNamespace(messages=[msg])) == 200


def test_estimate_request_tokens_unknown_shape_returns_one() -> None:
    assert _estimate_request_tokens(object()) == 1
