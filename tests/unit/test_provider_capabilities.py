"""Contract tests for provider capability gating.

`provider_capabilities.py` decides which generation hyperparameters are
silently emitted per provider (never errored). A wrong answer here means
params are dropped from payloads or rejected APIs receive them — both
silent failures. These tests pin the gate table so provider onboarding
(provider-onboarding skill step: update these sets) cannot regress
existing providers unnoticed.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.infrastructure.provider_capabilities import (
    SUPPORTS_SAMPLING_PENALTIES,
    SUPPORTS_SEED,
    SUPPORTS_STRUCTURED_OUTPUT,
    supports_sampling_penalties,
    supports_seed,
    supports_structured_output,
)


class TestFailClosedDefaults:
    """Unknown or absent provider -> no params emitted (fail-closed)."""

    @pytest.mark.parametrize(
        "fn",
        [supports_sampling_penalties, supports_seed, supports_structured_output],
    )
    def test_none_provider_is_false(self, fn):
        assert fn(None) is False

    @pytest.mark.parametrize(
        "fn",
        [supports_sampling_penalties, supports_seed, supports_structured_output],
    )
    def test_unknown_provider_is_false(self, fn):
        assert fn("definitely_not_a_provider") is False

    @pytest.mark.parametrize(
        "fn",
        [supports_sampling_penalties, supports_seed, supports_structured_output],
    )
    def test_empty_string_provider_is_false(self, fn):
        assert fn("") is False


class TestKnownProviderMatrix:
    """Representative membership pins per capability set."""

    @pytest.mark.parametrize(
        ("provider", "penalties", "seed", "structured"),
        [
            # (provider, penalties, seed, structured)
            ("openai", True, True, True),
            ("openrouter", True, True, True),
            ("ollama", True, False, True),  # degraded fallback: no seed
            ("vllm", False, False, True),  # structured-only provider
            ("groq", True, True, True),
            ("cerebras", True, True, True),
            ("gemini", True, True, True),
            ("deepseek", True, True, True),
            ("sambanova", True, False, True),
            ("mistral", True, True, True),
        ],
    )
    def test_capability_row(self, provider, penalties, seed, structured):
        assert supports_sampling_penalties(provider) is penalties
        assert supports_seed(provider) is seed
        assert supports_structured_output(provider) is structured


class TestTableInvariants:
    def test_anthropic_style_api_absent_everywhere(self):
        # Anthropic Messages API rejects penalties/seed/structured-output
        # params (module docstring); it must never appear in any set.
        for table in (
            SUPPORTS_SAMPLING_PENALTIES,
            SUPPORTS_SEED,
            SUPPORTS_STRUCTURED_OUTPUT,
        ):
            assert "anthropic" not in table

    def test_seed_support_implies_penalty_support(self):
        # Current invariant: anything accepting `seed` also accepts the
        # OpenAI-style penalty family (they travel in the same payload).
        assert SUPPORTS_SEED <= SUPPORTS_SAMPLING_PENALTIES
