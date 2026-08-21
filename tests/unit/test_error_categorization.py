"""Task 11: single source of truth for provider error categorization."""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
from data_engineering_copilot.infrastructure.error_categorization import categorize_provider_error
from data_engineering_copilot.infrastructure.llm_client import LLMClientError


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, "ModelError: model foo not supported", ProviderErrorCategory.INVALID_REQUEST),
        (401, "invalid api key", ProviderErrorCategory.AUTHENTICATION_ERROR),
        (429, "", ProviderErrorCategory.RATE_LIMITED),
        (500, "", ProviderErrorCategory.TEMPORARY_UNAVAILABLE),
        (400, "", ProviderErrorCategory.INVALID_REQUEST),
        (422, "", ProviderErrorCategory.INVALID_REQUEST),
    ],
)
def test_status_and_body_mapping(status, body, expected):
    exc = LLMClientError("x", status_code=status, response_body=body)
    err = categorize_provider_error(exc, "prov", "model")
    assert err.category is expected


def test_timeout_maps_retryable():
    err = categorize_provider_error(TimeoutError("t"), "prov", "model")
    assert err.category is ProviderErrorCategory.RETRYABLE


def test_message_fallbacks():
    rate = categorize_provider_error(RuntimeError("rate limit hit"), "p", "m")
    assert rate.category is ProviderErrorCategory.RATE_LIMITED
    quota = categorize_provider_error(RuntimeError("quota exceeded"), "p", "m")
    assert quota.category is ProviderErrorCategory.QUOTA_EXCEEDED
    auth = categorize_provider_error(RuntimeError("unauthorized access"), "p", "m")
    assert auth.category is ProviderErrorCategory.AUTHENTICATION_ERROR
    unknown = categorize_provider_error(RuntimeError("something odd"), "p", "m")
    assert unknown.category is ProviderErrorCategory.PERMANENT_ERROR


def test_factory_categorizers_delegate_to_shared():
    """The factory's embedding/rerank categorizers must be the shared one."""
    from data_engineering_copilot.factory import _categorize_embedding_error, _categorize_rerank_error
    from data_engineering_copilot.infrastructure.error_categorization import categorize_provider_error
    from data_engineering_copilot.infrastructure.provider_fallback import _default_categorizer

    assert _categorize_embedding_error is categorize_provider_error
    assert _categorize_rerank_error is categorize_provider_error
    assert _default_categorizer is categorize_provider_error


def test_chain_generate_signature_honest():
    """Task 12: chain.generate() must not accept args it silently ignores."""
    import inspect

    from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain

    params = inspect.signature(ProviderFallbackChain.generate).parameters
    assert list(params) == ["self", "prompt"], f"unexpected generate signature: {list(params)}"
