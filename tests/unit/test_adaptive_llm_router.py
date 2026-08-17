"""Tests for LLM error categorizer (moved from adaptive_llm_router to provider_fallback)."""

from __future__ import annotations

import httpx

from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
from data_engineering_copilot.infrastructure.llm_client import LLMClientError
from data_engineering_copilot.infrastructure.provider_fallback import _default_categorizer as _categorize_llm_error


class TestErrorCategorization:
    def test_http_429_is_rate_limited(self):
        resp = httpx.Response(429, request=httpx.Request("POST", "http://example.com"))
        exc = httpx.HTTPStatusError("too many", request=resp.request, response=resp)
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.RATE_LIMITED

    def test_http_401_is_auth_error(self):
        resp = httpx.Response(401, request=httpx.Request("POST", "http://example.com"))
        exc = httpx.HTTPStatusError("unauth", request=resp.request, response=resp)
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.AUTHENTICATION_ERROR

    def test_http_403_is_auth_error(self):
        resp = httpx.Response(403, request=httpx.Request("POST", "http://example.com"))
        exc = httpx.HTTPStatusError("forbidden", request=resp.request, response=resp)
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.AUTHENTICATION_ERROR

    def test_http_400_is_invalid_request(self):
        resp = httpx.Response(400, request=httpx.Request("POST", "http://example.com"))
        exc = httpx.HTTPStatusError("bad", request=resp.request, response=resp)
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.INVALID_REQUEST

    def test_http_503_is_temporary_unavailable(self):
        resp = httpx.Response(503, request=httpx.Request("POST", "http://example.com"))
        exc = httpx.HTTPStatusError("down", request=resp.request, response=resp)
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.TEMPORARY_UNAVAILABLE

    def test_timeout_is_retryable(self):
        exc = httpx.TimeoutException("timed out")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.RETRYABLE

    def test_connect_error_is_retryable(self):
        exc = httpx.ConnectError("connection refused")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.RETRYABLE

    def test_llm_client_error_uses_structured_status_code(self):
        exc = LLMClientError("rate limit window exhausted", status_code=429, retry_after=5.0)
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.RATE_LIMITED
        assert err.retry_after == 5.0

    def test_llm_client_error_rate_limit_text(self):
        exc = LLMClientError("Rate limit exceeded")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.RATE_LIMITED

    def test_llm_client_error_timeout_text(self):
        exc = LLMClientError("timed out after 120 seconds")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.RETRYABLE

    def test_llm_client_error_401_text(self):
        exc = LLMClientError("returned HTTP 401 Unauthorized")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.AUTHENTICATION_ERROR

    def test_llm_client_error_401_model_not_supported_is_invalid_request(self):
        """A 401 carrying a ``ModelError`` (wrong model) is NOT an auth failure:
        it must be categorised ``INVALID_REQUEST`` so the provider is not marked
        down as an authentication problem (opencodego regression)."""
        exc = LLMClientError(
            'LLM provider returned 401 Unauthorized — ModelError: "google/gemma-4-31b-it:free" is not supported. '
            "Check your API key.",
            status_code=401,
            response_body='ModelError: "google/gemma-4-31b-it:free" is not supported',
        )
        err = _categorize_llm_error(exc, "opencodego", "google/gemma-4-31b-it:free")
        assert err.category == ProviderErrorCategory.INVALID_REQUEST

    def test_http_401_model_not_supported_body_is_invalid_request(self):
        resp = httpx.Response(
            401,
            request=httpx.Request("POST", "http://example.com"),
            text='{"error": {"message": "Model X is not supported"}}',
        )
        exc = httpx.HTTPStatusError("401", request=resp.request, response=resp)
        err = _categorize_llm_error(exc, "opencodego", "model-x")
        assert err.category == ProviderErrorCategory.INVALID_REQUEST

    def test_llm_client_error_401_real_auth_stays_auth_error(self):
        exc = LLMClientError(
            "LLM provider returned 401 Unauthorized. Check your API key.",
            status_code=401,
            response_body='{"error": {"message": "Invalid API key"}}',
        )
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.AUTHENTICATION_ERROR

    def test_llm_client_error_401_no_body_stays_auth_error(self):
        exc = LLMClientError("LLM provider returned 401 Unauthorized. Check your API key.", status_code=401)
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.AUTHENTICATION_ERROR

    def test_llm_client_error_connection_text(self):
        exc = LLMClientError("Could not reach LLM provider")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.RETRYABLE

    def test_unknown_error_is_permanent(self):
        exc = ValueError("something unexpected")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.PERMANENT_ERROR
