from __future__ import annotations

import logging
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
from data_engineering_copilot.infrastructure.adaptive_llm_router import (
    AdaptiveLLMRouter,
    _categorize_llm_error,
)
from data_engineering_copilot.infrastructure.llm_client import LLMClientError
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter


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

    def test_llm_client_error_connection_text(self):
        exc = LLMClientError("Could not reach LLM provider")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.RETRYABLE

    def test_unknown_error_is_permanent(self):
        exc = ValueError("something unexpected")
        err = _categorize_llm_error(exc, "openrouter", "model")
        assert err.category == ProviderErrorCategory.PERMANENT_ERROR


class TestAdaptiveLLMRouter:
    @pytest.fixture
    def health(self):
        return ProviderHealthRegistry(
            success_rate_weight=0.6,
            latency_weight=0.2,
            recency_weight=0.2,
            consecutive_failure_penalty=0.3,
            default_cooldown_seconds=60.0,
        )

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        client.model = "test-model"
        client.last_usage = MagicMock()
        return client

    @pytest.fixture
    def router(self, health, mock_client):
        router = AdaptiveLLMRouter(
            clients=[("openrouter", mock_client)],
            health=health,
        )
        health.register_provider("openrouter", ["test-model"])
        return router

    @pytest.mark.asyncio
    async def test_successful_generation(self, router, mock_client, health):
        mock_client.generate = AsyncMock(return_value="Hello world")
        health.register_provider("openrouter", ["test-model"])

        result = await router.generate("test prompt")

        assert result == "Hello world"
        mock_client.generate.assert_called_once_with(
            prompt="test prompt",
            temperature=None,
            num_predict=None,
            num_ctx=None,
        )

    @pytest.mark.asyncio
    async def test_fails_after_single_attempt(self, router, mock_client, health):
        mock_client.generate = AsyncMock(side_effect=LLMClientError("timed out"))
        health.register_provider("openrouter", ["test-model"])

        with pytest.raises(LLMClientError, match="All LLM providers in adaptive fallback chain failed"):
            await router.generate("test prompt")

        assert mock_client.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_failover_on_first_error_no_same_provider_retry(self, health):
        client_a = AsyncMock()
        client_a.model = "a-model"
        client_a.generate = AsyncMock(side_effect=LLMClientError("timed out"))
        client_a.last_usage = MagicMock()

        client_b = AsyncMock()
        client_b.model = "b-model"
        client_b.generate = AsyncMock(return_value="From B")
        client_b.last_usage = MagicMock()

        router = AdaptiveLLMRouter(
            clients=[("openrouter", client_a), ("groq", client_b)],
            health=health,
        )
        health.register_provider("openrouter", ["a-model"])
        health.register_provider("groq", ["b-model"])

        result = await router.generate("test prompt")

        assert result == "From B"
        assert client_a.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_failover_to_ollama_when_external_exhausted(self, health):
        ext_client = AsyncMock()
        ext_client.model = "ext-model"
        ext_client.generate = AsyncMock(side_effect=LLMClientError("timed out"))
        ext_client.last_usage = MagicMock()

        ollama_client = AsyncMock()
        ollama_client.model = "ollama-model"
        ollama_client.generate = AsyncMock(return_value="Ollama response")
        ollama_client.last_usage = MagicMock()

        router = AdaptiveLLMRouter(
            clients=[("openrouter", ext_client), ("ollama", ollama_client)],
            health=health,
        )
        health.register_provider("openrouter", ["ext-model"])
        health.register_provider("ollama", ["ollama-model"])

        result = await router.generate("test prompt")

        assert result == "Ollama response"
        ext_client.generate.assert_called()
        ollama_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limited_with_other_provider_failovers(self, health):
        or_client = AsyncMock()
        or_client.model = "or-model"
        or_client.generate = AsyncMock(side_effect=LLMClientError("Rate limit exceeded"))
        or_client.last_usage = MagicMock()

        nv_client = AsyncMock()
        nv_client.model = "nv-model"
        nv_client.generate = AsyncMock(return_value="NVIDIA response")
        nv_client.last_usage = MagicMock()

        router = AdaptiveLLMRouter(
            clients=[("openrouter", or_client), ("nvidia", nv_client)],
            health=health,
        )
        health.register_provider("openrouter", ["or-model"])
        health.register_provider("nvidia", ["nv-model"])

        result = await router.generate("test prompt")

        assert result == "NVIDIA response"

    @pytest.mark.asyncio
    async def test_skips_providers_in_cooldown(self, health):
        client_a = AsyncMock()
        client_a.model = "model-a"
        client_a.generate = AsyncMock(return_value="From A")
        client_a.last_usage = MagicMock()

        client_b = AsyncMock()
        client_b.model = "model-b"
        client_b.generate = AsyncMock(return_value="From B")
        client_b.last_usage = MagicMock()

        router = AdaptiveLLMRouter(
            clients=[("openrouter", client_a), ("nvidia", client_b)],
            health=health,
        )
        health.register_provider("openrouter", ["model-a"])
        health.register_provider("nvidia", ["model-b"])

        health.mark_provider_cooldown("openrouter", duration=9999)

        result = await router.generate("test prompt")

        assert result == "From B"
        assert client_a.generate.call_count == 0

    @pytest.mark.asyncio
    async def test_skips_provider_over_rpm_window_without_call(self, health):
        rl = SlidingWindowRateLimiter(rpm_limit=1, rpd_limit=1000)
        assert await rl.try_acquire() is True

        client = AsyncMock()
        client.model = "or-model"
        client.generate = AsyncMock(return_value="must not be called")
        client.last_usage = MagicMock()

        router = AdaptiveLLMRouter(
            clients=[("openrouter", client)],
            health=health,
            rate_limiters={"openrouter": rl},
        )
        health.register_provider("openrouter", ["or-model"])

        with pytest.raises(LLMClientError, match="All LLM providers"):
            await router.generate("test prompt")

        client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_after_propagates_to_cooldown(self, health):
        client = AsyncMock()
        client.model = "or-model"
        client.generate = AsyncMock(side_effect=LLMClientError("rate limited", status_code=429, retry_after=5.0))
        client.last_usage = MagicMock()

        router = AdaptiveLLMRouter(clients=[("openrouter", client)], health=health)
        health.register_provider("openrouter", ["or-model"])

        with pytest.raises(LLMClientError):
            await router.generate("test prompt")

        mh = health.get_model_health("openrouter", "or-model")
        assert mh is not None
        assert mh.cooldown_until > time.monotonic()

    @pytest.mark.asyncio
    async def test_authentication_error_fails_immediately(self, router, mock_client, health):
        mock_client.generate = AsyncMock(side_effect=LLMClientError("returned HTTP 401 Unauthorized"))
        health.register_provider("openrouter", ["test-model"])

        with pytest.raises(LLMClientError):
            await router.generate("test prompt")

        assert mock_client.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_passes_params_to_client(self, router, mock_client, health):
        mock_client.generate = AsyncMock(return_value="ok")
        health.register_provider("openrouter", ["test-model"])

        await router.generate("prompt", temperature=0.7, num_predict=100, num_ctx=2048)

        mock_client.generate.assert_called_once_with(
            prompt="prompt",
            temperature=0.7,
            num_predict=100,
            num_ctx=2048,
        )

    @pytest.mark.asyncio
    async def test_streaming_fallback_to_generate(self, router, mock_client, health):
        mock_client.generate_stream = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "fail", request=MagicMock(), response=httpx.Response(503, request=MagicMock())
            )
        )
        mock_client.generate = AsyncMock(return_value="fallback text")
        health.register_provider("openrouter", ["test-model"])

        tokens = []
        async for token in router.generate_stream("prompt"):
            tokens.append(token)

        assert tokens == ["fallback text"]

    @pytest.mark.asyncio
    async def test_streaming_success(self, router, mock_client, health):
        async def _stream(*args, **kwargs):
            yield "token1"
            yield "token2"

        mock_client.generate_stream = _stream
        health.register_provider("openrouter", ["test-model"])

        tokens = []
        async for token in router.generate_stream("prompt"):
            tokens.append(token)

        assert tokens == ["token1", "token2"]

    @pytest.mark.asyncio
    async def test_skipped_provider_decision_is_logged(self, health, caplog):
        rl = SlidingWindowRateLimiter(rpm_limit=1, rpd_limit=1000)
        assert await rl.try_acquire() is True

        client = AsyncMock()
        client.model = "or-model"
        client.generate = AsyncMock(return_value="must not be called")
        client.last_usage = MagicMock()

        router = AdaptiveLLMRouter(
            clients=[("openrouter", client)],
            health=health,
            rate_limiters={"openrouter": rl},
        )
        health.register_provider("openrouter", ["or-model"])

        with caplog.at_level(logging.INFO), pytest.raises(LLMClientError):
            await router.generate("test prompt")

        assert "llm_provider_skipped" in caplog.text
        assert "rate_limit" in caplog.text

    def test_model_property(self, router):
        assert router.model == "test-model"

    def test_model_property_empty(self):
        empty = AdaptiveLLMRouter([], health=MagicMock())
        assert empty.model == ""

    @pytest.mark.asyncio
    async def test_close_calls_all_clients(self):
        client_a = AsyncMock()
        client_b = AsyncMock()
        router = AdaptiveLLMRouter(
            [("a", client_a), ("b", client_b)],
            health=MagicMock(),
        )

        await router.close()

        client_a.close.assert_awaited_once()
        client_b.close.assert_awaited_once()
