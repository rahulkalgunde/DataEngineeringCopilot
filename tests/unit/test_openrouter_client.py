"""Tests for OpenRouterLLMClient — async httpx-based OpenRouter generation client."""

from __future__ import annotations

import httpx
import pytest
import respx

from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.async_openrouter_client import (
    OpenRouterError,
    OpenRouterLLMClient,
)


@pytest.fixture
def client():
    return OpenRouterLLMClient(
        api_key="sk-or-v1-test-key",
        model="anthropic/claude-3.5-sonnet",
        timeout_seconds=120,
    )


def test_init(client):
    assert client.model == "anthropic/claude-3.5-sonnet"
    assert client.api_key == "sk-or-v1-test-key"
    assert client.timeout_seconds == 120


def test_last_usage_initial(client):
    usage = client.last_usage
    assert isinstance(usage, LLMUsage)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0


@pytest.mark.asyncio
async def test_generate_success(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Delta Lake supports ACID."}}],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 10,
                    },
                    "model": "anthropic/claude-3.5-sonnet",
                },
            )
        )
        result = await client.generate("What is Delta Lake?")
        assert result == "Delta Lake supports ACID."
        assert client.last_usage.prompt_tokens == 50
        assert client.last_usage.completion_tokens == 10
        assert client.last_usage.model == "anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_generate_sends_correct_payload(client):
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "answer"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                    "model": "anthropic/claude-3.5-sonnet",
                },
            )
        )
        await client.generate("test prompt")
        import json

        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "anthropic/claude-3.5-sonnet"
        assert body["messages"] == [{"role": "user", "content": "test prompt"}]
        assert body["temperature"] == 0.05


@pytest.mark.asyncio
async def test_generate_sends_auth_header(client):
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "model": "anthropic/claude-3.5-sonnet",
                },
            )
        )
        await client.generate("test")
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-or-v1-test-key"


@pytest.mark.asyncio
async def test_generate_empty_content_returns_empty(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": ""}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 0},
                    "model": "anthropic/claude-3.5-sonnet",
                },
            )
        )
        result = await client.generate("test")
        assert result == ""


@pytest.mark.asyncio
async def test_generate_http_error(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )
        with pytest.raises(OpenRouterError, match="Unauthorized"):
            await client.generate("test")


@pytest.mark.asyncio
async def test_generate_connection_error(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with pytest.raises(OpenRouterError, match="Could not reach OpenRouter"):
            await client.generate("test")


@pytest.mark.asyncio
async def test_generate_timeout_error(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        with pytest.raises(OpenRouterError, match="timed out"):
            await client.generate("test")


@pytest.mark.asyncio
async def test_generate_strips_thinking_tags(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "<think>reasoning</think>Final answer."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10},
                    "model": "anthropic/claude-3.5-sonnet",
                },
            )
        )
        result = await client.generate("test")
        assert "<think>" not in result
        assert "Final answer." in result


@pytest.mark.asyncio
async def test_generate_custom_temperature(client):
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "model": "anthropic/claude-3.5-sonnet",
                },
            )
        )
        await client.generate("test", temperature=0.7)
        import json

        body = json.loads(route.calls.last.request.content)
        assert body["temperature"] == 0.7


def test_custom_base_url():
    c = OpenRouterLLMClient(
        api_key="key",
        model="model",
        base_url="https://custom.api.com/v1",
        timeout_seconds=60,
    )
    assert c._base_url == "https://custom.api.com/v1"
