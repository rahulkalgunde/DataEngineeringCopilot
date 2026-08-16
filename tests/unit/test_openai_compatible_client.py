"""Tests for LLMClient with OpenRouter-style parametrization."""

from __future__ import annotations

import httpx
import pytest
import respx

from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.llm_client import (
    SYSTEM_BLOCK_SEPARATOR,
    LLMClient,
    LLMClientError,
    build_chat_messages,
)


@pytest.fixture
def client():
    return LLMClient(
        api_key="sk-or-v1-test-key",
        model="anthropic/claude-3.5-sonnet",
        timeout_seconds=120,
        base_url="https://openrouter.ai/api/v1",
        extra_headers={"HTTP-Referer": "https://data-engineering-copilot.local"},
    )


def test_init(client):
    assert client.model == "anthropic/claude-3.5-sonnet"
    assert client.api_key == "sk-or-v1-test-key"
    assert client.timeout_seconds == 120
    assert client.base_url == "https://openrouter.ai/api/v1"


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
async def test_generate_sends_constructor_max_tokens():
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
        capped = LLMClient(
            api_key="sk-test",
            model="anthropic/claude-3.5-sonnet",
            base_url="https://openrouter.ai/api/v1",
            max_tokens=4096,
        )
        await capped.generate("test")
        import json

        body = json.loads(route.calls.last.request.content)
        assert body["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_generate_max_completion_tokens_field_name():
    with respx.mock:
        route = respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "model": "gpt-oss-120b",
                },
            )
        )
        capped = LLMClient(
            api_key="sk-test",
            model="gpt-oss-120b",
            base_url="https://api.cerebras.ai/v1",
            max_tokens=1024,
            max_tokens_field="max_completion_tokens",
        )
        await capped.generate("test")
        import json

        body = json.loads(route.calls.last.request.content)
        assert body["max_completion_tokens"] == 1024
        assert "max_tokens" not in body


@pytest.mark.asyncio
async def test_generate_omits_max_tokens_when_unset(client):
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
        import json

        body = json.loads(route.calls.last.request.content)
        assert "max_tokens" not in body


def test_build_chat_messages_splits_system_block():
    prompt = f"## SYSTEM\nYou are an assistant.\n{SYSTEM_BLOCK_SEPARATOR}Context: docs\nQuestion: hi?"
    messages = build_chat_messages(prompt)
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "## SYSTEM\nYou are an assistant."}
    assert messages[1] == {"role": "user", "content": "Context: docs\nQuestion: hi?"}


def test_build_chat_messages_legacy_single_user_when_no_marker():
    messages = build_chat_messages("Just a prompt.")
    assert messages == [{"role": "user", "content": "Just a prompt."}]


def test_build_chat_messages_falls_back_when_one_side_empty():
    empty_system = f"{SYSTEM_BLOCK_SEPARATOR}only user content"
    assert build_chat_messages(empty_system) == [{"role": "user", "content": empty_system}]
    empty_user = f"only system content{SYSTEM_BLOCK_SEPARATOR}   "
    assert build_chat_messages(empty_user) == [{"role": "user", "content": empty_user}]


@pytest.mark.asyncio
async def test_generate_sends_system_and_user_messages(client):
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
        prompt = f"## SYSTEM\nYou are an assistant.\n{SYSTEM_BLOCK_SEPARATOR}What is Delta Lake?"
        await client.generate(prompt)
        import json

        body = json.loads(route.calls.last.request.content)
        assert body["messages"] == [
            {"role": "system", "content": "## SYSTEM\nYou are an assistant."},
            {"role": "user", "content": "What is Delta Lake?"},
        ]


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
async def test_generate_null_content_returns_empty(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": None}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 0},
                    "model": "anthropic/claude-3.5-sonnet",
                },
            )
        )
        result = await client.generate("test")
        assert result == ""


@pytest.mark.asyncio
async def test_generate_missing_usage_ok(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}], "model": "anthropic/claude-3.5-sonnet"},
            )
        )
        result = await client.generate("test")
        assert result == "ok"
        assert client.last_usage.prompt_tokens == 0
        assert client.last_usage.completion_tokens == 0


@pytest.mark.asyncio
async def test_generate_http_error(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )
        with pytest.raises(LLMClientError, match="401 Unauthorized"):
            await client.generate("test")


@pytest.mark.asyncio
async def test_generate_401_preserves_model_error_body(client):
    """A 401 from an OpenAI-compatible gateway must carry the server body so the
    categorizer can tell a bad key from a wrong model (opencodego ModelError)."""
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"message": 'ModelError: "google/gemma-4-31b-it:free" is not supported'}},
            )
        )
        with pytest.raises(LLMClientError) as excinfo:
            await client.generate("test")
        err = excinfo.value
        assert err.status_code == 401
        assert "not supported" in err.response_body


def test_extract_error_body_structured_envelope():
    import httpx

    from data_engineering_copilot.infrastructure.llm_client import _extract_error_body

    resp = httpx.Response(
        401,
        request=httpx.Request("POST", "http://x"),
        json={"error": {"message": "Model is not supported"}},
    )
    assert _extract_error_body(resp) == "Model is not supported"


def test_extract_error_body_flat_string():
    import httpx

    from data_engineering_copilot.infrastructure.llm_client import _extract_error_body

    resp = httpx.Response(
        401,
        request=httpx.Request("POST", "http://x"),
        text='ModelError: "model-x" is not supported',
    )
    assert "not supported" in _extract_error_body(resp)


def test_extract_error_body_empty():
    import httpx

    from data_engineering_copilot.infrastructure.llm_client import _extract_error_body

    resp = httpx.Response(401, request=httpx.Request("POST", "http://x"))
    assert _extract_error_body(resp) == ""


@pytest.mark.asyncio
async def test_generate_connection_error(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with pytest.raises(LLMClientError, match="Could not reach LLM provider"):
            await client.generate("test")


@pytest.mark.asyncio
async def test_generate_timeout_error(client):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        with pytest.raises(LLMClientError, match="timed out"):
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
    c = LLMClient(
        api_key="key",
        model="model",
        base_url="https://custom.api.com/v1",
        timeout_seconds=60,
    )
    assert c.base_url == "https://custom.api.com/v1"
