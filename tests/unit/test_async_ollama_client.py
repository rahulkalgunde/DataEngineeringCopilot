"""Tests for LLMClient with Ollama-style parametrization."""

from __future__ import annotations

import httpx
import pytest
import respx

from data_engineering_copilot.infrastructure.llm_client import LLMClient, LLMClientError


@pytest.fixture
def ollama_client():
    return LLMClient(
        base_url="http://localhost:11434/v1",
        model="llama3.2:3b",
        timeout_seconds=300,
        extra_body={
            "options": {
                "num_ctx": 4096,
                "num_predict": 512,
            }
        },
    )


def test_init(ollama_client):
    assert ollama_client.model == "llama3.2:3b"
    assert ollama_client.base_url == "http://localhost:11434/v1"
    assert ollama_client.timeout_seconds == 300
    assert ollama_client._extra_body == {"options": {"num_ctx": 4096, "num_predict": 512}}


def test_extract_strips_thinking_block(ollama_client):
    raw = "<think>reasoning here</think>Final answer."
    result = ollama_client._extract_final_response(raw)
    assert "<think>" not in result
    assert "Final answer." in result


def test_extract_reasoning_only_returns_empty(ollama_client):
    raw = "<think>I ran out of tokens."
    result = ollama_client._extract_final_response(raw)
    assert result == ""


@pytest.mark.asyncio
async def test_generate_success(ollama_client):
    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "<think>I should reason.</think>\nDelta Lake supports ACID transactions."
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                    "model": "llama3.2:3b",
                },
            )
        )
        result = await ollama_client.generate("Answer from context")
        assert result == "Delta Lake supports ACID transactions."


@pytest.mark.asyncio
async def test_generate_strips_thinking_block(ollama_client):
    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "<think>reasoning here</think>Final answer here."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                    "model": "llama3.2:3b",
                },
            )
        )
        result = await ollama_client.generate("test")
        assert "<think>" not in result
        assert "Final answer here" in result


@pytest.mark.asyncio
async def test_generate_reasoning_only_returns_empty(ollama_client):
    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "<think>I ran out of tokens."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 0},
                    "model": "llama3.2:3b",
                },
            )
        )
        result = await ollama_client.generate("test")
        assert result == ""


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_http_error(ollama_client):
    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with pytest.raises(LLMClientError, match="Could not reach LLM provider"):
            await ollama_client.generate("test")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_timeout_error(ollama_client):
    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        with pytest.raises(LLMClientError, match="timed out"):
            await ollama_client.generate("test")


@pytest.mark.asyncio
async def test_generate_empty_response(ollama_client):
    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": ""}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 0},
                    "model": "llama3.2:3b",
                },
            )
        )
        result = await ollama_client.generate("test")
        assert result == ""


@pytest.mark.asyncio
async def test_generate_sends_keep_alive_and_uses_phase_timeouts():
    client = LLMClient(
        base_url="http://localhost:11434/v1",
        model="llama3.2:3b",
        timeout_seconds=180,
        keep_alive="10m",
        connect_timeout_seconds=5,
        pool_timeout_seconds=5,
    )
    with respx.mock:
        route = respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "answer"}}], "usage": {}},
            )
        )
        assert await client.generate("test") == "answer"
        body = route.calls[0].request.content
        assert b'"keep_alive":"10m"' in body
        assert client.timeout_seconds == 180
        assert client.connect_timeout_seconds == 5
        assert client.pool_timeout_seconds == 5
