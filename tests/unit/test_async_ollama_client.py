"""Tests for AsyncOllamaClient — async httpx-based Ollama generation client."""

from __future__ import annotations

import httpx
import pytest
import respx

from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient, AsyncOllamaError


@pytest.fixture
def async_client():
    return AsyncOllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        timeout_seconds=300,
        num_ctx=4096,
        num_predict=512,
    )


def test_init(async_client):
    assert async_client.model == "llama3.2:3b"
    assert async_client.base_url == "http://localhost:11434"
    assert async_client.timeout_seconds == 300
    assert async_client.num_ctx == 4096
    assert async_client.num_predict == 512


def test_prompt_format(async_client):
    prompt = async_client._format_raw_chat_prompt("What is Delta Lake?")
    assert "DataEngineeringCopilot" in prompt
    assert "provided context" in prompt
    assert "What is Delta Lake?" in prompt


def test_extract_strips_thinking_block(async_client):
    raw = "<think>reasoning here</think>Final answer."
    result = async_client._extract_final_response(raw)
    assert "<think>" not in result
    assert "Final answer." in result


def test_extract_reasoning_only_returns_empty(async_client):
    raw = "<think>I ran out of tokens."
    result = async_client._extract_final_response(raw)
    assert result == ""


@pytest.mark.asyncio
async def test_generate_success(async_client):
    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(
                200,
                json={
                    "response": "<think>I should reason.</think>\nDelta Lake supports ACID transactions.",
                    "done_reason": "stop",
                },
            )
        )
        result = await async_client.generate("Answer from context")
        assert result == "Delta Lake supports ACID transactions."


@pytest.mark.asyncio
async def test_generate_strips_thinking_block(async_client):
    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(
                200,
                json={
                    "response": "<think>reasoning here</think>Final answer here.",
                    "done_reason": "stop",
                },
            )
        )
        result = await async_client.generate("test")
        assert "<think>" not in result
        assert "Final answer here" in result


@pytest.mark.asyncio
async def test_generate_reasoning_only_raises(async_client):
    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(
                200,
                json={
                    "response": "<think>I ran out of tokens.",
                    "done_reason": "length",
                },
            )
        )
        with pytest.raises(AsyncOllamaError, match="spent its output budget on reasoning"):
            await async_client.generate("test")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_http_error(async_client):
    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(side_effect=httpx.ConnectError("Connection refused"))
        with pytest.raises(AsyncOllamaError, match="Could not reach Ollama"):
            await async_client.generate("test")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_timeout_error(async_client):
    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(side_effect=httpx.TimeoutException("Request timed out"))
        with pytest.raises(AsyncOllamaError, match="timed out"):
            await async_client.generate("test")


@pytest.mark.asyncio
async def test_generate_empty_response(async_client):
    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(
                200,
                json={
                    "response": "",
                    "done_reason": "stop",
                },
            )
        )
        with pytest.raises(AsyncOllamaError, match="returned no final answer"):
            await async_client.generate("test")
