"""Tests for LLM single-attempt and embeddings retry resilience."""

from __future__ import annotations

import httpx
import pytest
import respx
from tenacity import wait_fixed

from data_engineering_copilot.infrastructure.llm_client import LLMClient, LLMClientError

OLLAMA_ENDPOINT = "/chat/completions"


@pytest.fixture
def client():
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


class TestLLMClientSingleAttempt:
    @pytest.mark.asyncio
    async def test_timeout_fails_after_single_attempt(self, client):
        with respx.mock:
            route = respx.post("http://localhost:11434/v1/chat/completions")
            route.side_effect = httpx.TimeoutException("persistent timeout")
            with pytest.raises(LLMClientError, match="timed out"):
                await client.generate("test prompt")
            assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_connect_error_fails_after_single_attempt(self, client):
        with respx.mock:
            route = respx.post("http://localhost:11434/v1/chat/completions")
            route.side_effect = httpx.ConnectError("connection refused")
            with pytest.raises(LLMClientError, match="Could not reach LLM provider"):
                await client.generate("test prompt")
            assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_os_error_fails_after_single_attempt(self, client):
        with respx.mock:
            route = respx.post("http://localhost:11434/v1/chat/completions")
            route.side_effect = OSError("connection reset")
            with pytest.raises(LLMClientError, match="Could not reach LLM provider"):
                await client.generate("test prompt")
            assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_success_without_retries(self, client):
        with respx.mock:
            route = respx.post("http://localhost:11434/v1/chat/completions")
            route.return_value = httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "answer"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                    "model": "llama3.2:3b",
                },
            )
            result = await client.generate("test prompt")
            assert result == "answer"
            assert route.call_count == 1


class TestEmbeddingRetry:
    @pytest.fixture
    def embeddings(self):
        from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

        return AsyncOllamaEmbeddings(model_name="nomic-embed-text", retry_wait=wait_fixed(0))

    @pytest.mark.asyncio
    async def test_network_error_retries_then_raises(self, embeddings):
        """Retryable network errors (TimeoutException) are retried 3 times
        by tenacity, then re-raised as the original exception type."""
        with respx.mock:
            respx.post(f"{embeddings.ollama_base_url}/api/embed").mock(side_effect=httpx.TimeoutException("timeout"))
            with pytest.raises(httpx.TimeoutException, match="timeout"):
                await embeddings._aollama_embed(["test"])

    @pytest.mark.asyncio
    async def test_connect_error_retries_then_raises(self, embeddings):
        """Retryable connection errors are retried 3 times, then re-raised."""
        with respx.mock:
            respx.post(f"{embeddings.ollama_base_url}/api/embed").mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            with pytest.raises(httpx.ConnectError, match="connection refused"):
                await embeddings._aollama_embed(["test"])

    @pytest.mark.asyncio
    async def test_success_after_retry_not_needed(self, embeddings):
        """When the first attempt succeeds, no retry is needed."""
        with respx.mock:
            respx.post(f"{embeddings.ollama_base_url}/api/embed").mock(
                return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
            )
            result = await embeddings._aollama_embed(["test"])
            assert len(result) == 1
