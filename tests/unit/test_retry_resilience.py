"""Tests for retry and resilience patterns (tenacity backoff)."""

from __future__ import annotations

import httpx
import pytest
import respx

from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient, AsyncOllamaError


@pytest.fixture
def client():
    return AsyncOllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        timeout_seconds=300,
        num_ctx=4096,
        num_predict=512,
    )


class TestOllamaRetry:
    @pytest.mark.asyncio
    async def test_retries_on_timeout_then_succeeds(self, client):
        with respx.mock:
            route = respx.post("http://localhost:11434/api/generate")
            route.side_effect = [
                httpx.TimeoutException("timeout 1"),
                httpx.TimeoutException("timeout 2"),
                httpx.Response(200, json={"response": "success after retry", "done_reason": "stop"}),
            ]
            result = await client.generate("test prompt")
            assert result == "success after retry"
            assert route.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_connect_error_then_succeeds(self, client):
        with respx.mock:
            route = respx.post("http://localhost:11434/api/generate")
            route.side_effect = [
                httpx.ConnectError("connection refused"),
                httpx.Response(200, json={"response": "connected", "done_reason": "stop"}),
            ]
            result = await client.generate("test prompt")
            assert result == "connected"
            assert route.call_count == 2

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(self, client):
        with respx.mock:
            route = respx.post("http://localhost:11434/api/generate")
            route.side_effect = httpx.TimeoutException("persistent timeout")
            with pytest.raises(AsyncOllamaError, match="timed out"):
                await client.generate("test prompt")
            assert route.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_os_error(self, client):
        with respx.mock:
            route = respx.post("http://localhost:11434/api/generate")
            route.side_effect = [
                OSError("connection reset"),
                httpx.Response(200, json={"response": "recovered", "done_reason": "stop"}),
            ]
            result = await client.generate("test prompt")
            assert result == "recovered"
            assert route.call_count == 2


class TestEmbeddingRetry:
    @pytest.fixture
    def embeddings(self):
        from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

        return AsyncOllamaEmbeddings(model_name="nomic-embed-text")

    @pytest.mark.asyncio
    async def test_network_error_wrapped_in_embedding_error(self, embeddings):
        """Network errors from Ollama are caught by the except block inside
        _aollama_embed_single_batch and wrapped in EmbeddingError immediately.

        The tenacity retry decorator on _aollama_embed_single_batch cannot
        see the original httpx exception because the except Exception block
        catches it first and re-raises as EmbeddingError.
        """
        with respx.mock:
            respx.post(f"{embeddings.ollama_base_url}/api/embed").mock(side_effect=httpx.TimeoutException("timeout"))
            with pytest.raises(RuntimeError, match="Failed to get embeddings from Ollama"):
                await embeddings._aollama_embed(["test"])

    @pytest.mark.asyncio
    async def test_connect_error_wrapped_in_embedding_error(self, embeddings):
        with respx.mock:
            respx.post(f"{embeddings.ollama_base_url}/api/embed").mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            with pytest.raises(RuntimeError, match="Failed to get embeddings from Ollama"):
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
