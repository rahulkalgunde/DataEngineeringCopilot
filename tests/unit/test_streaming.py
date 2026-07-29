"""Unit tests for token-by-token streaming."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from data_engineering_copilot.infrastructure.llm_client import LLMClient


class _MockStreamContext:
    """Async context manager that yields a mock response."""

    def __init__(self, response: MagicMock) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class TestLLMClientGenerateStream:
    @pytest.fixture
    def client(self) -> LLMClient:
        return LLMClient(
            base_url="http://localhost:11434/v1",
            model="test-model",
            api_key="",
            timeout_seconds=10,
        )

    async def test_generate_stream_yields_tokens(self, client: LLMClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = MagicMock()
        mock_client.stream.return_value = _MockStreamContext(mock_response)

        with patch.object(client, "_get_client", new_callable=AsyncMock, return_value=mock_client):
            tokens = []
            async for token in client.generate_stream("test prompt"):
                tokens.append(token)

        assert tokens == ["Hello", " world"]

    async def test_generate_stream_empty_response(self, client: LLMClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = MagicMock()
        mock_client.stream.return_value = _MockStreamContext(mock_response)

        with patch.object(client, "_get_client", new_callable=AsyncMock, return_value=mock_client):
            tokens = []
            async for token in client.generate_stream("test prompt"):
                tokens.append(token)

        assert tokens == []

    async def test_generate_stream_fallback_on_error(self, client: LLMClient) -> None:
        mock_client = MagicMock()
        mock_client.stream.side_effect = httpx.TimeoutException("timeout")

        with (
            patch.object(client, "_get_client", new_callable=AsyncMock, return_value=mock_client),
            patch.object(client, "generate", new_callable=AsyncMock, return_value="fallback answer"),
        ):
            tokens = []
            async for token in client.generate_stream("test prompt"):
                tokens.append(token)

        assert tokens == ["fallback answer"]

    async def test_generate_stream_skips_malformed_json(self, client: LLMClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        sse_lines = [
            "data: not json",
            'data: {"choices":[{"delta":{"content":"OK"}}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = MagicMock()
        mock_client.stream.return_value = _MockStreamContext(mock_response)

        with patch.object(client, "_get_client", new_callable=AsyncMock, return_value=mock_client):
            tokens = []
            async for token in client.generate_stream("test prompt"):
                tokens.append(token)

        assert tokens == ["OK"]


class TestSSEHelper:
    def test_sse_format(self) -> None:
        from data_engineering_copilot.services.async_rag import _sse

        result = _sse({"type": "token", "content": "hello"})
        assert result == 'data: {"type": "token", "content": "hello"}\n\n'

    def test_sse_empty_dict(self) -> None:
        from data_engineering_copilot.services.async_rag import _sse

        result = _sse({})
        assert result == "data: {}\n\n"
