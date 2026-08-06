"""Tests for AsyncOllamaEmbeddings — async httpx-based Ollama embedding provider."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings


@pytest.fixture
def async_embeddings():
    return AsyncOllamaEmbeddings(model_name="nomic-embed-text")


def test_init(async_embeddings):
    assert async_embeddings.model_name == "nomic-embed-text"
    assert async_embeddings.ollama_base_url
    assert async_embeddings._batch_size == 128
    assert async_embeddings._request_semaphore._value == 1


@pytest.mark.asyncio
async def test_embed_single_text(async_embeddings):
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
        )
        result = await async_embeddings._aollama_embed(["test text"])
        assert len(result) == 1
        assert result[0] == [0.1] * 768


@pytest.mark.asyncio
async def test_embed_multiple_texts(async_embeddings):
    embedding_vectors = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": embedding_vectors})
        )
        result = await async_embeddings._aollama_embed(["text 1", "text 2", "text 3"])
        assert len(result) == 3
        assert result == embedding_vectors


@pytest.mark.asyncio
async def test_embed_uses_correct_endpoint(async_embeddings):
    with respx.mock:
        route = respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
        )
        await async_embeddings._aollama_embed(["test"])
        assert route.called


@pytest.mark.asyncio
async def test_embed_request_payload(async_embeddings):
    with respx.mock:
        route = respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
        )
        await async_embeddings._aollama_embed(["test text"])
        request = route.calls[0].request
        import json

        body = json.loads(request.content)
        assert body["model"] == "nomic-embed-text"
        assert body["input"] == ["test text"]
        assert body["keep_alive"] == "10m"


@pytest.mark.asyncio
async def test_embed_batch_concurrency_is_configurable():
    embedder = AsyncOllamaEmbeddings(model_name="nomic-embed-text", max_concurrency=2)
    assert embedder._request_semaphore._value == 2


@pytest.mark.asyncio
async def test_embed_missing_embeddings_key(async_embeddings):
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * 768})
        )
        with pytest.raises(RuntimeError, match="missing 'embeddings' key"):
            await async_embeddings._aollama_embed(["test"])


@pytest.mark.asyncio
async def test_embed_count_mismatch(async_embeddings):
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
        )
        with pytest.raises(RuntimeError, match="returned 1 embeddings for 3 input texts"):
            await async_embeddings._aollama_embed(["text1", "text2", "text3"])


@pytest.mark.asyncio
async def test_embed_wrong_dimension(async_embeddings):
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1] * 512]})
        )
        with pytest.raises(RuntimeError, match="dimension 512"):
            await async_embeddings._aollama_embed(["test"])


@pytest.mark.asyncio
async def test_embed_empty_embedding(async_embeddings):
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[]]})
        )
        with pytest.raises(RuntimeError, match="empty"):
            await async_embeddings._aollama_embed(["test"])


@pytest.mark.asyncio
async def test_embed_network_error(async_embeddings):
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(side_effect=Exception("Connection refused"))
        with pytest.raises(Exception, match="Connection refused"):
            await async_embeddings._aollama_embed(["test"])


@pytest.mark.asyncio
async def test_embed_texts_calls_aollama_embed(async_embeddings):
    embedding_vectors = [[0.1] * 768, [0.2] * 768]
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": embedding_vectors})
        )
        result = await async_embeddings.embed_texts(["text1", "text2"])
        assert len(result) == 2


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector(async_embeddings):
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
        )
        result = await async_embeddings.embed_query("test query")
        assert result == [0.1] * 768
        assert len(result) == 768


@pytest.mark.asyncio
async def test_embed_query_empty_result_raises(async_embeddings):
    with respx.mock:
        respx.post(f"{async_embeddings.ollama_base_url}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [None]})
        )
        with pytest.raises(RuntimeError, match="not a list|empty result"):
            await async_embeddings.embed_query("test query")


def test_slice_texts_into_batches(async_embeddings):
    texts = [f"text{i}" for i in range(100)]
    batches = async_embeddings._slice_texts_into_batches(texts, batch_size=32)
    assert len(batches) == 4
    assert len(batches[0]) == 32


def test_slice_texts_into_batches_single_batch(async_embeddings):
    texts = ["text1", "text2"]
    batches = async_embeddings._slice_texts_into_batches(texts, batch_size=32)
    assert len(batches) == 1
    assert batches[0] == texts


def test_slice_texts_into_batches_empty(async_embeddings):
    batches = async_embeddings._slice_texts_into_batches([], batch_size=32)
    assert batches == []


def test_slice_texts_invalid_batch_size(async_embeddings):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        async_embeddings._slice_texts_into_batches(["text"], batch_size=0)


def test_client_is_recreated_across_event_loops():
    """Regression: a cached httpx client bound to a closed event loop must not
    be reused. RAGAS evaluates metrics in worker threads, each bridge call
    running its own ``asyncio.run`` loop, so a provider reused across calls
    must recreate its client (the old one is bound to a dead loop).

    Uses the production ``OpenAICompatibleEmbeddings`` client (the adaptive
    multi-provider fallback chain used for NVIDIA/OpenRouter) rather than the
    local-Ollama embedder."""
    from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
        OpenAICompatibleEmbeddings,
    )

    embedder = OpenAICompatibleEmbeddings(
        api_key="test-key",
        model_name="nvidia/nemotron-3-embed-1b",
        base_url="http://localhost:1",
    )

    async def embed() -> list[float]:
        return await embedder.embed_query("first loop")

    with respx.mock:
        respx.post("http://localhost:1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 2048}]},
            )
        )

        asyncio.run(embed())
        first_client = embedder._client
        asyncio.run(embed())
        second_client = embedder._client

    # The loop-bound guard must have recreated the client for the fresh loop
    # (on the unfixed code the same client object is reused across loops).
    assert first_client is not second_client
