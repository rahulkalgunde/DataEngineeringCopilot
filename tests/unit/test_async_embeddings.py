"""Tests for AsyncOllamaEmbeddings — async httpx-based Ollama embedding provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings


@pytest.fixture
def async_embeddings():
    return AsyncOllamaEmbeddings(model_name="nomic-embed-text")


def test_init(async_embeddings):
    assert async_embeddings.model_name == "nomic-embed-text"
    assert async_embeddings.ollama_base_url


async def test_embed_single_text(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await async_embeddings._aollama_embed(["test text"])
        assert len(result) == 1
        assert result[0] == [0.1] * 768


async def test_embed_multiple_texts(async_embeddings):
    embedding_vectors = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": embedding_vectors}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await async_embeddings._aollama_embed(["text 1", "text 2", "text 3"])
        assert len(result) == 3
        assert result == embedding_vectors


async def test_embed_uses_correct_endpoint(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(
        async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response
    ) as mock_post:
        await async_embeddings._aollama_embed(["test"])
        call_args = mock_post.call_args
        assert "/api/embed" in str(call_args)


async def test_embed_request_payload(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(
        async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response
    ) as mock_post:
        await async_embeddings._aollama_embed(["test text"])
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "nomic-embed-text"
        assert call_kwargs["json"]["input"] == ["test text"]


async def test_embed_missing_embeddings_key(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embedding": [0.1] * 768}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response), pytest.raises(RuntimeError, match="missing 'embeddings' key"):
        await async_embeddings._aollama_embed(["test"])


async def test_embed_count_mismatch(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response), pytest.raises(RuntimeError, match="returned 1 embeddings for 3 input texts"):
        await async_embeddings._aollama_embed(["text1", "text2", "text3"])


async def test_embed_wrong_dimension(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1] * 512]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response), pytest.raises(RuntimeError, match="dimension 512"):
        await async_embeddings._aollama_embed(["test"])


async def test_embed_empty_embedding(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[]]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response), pytest.raises(RuntimeError, match="empty"):
        await async_embeddings._aollama_embed(["test"])


async def test_embed_network_error(async_embeddings):
    with (
        patch.object(
            async_embeddings._client, "post", new_callable=AsyncMock, side_effect=Exception("Connection refused")
        ),
        pytest.raises(RuntimeError, match="Failed to get embeddings from Ollama"),
    ):
        await async_embeddings._aollama_embed(["test"])


async def test_embed_texts_calls_aollama_embed(async_embeddings):
    embedding_vectors = [[0.1] * 768, [0.2] * 768]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": embedding_vectors}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await async_embeddings.embed_texts(["text1", "text2"])
        assert len(result) == 2


async def test_embed_query_returns_single_vector(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await async_embeddings.embed_query("test query")
        assert result == [0.1] * 768
        assert len(result) == 768


async def test_embed_query_empty_result_raises(async_embeddings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [None]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(async_embeddings._client, "post", new_callable=AsyncMock, return_value=mock_response), pytest.raises(RuntimeError, match="not a list|empty result"):
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
