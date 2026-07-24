"""Tests for OpenAIEmbeddings — async httpx-based OpenAI embedding provider."""

from __future__ import annotations

import httpx
import pytest
import respx

from data_engineering_copilot.infrastructure.async_openai_embeddings import OpenAIEmbeddings


@pytest.fixture
def embeddings():
    return OpenAIEmbeddings(
        api_key="sk-test-key",
        model_name="text-embedding-3-small",
        embedding_dimension=1536,
        batch_size=4,
    )


def test_init(embeddings):
    assert embeddings.model_name == "text-embedding-3-small"
    assert embeddings._embedding_dimension == 1536
    assert embeddings._batch_size == 4


@pytest.mark.asyncio
async def test_embed_single_text(embeddings):
    with respx.mock:
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1536, "index": 0}]},
            )
        )
        result = await embeddings.embed_texts(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 1536


@pytest.mark.asyncio
async def test_embed_multiple_texts(embeddings):
    vectors = [[0.1] * 1536, [0.2] * 1536]
    with respx.mock:
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": vectors[0], "index": 0},
                        {"embedding": vectors[1], "index": 1},
                    ]
                },
            )
        )
        result = await embeddings.embed_texts(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == vectors[0]


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector(embeddings):
    with respx.mock:
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1536, "index": 0}]},
            )
        )
        result = await embeddings.embed_query("query")
        assert len(result) == 1536


@pytest.mark.asyncio
async def test_embed_sends_correct_payload(embeddings):
    with respx.mock:
        route = respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1536, "index": 0}]},
            )
        )
        await embeddings.embed_texts(["hello world"])
        import json

        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["hello world"]


@pytest.mark.asyncio
async def test_embed_sends_auth_header(embeddings):
    with respx.mock:
        route = respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1536, "index": 0}]},
            )
        )
        await embeddings.embed_texts(["test"])
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test-key"


@pytest.mark.asyncio
async def test_embed_wrong_dimension_raises(embeddings):
    with respx.mock:
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 512, "index": 0}]},
            )
        )
        from data_engineering_copilot.domain.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="dimension 512"):
            await embeddings.embed_texts(["test"])


@pytest.mark.asyncio
async def test_embed_empty_list(embeddings):
    result = await embeddings.embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_http_error(embeddings):
    with respx.mock:
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )
        from data_engineering_copilot.domain.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="Failed to get embeddings"):
            await embeddings.embed_texts(["test"])


@pytest.mark.asyncio
async def test_embed_batching():
    embs = OpenAIEmbeddings(
        api_key="key",
        model_name="text-embedding-3-small",
        embedding_dimension=1536,
        batch_size=2,
    )
    with respx.mock:
        route = respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": [0.1] * 1536, "index": 0},
                        {"embedding": [0.2] * 1536, "index": 1},
                    ]
                },
            )
        )
        result = await embs.embed_texts(["a", "b", "c", "d"])
        assert len(result) == 4
        assert route.call_count == 2
