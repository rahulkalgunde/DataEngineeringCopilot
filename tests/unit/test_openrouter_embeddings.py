"""Tests for OpenRouterEmbeddings — async httpx-based OpenRouter embedding provider."""

from __future__ import annotations

import httpx
import pytest
import respx

from data_engineering_copilot.infrastructure.async_openrouter_embeddings import OpenRouterEmbeddings


@pytest.fixture
def embeddings():
    return OpenRouterEmbeddings(
        api_key="sk-or-v1-test-key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        embedding_dimension=2048,
        batch_size=4,
    )


def test_init(embeddings):
    assert embeddings.model_name == "nvidia/nemotron-3-embed-1b:free"
    assert embeddings._embedding_dimension == 2048
    assert embeddings._batch_size == 4


@pytest.mark.asyncio
async def test_embed_single_text(embeddings):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        result = await embeddings.embed_texts(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 2048


@pytest.mark.asyncio
async def test_embed_multiple_texts(embeddings):
    vectors = [[0.1] * 2048, [0.2] * 2048]
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
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
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        result = await embeddings.embed_query("query")
        assert len(result) == 2048


@pytest.mark.asyncio
async def test_embed_sends_correct_payload(embeddings):
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embeddings.embed_texts(["hello world"])
        import json

        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "nvidia/nemotron-3-embed-1b:free"
        assert body["input"] == ["hello world"]


@pytest.mark.asyncio
async def test_embed_sends_auth_header(embeddings):
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 2048, "index": 0}]},
            )
        )
        await embeddings.embed_texts(["test"])
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-or-v1-test-key"


@pytest.mark.asyncio
async def test_embed_wrong_dimension_raises(embeddings):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 768, "index": 0}]},
            )
        )
        from data_engineering_copilot.domain.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="dimension 768"):
            await embeddings.embed_texts(["test"])


@pytest.mark.asyncio
async def test_embed_empty_list(embeddings):
    result = await embeddings.embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_http_error(embeddings):
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )
        from data_engineering_copilot.domain.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="Failed to get embeddings"):
            await embeddings.embed_texts(["test"])


@pytest.mark.asyncio
async def test_embed_batching():
    embs = OpenRouterEmbeddings(
        api_key="key",
        model_name="nvidia/nemotron-3-embed-1b:free",
        embedding_dimension=2048,
        batch_size=2,
    )
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": [0.1] * 2048, "index": 0},
                        {"embedding": [0.2] * 2048, "index": 1},
                    ]
                },
            )
        )
        result = await embs.embed_texts(["a", "b", "c", "d"])
        assert len(result) == 4
        assert route.call_count == 2
