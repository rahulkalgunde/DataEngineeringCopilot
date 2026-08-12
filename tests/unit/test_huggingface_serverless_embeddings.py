"""Hermetic tests for the Hugging Face serverless embedding provider.

Uses ``respx`` to mock the native ``feature-extraction`` pipeline route. No
network, no paid calls. Verifies prefix mapping (passage/query), batch
handling, dimension validation, budget rejection, and the chain ``call`` path.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from data_engineering_copilot.domain.exceptions import EmbeddingError
from data_engineering_copilot.domain.models import EmbeddingRequest
from data_engineering_copilot.infrastructure.huggingface_serverless_embeddings import (
    DEFAULT_BASE_URL,
    HuggingFaceServerlessEmbeddings,
)

MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
PIPELINE_URL = f"{DEFAULT_BASE_URL}/models/{MODEL}/pipeline/feature-extraction"


def _sent_inputs(route: respx.Route) -> list[str]:
    payload = json.loads(route.calls[0].request.content.decode("utf-8"))
    return payload["inputs"]


@pytest.fixture
def embeddings():
    return HuggingFaceServerlessEmbeddings(
        api_key="hf_test",
        model_name=MODEL,
        embedding_dimension=4,
        batch_size=2,
    )


def test_init(embeddings):
    assert embeddings.model_name == MODEL
    assert embeddings._embedding_dimension == 4
    assert embeddings._batch_size == 2
    assert embeddings.model == MODEL


@pytest.mark.asyncio
async def test_embed_texts_uses_passage_prefix_and_returns_vectors(embeddings):
    with respx.mock:
        route = respx.post(PIPELINE_URL).mock(
            return_value=httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        )
        result = await embeddings.embed_texts(["chunk a", "chunk b"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert _sent_inputs(route) == ["passage: chunk a", "passage: chunk b"]


@pytest.mark.asyncio
async def test_embed_query_uses_query_prefix(embeddings):
    with respx.mock:
        route = respx.post(PIPELINE_URL).mock(return_value=httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4]]))
        result = await embeddings.embed_query("what is spark?")
        assert result == [0.1, 0.2, 0.3, 0.4]
        assert _sent_inputs(route) == ["query: what is spark?"]


@pytest.mark.asyncio
async def test_call_forwards_embedding_request(embeddings):
    with respx.mock:
        route = respx.post(PIPELINE_URL).mock(return_value=httpx.Response(200, json=[[1.0, 1.0, 1.0, 1.0]]))
        await embeddings.call(EmbeddingRequest(input_type="query", texts=["q"]))
        assert _sent_inputs(route) == ["query: q"]


@pytest.mark.asyncio
async def test_call_plain_list_defaults_to_passage(embeddings):
    with respx.mock:
        route = respx.post(PIPELINE_URL).mock(return_value=httpx.Response(200, json=[[1.0, 1.0, 1.0, 1.0]]))
        await embeddings.call(["legacy chunk"])
        assert _sent_inputs(route) == ["passage: legacy chunk"]


@pytest.mark.asyncio
async def test_batches_large_inputs(embeddings):
    with respx.mock:
        route = respx.post(PIPELINE_URL).mock(
            side_effect=[
                httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
                httpx.Response(200, json=[[0.9, 0.1, 0.2, 0.3]]),
            ]
        )
        result = await embeddings.embed_texts(["a", "b", "c"])
        assert len(result) == 3
        assert len(route.calls) == 2


@pytest.mark.asyncio
async def test_dimension_mismatch_raises(embeddings):
    with respx.mock:
        respx.post(PIPELINE_URL).mock(return_value=httpx.Response(200, json=[[0.1, 0.2, 0.3]]))
        with pytest.raises(EmbeddingError, match="dimension 3, expected 4"):
            await embeddings.embed_texts(["chunk"])


@pytest.mark.asyncio
async def test_wrong_count_raises(embeddings):
    with respx.mock:
        respx.post(PIPELINE_URL).mock(return_value=httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4]]))
        with pytest.raises(EmbeddingError, match="1 embeddings for 2 input texts"):
            await embeddings.embed_texts(["a", "b"])


@pytest.mark.asyncio
async def test_over_budget_rejected_before_request(embeddings):
    embeddings._max_tokens_per_input = 2
    with respx.mock:
        route = respx.post(PIPELINE_URL).mock(return_value=httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4]]))
        with pytest.raises(EmbeddingError, match="exceeds budget"):
            await embeddings.embed_texts(["this text has many tokens for a tiny budget"])
        assert not route.calls


@pytest.mark.asyncio
async def test_embed_texts_empty_returns_empty(embeddings):
    assert await embeddings.embed_texts([]) == []


def test_last_usage_shape(embeddings):
    assert embeddings.last_usage.prompt_tokens == 0
