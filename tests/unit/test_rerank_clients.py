"""Hermetic tests for the rerank provider clients.

Uses ``respx`` to mock the provider HTTP endpoints. No network, no paid calls.
Verifies request payload shapes, score normalization (OpenRouter relevance,
NVIDIA sigmoid of logit, HF text-classification parsing), 429 -> Retry-After
handling, and the local cross-encoder adapter.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from tenacity.wait import wait_fixed

from data_engineering_copilot.domain.exceptions import RerankError
from data_engineering_copilot.domain.models import RerankRequest, RerankResult
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter
from data_engineering_copilot.infrastructure.rerank_clients import (
    HuggingFaceRerankClient,
    LocalRerankerClient,
    NvidiaRerankClient,
    OpenRouterRerankClient,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/rerank"
NVIDIA_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
HF_MODEL = "BAAI/bge-reranker-v2-m3"
HF_PIPELINE_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/text-classification"

RELEVANT = "A DataFrame is a distributed collection of data organized into named columns."
IRRELEVANT = "PyTorch is an open-source machine learning framework for research prototyping."
QUERY = "How do I create a Spark DataFrame?"


def _request(docs: list[str] | None = None) -> RerankRequest:
    return RerankRequest(
        query=QUERY,
        documents=[RELEVANT, IRRELEVANT] if docs is None else docs,
        top_n=2,
    )


def _sent_json(route: respx.Route) -> dict:
    return json.loads(route.calls[0].request.content.decode("utf-8"))


class TestOpenRouterRerankClient:
    @pytest.fixture
    def client(self):
        return OpenRouterRerankClient(api_key="or_test", retry_wait=wait_fixed(0))

    @pytest.mark.asyncio
    async def test_request_payload_and_score_passthrough(self, client):
        with respx.mock:
            route = respx.post(OPENROUTER_URL).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "results": [
                            {"index": 1, "relevance_score": 0.0005, "document": {"text": IRRELEVANT}},
                            {"index": 0, "relevance_score": 0.594, "document": {"text": RELEVANT}},
                        ]
                    },
                )
            )
            result = await client.call(_request())
            payload = _sent_json(route)
            assert payload["model"] == "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
            assert payload["query"] == QUERY
            assert payload["documents"] == [RELEVANT, IRRELEVANT]
            assert payload["top_n"] == 2
            # Sorted descending by score, index preserved.
            assert result.rankings == ((0, 0.594), (1, 0.0005))
            # OpenRouter provenance headers present (httpx lowercases header keys).
            headers = dict(route.calls[0].request.headers)
            assert headers.get("http-referer") == "http://localhost"
            assert headers.get("x-title") == "data-engineering-copilot"

    @pytest.mark.asyncio
    async def test_empty_documents(self, client):
        assert await client.call(_request([])) == RerankResult()

    @pytest.mark.asyncio
    async def test_missing_results_raises(self, client):
        with respx.mock:
            respx.post(OPENROUTER_URL).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(RerankError, match="results"):
                await client.call(_request())


class TestNvidiaRerankClient:
    @pytest.fixture
    def client(self):
        return NvidiaRerankClient(api_key="nv_test", retry_wait=wait_fixed(0))

    @pytest.mark.asyncio
    async def test_request_payload_and_sigmoid_normalization(self, client):
        with respx.mock:
            route = respx.post(NVIDIA_URL).mock(
                return_value=httpx.Response(
                    200,
                    json={"rankings": [{"index": 0, "logit": 3.32421875}, {"index": 1, "logit": -19.34375}]},
                )
            )
            result = await client.call(_request())
            payload = _sent_json(route)
            assert payload["model"] == "nv-rerank-qa-mistral-4b:1"
            assert payload["query"] == {"text": QUERY}
            assert payload["passages"] == [{"text": RELEVANT}, {"text": IRRELEVANT}]
            assert payload["truncate"] == "END"
            assert result.rankings[0][0] == 0
            assert result.rankings[0][1] == pytest.approx(0.965, abs=1e-3)
            assert result.rankings[1][1] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_empty_documents(self, client):
        assert await client.call(_request([])) == RerankResult()

    @pytest.mark.asyncio
    async def test_missing_rankings_raises(self, client):
        with respx.mock:
            respx.post(NVIDIA_URL).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(RerankError, match="rankings"):
                await client.call(_request())


class TestHuggingFaceRerankClient:
    @pytest.fixture
    def client(self):
        return HuggingFaceRerankClient(api_key="hf_test", retry_wait=wait_fixed(0))

    @pytest.mark.asyncio
    async def test_flat_serverless_shape_in_input_order(self, client):
        with respx.mock:
            route = respx.post(HF_PIPELINE_URL).mock(
                return_value=httpx.Response(
                    200,
                    json=[[{"label": "LABEL_0", "score": 0.1497}, {"label": "LABEL_0", "score": 0.00007}]],
                )
            )
            result = await client.call(_request())
            payload = _sent_json(route)
            assert payload["inputs"] == [f"{QUERY} {RELEVANT}", f"{QUERY} {IRRELEVANT}"]
            assert result.rankings == ((0, 0.1497), (1, 0.00007))

    @pytest.mark.asyncio
    async def test_standard_per_input_shape_takes_max_label(self, client):
        with respx.mock:
            respx.post(HF_PIPELINE_URL).mock(
                return_value=httpx.Response(
                    200,
                    json=[
                        [{"label": "LABEL_0", "score": 0.1}, {"label": "LABEL_1", "score": 0.9}],
                        [{"label": "LABEL_0", "score": 0.8}, {"label": "LABEL_1", "score": 0.2}],
                    ],
                )
            )
            result = await client.call(_request())
            assert result.rankings == ((0, 0.9), (1, 0.8))

    @pytest.mark.asyncio
    async def test_empty_documents(self, client):
        assert await client.call(_request([])) == RerankResult()

    @pytest.mark.asyncio
    async def test_shape_mismatch_raises(self, client):
        with respx.mock:
            respx.post(HF_PIPELINE_URL).mock(return_value=httpx.Response(200, json=[[{"label": "L0", "score": 0.5}]]))
            with pytest.raises(RerankError, match="shape mismatch"):
                await client.call(_request())


class TestRerankRateLimitHandling:
    @pytest.mark.asyncio
    async def test_429_raises_http_status_error_for_chain(self):
        """429 must surface as an httpx.HTTPStatusError so the fallback chain's
        categorizer maps it to RATE_LIMITED and honors Retry-After."""
        client = OpenRouterRerankClient(api_key="or_test", retry_wait=wait_fixed(0))
        with respx.mock:
            route = respx.post(OPENROUTER_URL).mock(
                side_effect=lambda request: httpx.Response(429, headers={"Retry-After": "30"}, json={})
            )
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.call(_request())
            assert exc_info.value.response.status_code == 429
            assert len(route.calls) == 4  # tenacity stop_after_attempt(4)

    @pytest.mark.asyncio
    async def test_429_applies_retry_after_backoff(self):
        """The client must wait the provider's Retry-After on 429 before the
        tenacity retries resume."""

        limiter = SlidingWindowRateLimiter(rpm_limit=10, rpd_limit=100)
        client = OpenRouterRerankClient(api_key="or_test", rate_limiter=limiter, retry_wait=wait_fixed(0))
        with respx.mock:
            respx.post(OPENROUTER_URL).mock(
                side_effect=lambda request: httpx.Response(429, headers={"Retry-After": "2"}, json={})
            )
            with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock, pytest.raises(httpx.HTTPStatusError):
                await client.call(_request())
        # handle_429 slept Retry-After seconds before the retries resumed.
        assert any(call.args == (2,) for call in sleep_mock.call_args_list)


class TestLocalRerankerClient:
    @pytest.mark.asyncio
    async def test_scores_documents_and_returns_rankings(self):
        class _FakeReranker:
            model_name = "local-test"

            async def score_documents(self, query, documents):
                # Higher score for the first document.
                return [0.9, 0.1]

        client = LocalRerankerClient(_FakeReranker())
        result = await client.call(_request())
        assert result.rankings == ((0, 0.9), (1, 0.1))
        assert client.model == "local-test"

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        client = LocalRerankerClient(_FakeReranker())
        assert await client.call(_request([])) == RerankResult()

    @pytest.mark.asyncio
    async def test_close_forwards_to_reranker(self):
        closed = False

        class _FakeReranker:
            async def close(self):
                nonlocal closed
                closed = True

        client = LocalRerankerClient(_FakeReranker())
        await client.close()
        assert closed is True


class _FakeReranker:
    """Local cross-encoder double used by the local adapter tests."""

    model_name = "local-test"

    async def score_documents(self, query, documents):
        return [0.9, 0.1]
