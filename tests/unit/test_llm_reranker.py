"""Behavioral tests for the LLMReranker facade (cloud chain + local fallback).

Exercises the real ``LLMReranker`` with a real ``ProviderFallbackChain`` built
from fake provider clients, plus a fake local cross-encoder, so the cloud ->
local degraded fallback path is exercised with real objects (type-only commit
rule).
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import (
    DocumentChunk,
    RerankRequest,
    RerankResult,
    RetrievedChunk,
)
from data_engineering_copilot.infrastructure.provider_fallback import (
    FallbackChainConfig,
    ProviderConfig,
    ProviderFallbackChain,
)
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.services.llm_reranker import LLMReranker


def _chunk(text: str, idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=f"chunk_{idx}",
            source_name="test_source",
            title=f"Chunk {idx}",
            url=f"http://example.com/{idx}",
            text=text,
            content_hash=f"hash_{idx}",
        ),
        distance=0.5,
        confidence=0.5,
    )


def _chunks(n: int) -> list[RetrievedChunk]:
    return [_chunk(f"document {i}", i) for i in range(n)]


class _FakeCloudClient:
    """ProviderClient double returning scripted rankings."""

    def __init__(self, rankings: tuple[tuple[int, float], ...] | None = None):
        self._rankings = rankings
        self.calls: list[RerankRequest] = []

    @property
    def model(self) -> str:
        return "fake-cloud"

    @property
    def last_usage(self):
        return None

    async def call(self, request: RerankRequest) -> RerankResult:
        self.calls.append(request)
        if self._rankings is None:
            raise RuntimeError("fake cloud provider down")
        return RerankResult(rankings=self._rankings)

    async def close(self) -> None:
        return None


class _FakeLocalReranker:
    """Local cross-encoder double returning scripted scores."""

    def __init__(self, scores: list[float] | None = None):
        self._scores = scores
        self.calls = 0
        self._closed = False

    @property
    def model_name(self) -> str:
        return "local-test"

    def is_available(self) -> bool:
        return self._scores is not None

    async def initialize(self) -> None:
        self.calls += 1

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.calls += 1
        if self._scores is None:
            return chunks[:top_k]
        scores = self._scores
        ordered = sorted(
            enumerate(chunks),
            key=lambda pair: scores[pair[0]],
            reverse=True,
        )
        return [pair[1] for pair in ordered[:top_k]]

    async def score_documents(self, query: str, documents: list[str]) -> list[float]:
        self.calls += 1
        if self._scores is None:
            return [0.0] * len(documents)
        return list(self._scores)

    async def close(self) -> None:
        self._closed = True

    def diversify_by_lexical_content(self, chunks, top_k: int = 5):
        return chunks[:top_k]


def _build_chain(cloud, local: _FakeLocalReranker) -> ProviderFallbackChain:
    """Build a real fallback chain: cloud providers + the local cross-encoder
    wrapped in the production ``LocalRerankerClient`` degraded fallback."""
    from data_engineering_copilot.infrastructure.rerank_clients import LocalRerankerClient

    config = FallbackChainConfig(
        providers=[ProviderConfig(name="cloud", client=cloud)],
        degraded_fallback=ProviderConfig(name="local-crossencoder", client=LocalRerankerClient(local)),
        max_degraded_consecutive_failures=3,
    )
    return ProviderFallbackChain(config, ProviderHealthRegistry())


class TestLLMReranker:
    @pytest.mark.asyncio
    async def test_cloud_success_reranks_and_reorders(self):
        cloud = _FakeCloudClient(rankings=((2, 0.95), (0, 0.8), (1, 0.2)))
        local = _FakeLocalReranker(scores=[0.1, 0.1, 0.9])
        reranker = LLMReranker(chain=_build_chain(cloud, local), local=local)

        assert reranker.is_available() is True
        result = await reranker.rerank("query", _chunks(3), top_k=3)

        assert len(result) == 3
        assert result[0].chunk.text == "document 2"
        assert result[0].confidence == pytest.approx(0.95)
        assert result[0].distance == pytest.approx(0.05)
        assert cloud.calls[0].query == "query"
        assert cloud.calls[0].documents == ["document 0", "document 1", "document 2"]
        assert local.calls == 0  # cloud served; local untouched

    @pytest.mark.asyncio
    async def test_cloud_failure_uses_local_degraded(self):
        cloud = _FakeCloudClient(rankings=None)  # provider down
        local = _FakeLocalReranker(scores=[0.1, 0.9, 0.5])
        reranker = LLMReranker(chain=_build_chain(cloud, local), local=local)

        result = await reranker.rerank("query", _chunks(3), top_k=3)

        assert local.calls == 1
        assert result[0].chunk.text == "document 1"  # local scores 0.9

    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_chunks_unchanged(self):
        cloud = _FakeCloudClient(rankings=None)  # down
        local = _FakeLocalReranker(scores=None)  # unavailable -> zero scores
        reranker = LLMReranker(chain=_build_chain(cloud, local), local=local)

        result = await reranker.rerank("query", _chunks(3), top_k=3)

        # All-zero scores keep the original order (confidence 0).
        assert [c.chunk.text for c in result] == ["document 0", "document 1", "document 2"]
        assert all(c.confidence == 0.0 for c in result)

    @pytest.mark.asyncio
    async def test_chain_total_failure_returns_chunks_unchanged(self):
        """Cloud fails and the local degraded fallback also fails -> the facade
        catches LLMClientError and returns chunks unchanged."""
        cloud = _FakeCloudClient(rankings=None)

        class _FailingLocal(_FakeLocalReranker):
            async def score_documents(self, query, documents):
                raise RuntimeError("local model crashed")

        local = _FailingLocal(scores=[1.0])
        reranker = LLMReranker(chain=_build_chain(cloud, local), local=local)

        result = await reranker.rerank("query", _chunks(3), top_k=3)

        assert [c.chunk.text for c in result] == ["document 0", "document 1", "document 2"]

    @pytest.mark.asyncio
    async def test_no_chain_delegates_to_local(self):
        local = _FakeLocalReranker(scores=[0.1, 0.9, 0.5])
        reranker = LLMReranker(chain=None, local=local)

        assert reranker.is_available() is True
        result = await reranker.rerank("query", _chunks(3), top_k=3)
        assert local.calls == 1
        assert result[0].chunk.text == "document 1"

    @pytest.mark.asyncio
    async def test_no_chain_and_no_local_returns_trimmed(self):
        reranker = LLMReranker(chain=None, local=None)
        assert reranker.is_available() is False
        result = await reranker.rerank("query", _chunks(5), top_k=2)
        assert [c.chunk.text for c in result] == ["document 0", "document 1"]

    @pytest.mark.asyncio
    async def test_initialize_noop_with_cloud_chain(self):
        cloud = _FakeCloudClient(rankings=((0, 1.0),))
        local = _FakeLocalReranker(scores=[1.0])
        reranker = LLMReranker(chain=_build_chain(cloud, local), local=local)
        await reranker.initialize()
        assert local.calls == 0  # cloud-first: local loads lazily

    @pytest.mark.asyncio
    async def test_initialize_loads_local_without_chain(self):
        local = _FakeLocalReranker(scores=[1.0])
        reranker = LLMReranker(chain=None, local=local)
        await reranker.initialize()
        assert local.calls == 1

    @pytest.mark.asyncio
    async def test_close_closes_chain_clients(self):
        cloud = _FakeCloudClient(rankings=((0, 1.0),))
        local = _FakeLocalReranker(scores=[1.0])
        reranker = LLMReranker(chain=_build_chain(cloud, local), local=local)
        await reranker.close()

    @pytest.mark.asyncio
    async def test_diversify_delegates_to_local(self):
        local = _FakeLocalReranker(scores=[1.0])
        reranker = LLMReranker(chain=None, local=local)
        chunks = _chunks(3)
        assert reranker.diversify_by_lexical_content(chunks, top_k=2) == chunks[:2]

    @pytest.mark.asyncio
    async def test_rerank_empty_chunks(self):
        reranker = LLMReranker(chain=None, local=None)
        assert await reranker.rerank("query", [], top_k=3) == []


class TestLLMRerankerWithRealChain:
    @pytest.mark.asyncio
    async def test_chain_uses_provider_calls_in_order(self):
        """End-to-end: the facade drives a real fallback chain; the first
        available provider wins and the local is skipped."""
        cloud = _FakeCloudClient(rankings=((1, 0.7), (0, 0.3)))
        local = _FakeLocalReranker(scores=[0.9, 0.1])
        chain = _build_chain(cloud, local)
        reranker = LLMReranker(chain=chain, local=local)

        result = await reranker.rerank("q", _chunks(2), top_k=2)

        assert cloud.calls
        assert local.calls == 0
        assert result[0].chunk.text == "document 1"

    @pytest.mark.asyncio
    async def test_rate_limit_failure_fails_over_to_local(self):
        """A 429-class failure on the cloud provider must fail over to the
        local degraded fallback (Retry-After honored by the chain cooldown)."""
        import httpx

        class _RateLimitedClient:
            @property
            def model(self) -> str:
                return "rate-limited"

            @property
            def last_usage(self):
                return None

            async def call(self, request: RerankRequest) -> RerankResult:
                raise httpx.HTTPStatusError(
                    "429",
                    request=httpx.Request("POST", "https://example.com"),
                    response=httpx.Response(429, headers={"Retry-After": "30"}, json={}),
                )

            async def close(self) -> None:
                return None

        cloud = _RateLimitedClient()
        local = _FakeLocalReranker(scores=[0.9, 0.1])
        chain = _build_chain(cloud, local)
        reranker = LLMReranker(chain=chain, local=local)

        result = await reranker.rerank("q", _chunks(2), top_k=2)

        assert local.calls == 1
        assert result[0].chunk.text == "document 0"
