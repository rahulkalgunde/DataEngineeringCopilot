"""Tests for CooldownAwareEmbeddingRouter.

Verifies that the router waits for external provider cooldown before degrading
to local, and that it falls back to local only after the wait budget is
exhausted or no external provider is configured.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from data_engineering_copilot.infrastructure.cooldown_aware_router import CooldownAwareEmbeddingRouter
from data_engineering_copilot.infrastructure.llm_client import LLMClientError
from data_engineering_copilot.infrastructure.provider_fallback import (
    FallbackChainConfig,
    ProviderConfig,
    ProviderFallbackChain,
)
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry


class _StubClient:
    def __init__(self, model: str, fail: bool = False) -> None:
        self.model = model
        self.fail = fail
        self.calls: list[Any] = []

    async def call(self, request: Any) -> list[list[float]]:
        self.calls.append(request)
        if self.fail:
            raise LLMClientError("stub provider failure")
        texts = getattr(request, "texts", [])
        return [[float(len(t))] for t in texts]

    async def close(self) -> None: ...

    @property
    def last_usage(self):
        return None


def _router(
    providers: list[ProviderConfig],
    degraded: ProviderConfig | None = None,
    max_cooldown_wait_s: int = 60,
) -> CooldownAwareEmbeddingRouter:
    health = ProviderHealthRegistry()
    for p in providers:
        health.register_provider(p.name, [p.client.model])
    if degraded is not None:
        health.register_provider(degraded.name, [degraded.client.model])
    chain = ProviderFallbackChain(
        config=FallbackChainConfig(providers=providers, degraded_fallback=degraded),
        health=health,
    )
    return CooldownAwareEmbeddingRouter(
        chain=chain,
        health=health,
        max_cooldown_wait_s=max_cooldown_wait_s,
    )


def pytest_asyncio_run(coro):
    return __import__("asyncio").run(coro)


def test_external_available_delegates_immediately() -> None:
    client = _StubClient(model="ext-model")
    provider = ProviderConfig(name="openrouter", client=client)
    router = _router([provider])

    vectors = pytest_asyncio_run(router.embed_texts(["hello", "world"]))

    assert vectors == [[5.0], [5.0]]
    assert len(client.calls) == 1


def test_all_external_failed_falls_back_to_local() -> None:
    ext = _StubClient(model="ext-model", fail=True)
    local = _StubClient(model="local-model")
    provider = ProviderConfig(name="openrouter", client=ext)
    degraded = ProviderConfig(name="local-hf", client=local)
    router = _router([provider], degraded=degraded, max_cooldown_wait_s=1)

    vectors = pytest_asyncio_run(router.embed_texts(["abc"]))

    assert vectors == [[3.0]]
    assert len(ext.calls) == 1
    assert len(local.calls) == 1


def test_waits_for_shortest_cooldown_before_call() -> None:
    ext = _StubClient(model="ext-model", fail=True)
    local = _StubClient(model="local-model")
    provider = ProviderConfig(name="openrouter", client=ext)
    degraded = ProviderConfig(name="local-hf", client=local)
    router = _router([provider], degraded=degraded, max_cooldown_wait_s=10)

    health = router._health
    health.track_failure(
        "openrouter",
        "ext-model",
        category=__import__(
            "data_engineering_copilot.domain.exceptions",
            fromlist=["ProviderErrorCategory"],
        ).ProviderErrorCategory.RATE_LIMITED,
        retry_after=2.0,
    )

    start = time.monotonic()
    vectors = pytest_asyncio_run(router.embed_texts(["abc"]))
    elapsed = time.monotonic() - start

    assert vectors == [[3.0]]
    assert elapsed >= 2.0
    assert len(ext.calls) == 1
    assert len(local.calls) == 1


def test_wait_capped_by_max_cooldown_wait() -> None:
    ext = _StubClient(model="ext-model", fail=True)
    local = _StubClient(model="local-model")
    provider = ProviderConfig(name="openrouter", client=ext)
    degraded = ProviderConfig(name="local-hf", client=local)
    router = _router([provider], degraded=degraded, max_cooldown_wait_s=1)

    health = router._health
    health.track_failure(
        "openrouter",
        "ext-model",
        category=__import__(
            "data_engineering_copilot.domain.exceptions",
            fromlist=["ProviderErrorCategory"],
        ).ProviderErrorCategory.RATE_LIMITED,
        retry_after=10.0,
    )

    start = time.monotonic()
    vectors = pytest_asyncio_run(router.embed_texts(["abc"]))
    elapsed = time.monotonic() - start

    assert vectors == [[3.0]]
    assert 1.0 <= elapsed < 3.0
    assert len(ext.calls) == 0
    assert len(local.calls) == 1


def test_no_degraded_fallback_raises_after_wait() -> None:
    ext = _StubClient(model="ext-model", fail=True)
    provider = ProviderConfig(name="openrouter", client=ext)
    router = _router([provider], degraded=None, max_cooldown_wait_s=1)

    health = router._health
    health.track_failure(
        "openrouter",
        "ext-model",
        category=__import__(
            "data_engineering_copilot.domain.exceptions",
            fromlist=["ProviderErrorCategory"],
        ).ProviderErrorCategory.RATE_LIMITED,
        retry_after=2.0,
    )

    with pytest.raises(LLMClientError):
        pytest_asyncio_run(router.embed_texts(["abc"]))
