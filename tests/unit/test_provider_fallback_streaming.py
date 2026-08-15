"""Tests for real streaming through ProviderFallbackChain (Phase A).

Verifies that ``generate_stream`` walks providers like ``execute`` (gates,
degraded fallback) while emitting incremental tokens, and that a mid-stream
failure after tokens are emitted re-raises instead of silently truncating.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from data_engineering_copilot.infrastructure.provider_fallback import (
    FallbackChainConfig,
    ProviderConfig,
    ProviderFallbackChain,
)
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry


class _StreamingClient:
    """LLM-like client with a real ``generate_stream``."""

    def __init__(self, model: str, tokens: list[str], fail_after: int | None = None) -> None:
        self.model = model
        self._tokens = tokens
        self._fail_after = fail_after

    async def call(self, request: str) -> str:
        return "".join(self._tokens)

    async def generate_stream(self, prompt: str, temperature: float | None = None) -> AsyncIterator[str]:
        if self._fail_after == 0:
            raise RuntimeError("boom before first token")
        for i, token in enumerate(self._tokens):
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError("mid-stream boom")
            yield token

    async def close(self) -> None: ...

    @property
    def last_usage(self):
        return None


class _NonStreamingClient:
    """Client with no ``generate_stream`` — must fall back to ``call``."""

    def __init__(self, model: str, answer: str) -> None:
        self.model = model
        self._answer = answer

    async def call(self, request: str) -> str:
        return self._answer

    async def close(self) -> None: ...

    @property
    def last_usage(self):
        return None


def _chain(providers: list[ProviderConfig], degraded: ProviderConfig | None = None) -> ProviderFallbackChain:
    health = ProviderHealthRegistry()
    for p in providers:
        health.register_provider(p.name, [p.client.model])
    if degraded is not None:
        health.register_provider(degraded.name, [degraded.client.model])
    return ProviderFallbackChain(
        config=FallbackChainConfig(providers=providers, degraded_fallback=degraded),
        health=health,
    )


def _provider(name: str, client) -> ProviderConfig:
    return ProviderConfig(name=name, client=client)


async def _collect(stream) -> list[str]:
    return [t async for t in stream]


@pytest.mark.asyncio
async def test_generate_stream_emits_incremental_tokens():
    chain = _chain([_provider("primary", _StreamingClient("p", ["A", "B", "C"]))])
    assert await _collect(chain.generate_stream("q")) == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_generate_stream_falls_back_to_next_provider_when_primary_fails_before_tokens():
    failing = _provider("failing", _StreamingClient("f", [], fail_after=0))
    good = _provider("good", _StreamingClient("g", ["X", "Y"]))
    chain = _chain([failing, good])
    assert await _collect(chain.generate_stream("q")) == ["X", "Y"]


@pytest.mark.asyncio
async def test_generate_stream_uses_degraded_fallback_when_all_main_fail():
    failing = _provider("failing", _StreamingClient("f", [], fail_after=0))
    degraded = _provider("ollama", _StreamingClient("o", ["Z"]))
    chain = _chain([failing], degraded=degraded)
    assert await _collect(chain.generate_stream("q")) == ["Z"]


@pytest.mark.asyncio
async def test_generate_stream_re_raises_on_mid_stream_failure_after_tokens():
    """A provider that emitted >=1 token then fails must NOT fall through."""
    from data_engineering_copilot.domain.exceptions import ProviderError

    flaky = _provider("flaky", _StreamingClient("f", ["A", "B", "C"], fail_after=1))
    good = _provider("good", _StreamingClient("g", ["X", "Y"]))
    chain = _chain([flaky, good])

    collected: list[str] = []
    with pytest.raises(ProviderError):
        async for token in chain.generate_stream("q"):
            collected.append(token)
    # The good provider must never have been reached after partial output.
    assert collected == ["A"]


@pytest.mark.asyncio
async def test_generate_stream_falls_back_to_non_streaming_call():
    chain = _chain([_provider("plain", _NonStreamingClient("p", "whole answer"))])
    assert await _collect(chain.generate_stream("q")) == ["whole answer"]


@pytest.mark.asyncio
async def test_generate_stream_all_fail_raises():
    from data_engineering_copilot.infrastructure.llm_client import LLMClientError

    failing = _provider("failing", _StreamingClient("f", [], fail_after=0))
    chain = _chain([failing])
    with pytest.raises(LLMClientError):
        await _collect(chain.generate_stream("q"))


@pytest.mark.asyncio
async def test_generate_stream_skips_cooldown_provider():
    from data_engineering_copilot.domain.exceptions import ProviderErrorCategory

    health = ProviderHealthRegistry()
    failing_client = _StreamingClient("f", ["Z"])
    health.register_provider("failing", ["f"])
    health.track_failure("failing", "f", ProviderErrorCategory.RETRYABLE, retry_after=3600)

    good = _provider("good", _StreamingClient("g", ["OK"]))
    chain = ProviderFallbackChain(
        config=FallbackChainConfig(providers=[_provider("failing", failing_client), good]),
        health=health,
    )
    assert await _collect(chain.generate_stream("q")) == ["OK"]
