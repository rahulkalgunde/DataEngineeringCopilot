"""ProviderFallbackChain integration with the availability-aware selector.

Verifies the selector-driven path: health-scored failover, immediate re-select
without stalling, the shared wait-then-recover loop, degraded fallback after
the deadline, per-request exclusion of failed providers, and streaming.
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
from data_engineering_copilot.infrastructure.provider_selector import (
    InMemoryRouterState,
    ProviderRouterConfig,
    ProviderSelector,
)


class _ScriptedClient:
    """Calls succeed/fail in scripted order; each failure raises RuntimeError."""

    def __init__(self, model: str, outcomes: list[bool], answer: str = "ok") -> None:
        self.model = model
        self._outcomes = outcomes
        self._answer = answer
        self.call_count = 0

    async def call(self, request):
        ok = self._outcomes[min(self.call_count, len(self._outcomes) - 1)]
        self.call_count += 1
        if not ok:
            raise RuntimeError("boom")
        return self._answer

    async def close(self) -> None: ...

    @property
    def last_usage(self):
        return None


class _StreamingClient:
    def __init__(self, model: str, tokens: list[str], fail_after: int | None = None) -> None:
        self.model = model
        self._tokens = tokens
        self._fail_after = fail_after

    async def call(self, request: str) -> str:
        return "".join(self._tokens)

    async def generate_stream(self, prompt: str, temperature: float | None = None) -> AsyncIterator[str]:
        if self._fail_after == 0:
            raise RuntimeError("boom before first token")
        for token in self._tokens:
            yield token

    async def close(self) -> None: ...

    @property
    def last_usage(self):
        return None


def _provider(name: str, client) -> ProviderConfig:
    return ProviderConfig(name=name, client=client)


def _build(
    providers: list[ProviderConfig],
    degraded: ProviderConfig | None = None,
    deadline: float = 45.0,
):
    health = ProviderHealthRegistry()
    for p in providers:
        health.register_provider(p.name, [p.client.model])
    if degraded is not None:
        health.register_provider(degraded.name, [degraded.client.model])
    state = InMemoryRouterState()
    selector = ProviderSelector(
        providers,
        health,
        state,
        ProviderRouterConfig(purpose="test", best_cache_ttl_seconds=0.0),
    )
    chain = ProviderFallbackChain(
        config=FallbackChainConfig(
            providers=providers,
            degraded_fallback=degraded,
            router_deadline_seconds=deadline,
        ),
        health=health,
        router=selector,
    )
    return chain, selector, state


@pytest.mark.asyncio
async def test_execute_fails_over_to_next_provider():
    failing_client = _ScriptedClient("f", [False])
    good_client = _ScriptedClient("g", [True])
    failing = _provider("failing", failing_client)
    good = _provider("good", good_client)
    chain, _, _ = _build([failing, good])

    result = await chain.execute("q")

    assert result == "ok"
    assert failing_client.call_count == 1
    assert good_client.call_count == 1


@pytest.mark.asyncio
async def test_execute_does_not_re_enter_failed_provider_same_request():
    """A provider that already failed this request must not be re-picked, even
    if it would be available again before the next provider wakes."""
    flaky_client = _ScriptedClient("f", [False, True])
    good_client = _ScriptedClient("g", [True])
    flaky = _provider("flaky", flaky_client)
    good = _provider("good", good_client)
    chain, _, _ = _build([flaky, good])

    result = await chain.execute("q")

    assert result == "ok"
    assert flaky_client.call_count == 1
    assert good_client.call_count == 1


@pytest.mark.asyncio
async def test_execute_waits_then_succeeds_after_cooldown_recovers():
    a_client = _ScriptedClient("a", [True])
    b_client = _ScriptedClient("b", [True])
    a = _provider("a", a_client)
    b = _provider("b", b_client)
    chain, selector, state = _build([a, b], deadline=3.0)

    await state.set_cooldown("a", 0.15)
    await state.set_cooldown("b", 0.15)

    result = await chain.execute("q")

    assert result == "ok"
    assert a_client.call_count + b_client.call_count == 1


@pytest.mark.asyncio
async def test_execute_uses_degraded_fallback_when_all_down_past_deadline():
    a_client = _ScriptedClient("a", [True])
    b_client = _ScriptedClient("b", [True])
    a = _provider("a", a_client)
    b = _provider("b", b_client)
    degraded = _provider("ollama", _ScriptedClient("o", [True], answer="degraded"))
    chain, _, state = _build([a, b], degraded=degraded, deadline=0.0)

    await state.set_cooldown("a", 60.0)
    await state.set_cooldown("b", 60.0)

    result = await chain.execute("q")

    assert result == "degraded"
    assert a_client.call_count == 0
    assert b_client.call_count == 0


@pytest.mark.asyncio
async def test_execute_failure_records_cross_process_cooldown():
    """A provider failure must be recorded in the shared state, not just the
    per-process registry."""
    failing = _provider("failing", _ScriptedClient("f", [False]))
    good = _provider("good", _ScriptedClient("g", [True]))
    chain, _, state = _build([failing, good])

    await chain.execute("q")

    until = await state.get_cooldown_until("failing")
    assert until > 0.0
    # 'good' succeeded → its shared cooldown is cleared.
    assert await state.get_cooldown_until("good") == 0.0


@pytest.mark.asyncio
async def test_generate_stream_fails_over_before_first_token():
    failing = _provider("failing", _StreamingClient("f", [], fail_after=0))
    good = _provider("good", _StreamingClient("g", ["X", "Y"]))
    chain, _, _ = _build([failing, good])

    tokens = [t async for t in chain.generate_stream("q")]

    assert tokens == ["X", "Y"]


@pytest.mark.asyncio
async def test_generate_stream_uses_degraded_fallback_when_all_main_fail():
    failing = _provider("failing", _StreamingClient("f", [], fail_after=0))
    degraded = _provider("ollama", _StreamingClient("o", ["Z"]))
    chain, _, _ = _build([failing], degraded=degraded, deadline=0.0)

    tokens = [t async for t in chain.generate_stream("q")]

    assert tokens == ["Z"]
