"""Tests for the availability-fraction wait policy and shared wait gate."""

from __future__ import annotations

import asyncio
import time

import pytest

from data_engineering_copilot.infrastructure.provider_fallback import ProviderConfig
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.infrastructure.provider_selector import (
    InMemoryRouterState,
    ProviderRouterConfig,
    ProviderSelector,
)


class _FakeClient:
    def __init__(self, model: str) -> None:
        self.model = model

    async def call(self, request):
        return "ok"

    async def close(self) -> None: ...

    @property
    def last_usage(self):
        return None


def _provider(name: str) -> ProviderConfig:
    return ProviderConfig(name=name, client=_FakeClient(f"{name}-model"))


def _selector(provider_names: list[str], **cfg_kwargs) -> tuple[ProviderSelector, InMemoryRouterState]:
    health = ProviderHealthRegistry()
    for name in provider_names:
        health.register_provider(name, [f"{name}-model"])
    state = InMemoryRouterState()
    config = ProviderRouterConfig(purpose="test", **cfg_kwargs)
    return ProviderSelector([_provider(n) for n in provider_names], health, state, config), state


@pytest.mark.asyncio
async def test_seconds_until_fraction_waits_for_threshold_provider():
    sel, state = _selector(["a", "b", "c", "d"])
    await state.set_cooldown("a", 2.0)
    await state.set_cooldown("b", 5.0)
    await state.set_cooldown("c", 10.0)

    # 'd' is already available (0.0). 50% of 4 → needs 2 available → d + a at 2s.
    assert await sel.seconds_until_fraction(0.5) == pytest.approx(2.0, abs=0.5)
    # 75% of 4 → needs 3 available → d + a + b at 5s.
    assert await sel.seconds_until_fraction(0.75) == pytest.approx(5.0, abs=0.5)


@pytest.mark.asyncio
async def test_seconds_until_fraction_zero_when_already_enough():
    sel, _ = _selector(["a", "b"])
    assert await sel.seconds_until_fraction(0.5) == 0.0


@pytest.mark.asyncio
async def test_seconds_until_fraction_ignores_excluded_providers():
    sel, state = _selector(["a", "b"])
    await state.set_cooldown("a", 30.0)
    await state.set_cooldown("b", 0.2)

    # 'a' is excluded (already failed this request) → only 'b' counts.
    assert await sel.seconds_until_fraction(0.5, exclude={"a"}) == pytest.approx(0.2, abs=0.1)


@pytest.mark.asyncio
async def test_wait_for_availability_recovers_short_cooldown():
    sel, state = _selector(["a", "b"], wait_max_seconds=5.0)
    await state.set_cooldown("a", 0.15)
    await state.set_cooldown("b", 0.15)

    deadline = time.monotonic() + 2.0
    recovered = await sel.wait_for_availability(deadline)

    assert recovered is True
    assert await sel.pick() is not None


@pytest.mark.asyncio
async def test_wait_for_availability_returns_false_when_deadline_expired():
    sel, state = _selector(["a", "b"])
    await state.set_cooldown("a", 60.0)
    await state.set_cooldown("b", 60.0)

    deadline = time.monotonic() - 1.0
    assert await sel.wait_for_availability(deadline) is False


@pytest.mark.asyncio
async def test_wait_for_availability_shared_gate_sleeps_once():
    """Concurrent waiters on the same all-down window share one timer."""
    sel, state = _selector(["a", "b"], wait_max_seconds=5.0)
    await state.set_cooldown("a", 0.25)
    await state.set_cooldown("b", 0.25)

    deadline = time.monotonic() + 3.0
    start = time.monotonic()
    results = await asyncio.gather(
        sel.wait_for_availability(deadline),
        sel.wait_for_availability(deadline),
    )
    elapsed = time.monotonic() - start

    assert results == [True, True]
    # Shared: ~0.3s. Two independent sleeps would be ~0.6s.
    assert elapsed < 0.45
