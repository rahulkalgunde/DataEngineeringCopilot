"""Tests for the availability-aware ProviderSelector.

Pins the O(1) cached-best hot path, health-scored ready-pool ordering, the
purpose-preference weight, cooldown skipping, and per-request exclusion.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
from data_engineering_copilot.infrastructure.provider_fallback import ProviderConfig
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.infrastructure.provider_selector import (
    InMemoryRouterState,
    ProviderRouterConfig,
    ProviderSelector,
)


class _FakeClient:
    def __init__(self, model: str, answer: str = "ok") -> None:
        self.model = model
        self._answer = answer

    async def call(self, request):
        return self._answer

    async def close(self) -> None: ...

    @property
    def last_usage(self):
        return None


def _provider(name: str) -> ProviderConfig:
    return ProviderConfig(name=name, client=_FakeClient(f"{name}-model"))


def _selector(provider_names: list[str], **cfg_kwargs) -> ProviderSelector:
    health = ProviderHealthRegistry()
    for name in provider_names:
        health.register_provider(name, [f"{name}-model"])
    config = ProviderRouterConfig(purpose="test", **cfg_kwargs)
    return ProviderSelector([_provider(n) for n in provider_names], health, InMemoryRouterState(), config)


def _set_success_rate(health: ProviderHealthRegistry, provider: str, rate: float) -> None:
    """Force a provider's success rate without touching availability/cooldowns."""
    mh = health.get_model_health(provider, f"{provider}-model")
    assert mh is not None
    failures = 1
    successes = 0
    while (successes / (successes + failures)) < rate and successes < 1000:
        successes += 1
    while (successes / (successes + failures)) > rate and failures < 1000:
        failures += 1
    mh.total_success = successes
    mh.total_failures = failures
    mh.total_latency = 0.0


@pytest.mark.asyncio
async def test_pick_returns_healthiest_available_provider():
    sel = _selector(["a", "b"])
    _set_success_rate(sel.health, "a", 0.9)
    _set_success_rate(sel.health, "b", 0.5)

    picked = await sel.pick()

    assert picked is not None
    assert picked.name == "a"


@pytest.mark.asyncio
async def test_pick_skips_provider_in_cooldown():
    sel = _selector(["a", "b"])
    sel.health.track_failure("a", "a-model", ProviderErrorCategory.RATE_LIMITED, retry_after=60.0)

    picked = await sel.pick()

    assert picked is not None
    assert picked.name == "b"


@pytest.mark.asyncio
async def test_pick_applies_preference_weight_to_pinned_provider():
    sel = _selector(["a", "b"], preference_provider="b", preference_weight=0.15)
    _set_success_rate(sel.health, "a", 0.5)
    _set_success_rate(sel.health, "b", 0.4)

    picked = await sel.pick()

    assert picked is not None
    assert picked.name == "b"


@pytest.mark.asyncio
async def test_pick_returns_none_when_all_providers_in_cooldown():
    sel = _selector(["a", "b"])
    sel.health.track_failure("a", "a-model", ProviderErrorCategory.RATE_LIMITED, retry_after=60.0)
    sel.health.track_failure("b", "b-model", ProviderErrorCategory.RATE_LIMITED, retry_after=60.0)

    assert await sel.pick() is None


@pytest.mark.asyncio
async def test_pick_uses_cached_best_on_hot_path():
    sel = _selector(["a", "b"], best_cache_ttl_seconds=60.0)
    _set_success_rate(sel.health, "a", 0.9)
    _set_success_rate(sel.health, "b", 0.5)

    first = await sel.pick()
    assert first is not None and first.name == "a"

    _set_success_rate(sel.health, "a", 0.1)
    _set_success_rate(sel.health, "b", 0.9)

    # Cache still valid → must NOT re-rank, keeps serving the cached best.
    second = await sel.pick()
    assert second is not None and second.name == "a"

    await sel.state.clear_cached_best(sel.config.purpose)
    third = await sel.pick()
    assert third is not None and third.name == "b"


@pytest.mark.asyncio
async def test_pick_rebuilds_when_cached_best_enters_cooldown():
    sel = _selector(["a", "b"], best_cache_ttl_seconds=60.0)

    first = await sel.pick()
    assert first is not None and first.name == "a"

    sel.health.track_failure("a", "a-model", ProviderErrorCategory.RATE_LIMITED, retry_after=60.0)

    second = await sel.pick()
    assert second is not None and second.name == "b"


@pytest.mark.asyncio
async def test_pick_excludes_request_failed_provider():
    sel = _selector(["a", "b"])

    picked = await sel.pick(exclude={"a"})

    assert picked is not None
    assert picked.name == "b"
