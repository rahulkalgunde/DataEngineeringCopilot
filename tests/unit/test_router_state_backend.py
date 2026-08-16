"""Behavioral contract for RouterStateBackend implementations.

Both ``InMemoryRouterState`` (default) and ``RedisRouterState`` (shared across
processes, tested against the in-memory Redis double) must behave identically:
cooldown round-trips, best-cache round-trips, and best-cache TTL expiry.
"""

from __future__ import annotations

import time

import pytest

from data_engineering_copilot.infrastructure.provider_selector import InMemoryRouterState, RedisRouterState
from tests.doubles.redis import _StubRedis

_BACKEND_FACTORIES = [
    (InMemoryRouterState, "in-memory"),
    (lambda: RedisRouterState(_StubRedis()), "redis"),
]


@pytest.mark.parametrize(
    ("state_factory", "backend_id"),
    _BACKEND_FACTORIES,
    ids=[b[1] for b in _BACKEND_FACTORIES],
)
@pytest.mark.asyncio
async def test_cooldown_round_trip(state_factory, backend_id):
    state = state_factory()

    assert await state.get_cooldown_until("openrouter") == 0.0

    await state.set_cooldown("openrouter", 60.0)
    until = await state.get_cooldown_until("openrouter")
    assert time.time() < until <= time.time() + 60.0

    await state.clear_cooldown("openrouter")
    assert await state.get_cooldown_until("openrouter") == 0.0


@pytest.mark.parametrize(
    ("state_factory", "backend_id"),
    _BACKEND_FACTORIES,
    ids=[b[1] for b in _BACKEND_FACTORIES],
)
@pytest.mark.asyncio
async def test_best_cache_round_trip(state_factory, backend_id):
    state = state_factory()

    assert await state.get_cached_best("answer") is None

    await state.set_cached_best("answer", "groq", 30.0)
    assert await state.get_cached_best("answer") == "groq"

    await state.clear_cached_best("answer")
    assert await state.get_cached_best("answer") is None


@pytest.mark.asyncio
async def test_inmemory_best_cache_expires():
    state = InMemoryRouterState()

    await state.set_cached_best("answer", "groq", 0.05)
    time.sleep(0.1)

    assert await state.get_cached_best("answer") is None


@pytest.mark.asyncio
async def test_redis_backend_degrades_gracefully_when_redis_down():
    """A failing Redis must not break routing: reads report no state."""

    class _BrokenRedis:
        async def get(self, key):
            raise ConnectionError("redis down")

        async def set(self, key, value, ex=None):  # noqa: A002
            raise ConnectionError("redis down")

        async def delete(self, *keys):
            raise ConnectionError("redis down")

    state = RedisRouterState(_BrokenRedis())

    assert await state.get_cooldown_until("openrouter") == 0.0
    assert await state.get_cached_best("answer") is None
    await state.set_cooldown("openrouter", 60.0)  # must not raise
    await state.set_cached_best("answer", "groq", 30.0)  # must not raise
