"""OfflineEmbeddingWaitController behavioral tests.

Pins the contract that the batch wall depends on:

* The backoff level PERSISTS across ``embed_texts`` calls — it is not reset
  per batch, so a sustained 503 outage escalates 10→20→40→…→cap instead of
  looking like a flat 10s retry.
* The sleep is never clamped below the exponential schedule by a short
  provider cooldown (``max``, not ``min``).
* The cumulative wait budget is shared across the whole controller lifetime
  (a paused controller pauses immediately the next call).
* One success cools the level by one notch (no hard reset); the cap holds.

Uses synthetic time: ``asyncio.sleep`` is replaced by a recorder that also
advances a fake ``time.monotonic``, so cooldowns expire deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from data_engineering_copilot.domain.exceptions import ProviderError, ProviderErrorCategory
from data_engineering_copilot.infrastructure.offline_embedding_wait import (
    OfflineEmbeddingPaused,
    OfflineEmbeddingWaitController,
)
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeProviderHealth:
    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self.cooldown_until = 0.0
        self.models: dict = {}

    @property
    def is_available(self) -> bool:
        return self._clock.now >= self.cooldown_until


class _FakeHealth:
    def __init__(self, ph: _FakeProviderHealth) -> None:
        self._ph = ph

    def get_provider_health(self, name: str) -> _FakeProviderHealth | None:
        if name != "nvidia":
            return None
        return self._ph


class _FakeChain:
    def __init__(self, ph: _FakeProviderHealth, outcomes: list[str], cooldown_s: float = 0.5) -> None:
        self._config = SimpleNamespace(
            providers=[SimpleNamespace(name="nvidia", rate_limiter=None)],
            degraded_fallback=None,
        )
        self._ph = ph
        self._outcomes = list(outcomes)
        self._cooldown_s = cooldown_s
        self.calls = 0

    async def execute(self, request) -> object:
        self.calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else "ok"
        if outcome == "fail":
            self._ph.cooldown_until = self._ph._clock.now + self._cooldown_s
            raise ProviderError(  # 503 -> TEMPORARY_UNAVAILABLE (single client attempt)
                ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
                "nvidia",
                "nvidia/nemotron-3-embed-1b",
                message="Service Unavailable (503)",
            )
        return [request]


@pytest.fixture
def fake_time(monkeypatch):
    """Install a fake `asyncio.sleep` + `time.monotonic` driven by one clock."""

    def _install(clock: _Clock) -> list[float]:
        sleeps: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            if seconds is not None and seconds > 0:
                sleeps.append(float(seconds))
                clock.advance(float(seconds))

        monkeypatch.setattr("data_engineering_copilot.infrastructure.offline_embedding_wait.asyncio.sleep", _fake_sleep)
        monkeypatch.setattr(
            "data_engineering_copilot.infrastructure.offline_embedding_wait.time.monotonic", clock.monotonic
        )
        return sleeps

    return _install


def _make_controller(clock: _Clock, ph: _FakeProviderHealth, chain: _FakeChain, **kwargs):
    return OfflineEmbeddingWaitController(
        chain=chain,
        health=cast(ProviderHealthRegistry, _FakeHealth(ph)),
        app_settings=None,
        backoff_base_s=1.0,
        backoff_cap_s=4.0,
        jitter=0.0,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_backoff_level_persists_across_embed_texts_calls(fake_time):
    """Level + budget survive embed_texts; a paused call stays paused next time."""
    clock = _Clock()
    sleeps = fake_time(clock)
    ph = _FakeProviderHealth(clock)
    chain = _FakeChain(ph, outcomes=["fail"] * 20)
    controller = _make_controller(clock, ph, chain, max_wait_s=7.0)

    with pytest.raises(OfflineEmbeddingPaused) as exc1:
        await controller.embed_texts(["t"])
    # 1 + 2 + 4 = 7s cumulative, then even the capped 4s step would exceed budget.
    assert sleeps == [1.0, 2.0, 4.0]
    assert exc1.value.waited_s == 7.0
    assert controller._waited_s == 7.0
    assert controller._attempt == 3

    # Next call: provider still failing, budget already consumed -> pause
    # immediately WITHOUT another sleep (state persisted, not per-batch).
    with pytest.raises(OfflineEmbeddingPaused) as exc2:
        await controller.embed_texts(["t"])
    assert exc2.value.waited_s == 7.0
    assert sleeps == [1.0, 2.0, 4.0]
    assert controller._attempt == 3


@pytest.mark.asyncio
async def test_short_cooldown_does_not_clamp_sleep_below_schedule(fake_time):
    """Sleep = max(backoff_level, min_wait) — a tiny cooldown must not flatten escalation."""
    clock = _Clock()
    sleeps = fake_time(clock)
    ph = _FakeProviderHealth(clock)
    chain = _FakeChain(ph, outcomes=["fail", "ok"], cooldown_s=0.2)
    controller = _make_controller(clock, ph, chain, max_wait_s=100.0)

    await controller.embed_texts(["t"])
    # First sleep is the level-0 backoff (1.0), NOT the 0.2s cooldown min_wait.
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_one_success_cools_level_by_one_notch(fake_time):
    """Two failures then a success: level 2 -> 1 (decay), not a hard reset to 0."""
    clock = _Clock()
    sleeps = fake_time(clock)
    ph = _FakeProviderHealth(clock)
    chain = _FakeChain(ph, outcomes=["fail", "fail", "ok"])
    controller = _make_controller(clock, ph, chain, max_wait_s=100.0)

    await controller.embed_texts(["t"])
    assert sleeps == [1.0, 2.0]
    assert controller._attempt == 1  # would be 0 if success hard-reset


@pytest.mark.asyncio
async def test_backoff_caps_at_configured_cap(fake_time):
    """Level keeps climbing but the sleep never exceeds the cap."""
    clock = _Clock()
    sleeps = fake_time(clock)
    ph = _FakeProviderHealth(clock)
    chain = _FakeChain(ph, outcomes=["fail"] * 50)
    controller = _make_controller(clock, ph, chain, max_wait_s=60.0)

    with pytest.raises(OfflineEmbeddingPaused):
        await controller.embed_texts(["t"])
    assert sleeps
    assert max(sleeps) == 4.0
    assert len(sleeps) > 5  # many capped steps fit inside the budget


@pytest.mark.asyncio
async def test_success_decays_fully_after_clean_streak(fake_time):
    """After enough successes the level reaches 0 again."""
    clock = _Clock()
    sleeps = fake_time(clock)
    ph = _FakeProviderHealth(clock)
    chain = _FakeChain(ph, outcomes=["fail", "ok", "ok", "ok"])
    controller = _make_controller(clock, ph, chain, max_wait_s=100.0)

    await controller.embed_texts(["t"])  # fail -> sleep 1 -> ok (level 1 -> 0)
    assert controller._attempt == 0
    assert sleeps == [1.0]
    for _ in range(3):  # 3 clean successes, no sleeps
        await controller.embed_texts(["t"])
    assert sleeps == [1.0]
    assert controller._attempt == 0
