"""Availability-aware provider selection for the fallback chain.

Replaces the fixed-order walk of ``ProviderFallbackChain`` with a router that:

1. Serves the O(1) hot path from a short-lived cached best provider.
2. Rebuilds a health-scored ready pool when the cache misses or goes stale.
3. Tracks cooldowns in an optional shared backend (Redis) so every process
   (API, Celery workers, CLI) agrees on which providers are out.
4. Waits (via a shared gate) for at least ``wait_min_available_fraction`` of
   providers to exit cooldown when everything is down, instead of raising
   immediately.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol, cast

from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
from data_engineering_copilot.infrastructure.provider_fallback import ProviderConfig
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry

logger = logging.getLogger(__name__)

_COOLDOWN_BY_CATEGORY: dict[ProviderErrorCategory, float] = {
    ProviderErrorCategory.RETRYABLE: 10.0,
    ProviderErrorCategory.TEMPORARY_UNAVAILABLE: 10.0,
    ProviderErrorCategory.RATE_LIMITED: 60.0,
    ProviderErrorCategory.AUTHENTICATION_ERROR: 60.0,
    ProviderErrorCategory.QUOTA_EXCEEDED: 60.0,
    ProviderErrorCategory.INVALID_REQUEST: 60.0,
    ProviderErrorCategory.PERMANENT_ERROR: 60.0,
}
_DEFAULT_COOLDOWN_SECONDS = 60.0


@dataclass(slots=True)
class ProviderRouterConfig:
    """Selection + wait-policy knobs for ``ProviderSelector``."""

    purpose: str = "global"
    preference_provider: str | None = None
    preference_weight: float = 0.1
    best_cache_ttl_seconds: float = 15.0
    wait_min_available_fraction: float = 0.5
    wait_max_seconds: float = 15.0


class RouterStateBackend(Protocol):
    """Cross-process router state. Cooldowns use epoch seconds."""

    async def get_cooldown_until(self, provider: str) -> float: ...

    async def set_cooldown(self, provider: str, seconds: float) -> None: ...

    async def clear_cooldown(self, provider: str) -> None: ...

    async def get_cached_best(self, purpose: str) -> str | None: ...

    async def set_cached_best(self, purpose: str, provider: str, ttl_seconds: float) -> None: ...

    async def clear_cached_best(self, purpose: str) -> None: ...


class InMemoryRouterState:
    """Process-local router state. The default when Redis sharing is off."""

    def __init__(self) -> None:
        self._cooldowns: dict[str, float] = {}
        self._best: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    async def get_cooldown_until(self, provider: str) -> float:
        with self._lock:
            return self._cooldowns.get(provider, 0.0)

    async def set_cooldown(self, provider: str, seconds: float) -> None:
        with self._lock:
            self._cooldowns[provider] = time.time() + seconds

    async def clear_cooldown(self, provider: str) -> None:
        with self._lock:
            self._cooldowns.pop(provider, None)

    async def get_cached_best(self, purpose: str) -> str | None:
        with self._lock:
            entry = self._best.get(purpose)
            if entry is None:
                return None
            provider, until = entry
            if time.time() >= until:
                self._best.pop(purpose, None)
                return None
            return provider

    async def set_cached_best(self, purpose: str, provider: str, ttl_seconds: float) -> None:
        with self._lock:
            self._best[purpose] = (provider, time.time() + ttl_seconds)

    async def clear_cached_best(self, purpose: str) -> None:
        with self._lock:
            self._best.pop(purpose, None)


class RedisRouterState:
    """Redis-backed router state shared across processes.

    Keys: ``dec:router:cooldown:{provider}`` (TTL = cooldown) and
    ``dec:router:best:{purpose}`` (TTL = best-cache TTL).  Every Redis call is
    guarded: when Redis is unreachable the backend degrades to "no state" (no
    cooldown, no cached best) so the router keeps working with per-process
    cooldowns only.
    """

    _COOLDOWN_PREFIX = "dec:router:cooldown:"
    _BEST_PREFIX = "dec:router:best:"

    def __init__(self, redis_client=None) -> None:
        self._redis = redis_client
        self._lock = threading.Lock()

    def _client(self):
        if self._redis is not None:
            return self._redis
        with self._lock:
            if self._redis is None:
                try:
                    from data_engineering_copilot.factory import get_shared_redis_client

                    self._redis = get_shared_redis_client()
                except Exception:
                    logger.warning("Redis router state unavailable — degrading to per-process routing")
                    return None
        return self._redis

    async def get_cooldown_until(self, provider: str) -> float:
        client = self._client()
        if client is None:
            return 0.0
        try:
            raw = await client.get(self._COOLDOWN_PREFIX + provider)
            return float(raw) if raw else 0.0
        except Exception:
            return 0.0

    async def set_cooldown(self, provider: str, seconds: float) -> None:
        client = self._client()
        if client is None:
            return
        ttl = max(1, int(seconds) + 1)
        with contextlib.suppress(Exception):
            await client.set(self._COOLDOWN_PREFIX + provider, str(time.time() + seconds), ex=ttl)

    async def clear_cooldown(self, provider: str) -> None:
        client = self._client()
        if client is None:
            return
        with contextlib.suppress(Exception):
            await client.delete(self._COOLDOWN_PREFIX + provider)

    async def get_cached_best(self, purpose: str) -> str | None:
        client = self._client()
        if client is None:
            return None
        try:
            value = await client.get(self._BEST_PREFIX + purpose)
            if value is None:
                return None
            return value.decode("utf-8") if isinstance(value, bytes) else value
        except Exception:
            return None

    async def set_cached_best(self, purpose: str, provider: str, ttl_seconds: float) -> None:
        client = self._client()
        if client is None:
            return
        with contextlib.suppress(Exception):
            await client.set(self._BEST_PREFIX + purpose, provider, ex=max(1, int(ttl_seconds)))

    async def clear_cached_best(self, purpose: str) -> None:
        client = self._client()
        if client is None:
            return
        with contextlib.suppress(Exception):
            await client.delete(self._BEST_PREFIX + purpose)


class ProviderSelector:
    """Picks the best currently-callable provider.

    Not thread-safe across event loops; each asyncio chain owns one selector.
    """

    # Shared wait gates keyed by rounded wake-up epoch, so concurrent requests
    # blocked on the same all-down window sleep together (one timer, no herd).
    _wait_gates: dict[float, asyncio.Event] = {}
    _wait_gates_lock = asyncio.Lock()

    def __init__(
        self,
        providers: list[ProviderConfig],
        health: ProviderHealthRegistry,
        state: RouterStateBackend,
        config: ProviderRouterConfig,
    ) -> None:
        self.config = config
        self.state = state
        self.health = health
        self._providers = providers
        self._by_name = {p.name: p for p in providers}
        self._health = health

    @property
    def providers(self) -> list[ProviderConfig]:
        return self._providers

    async def pick(self, exclude: set[str] | None = None) -> ProviderConfig | None:
        """Return the best callable provider, or ``None`` when none is ready."""
        exclude = exclude or set()

        cached = await self.state.get_cached_best(self.config.purpose)
        if cached is not None and cached in self._by_name and cached not in exclude and await self._is_callable(cached):
            self._health.mark_selected(cached)
            return self._by_name[cached]

        ready = [p for p in self._providers if p.name not in exclude and await self._is_callable(p.name)]
        if not ready:
            return None

        ready.sort(
            key=lambda p: (
                self._score(p.name),
                -self._health.get_last_selected(p.name),
            ),
            reverse=True,
        )
        best = ready[0]
        await self.state.set_cached_best(self.config.purpose, best.name, self.config.best_cache_ttl_seconds)
        self._health.mark_selected(best.name)
        return best

    async def seconds_until_fraction(self, fraction: float, exclude: set[str] | None = None) -> float:
        """Seconds until at least ``fraction`` of providers are callable.

        Providers currently callable contribute ``0.0``. Returns ``0.0`` when
        the threshold is already met.
        """
        exclude = exclude or set()
        remaining: list[float] = []
        for provider in self._providers:
            if provider.name in exclude:
                continue
            remaining.append(await self._seconds_until_callable(provider.name))
        if not remaining:
            return 0.0
        remaining.sort()
        required = max(1, math.ceil(fraction * len(remaining)))
        return remaining[required - 1]

    async def wait_for_availability(
        self,
        deadline_monotonic: float,
        exclude: set[str] | None = None,
    ) -> bool:
        """Block until the availability fraction is met or *deadline_monotonic*.

        Re-loops while providers keep failing: each iteration waits (bounded by
        ``wait_max_seconds``) until the fraction is met, then returns ``True``.
        Returns ``False`` once the deadline passes — the caller should degrade.
        """
        exclude = exclude or set()
        while True:
            if time.monotonic() >= deadline_monotonic:
                return False
            wake_seconds = await self.seconds_until_fraction(
                self.config.wait_min_available_fraction,
                exclude=exclude,
            )
            if wake_seconds <= 0:
                return True
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                return False
            await self._sleep_shared(min(wake_seconds, self.config.wait_max_seconds, remaining))

    async def record_success(self, provider: str, model: str, latency: float) -> None:
        self._health.track_success(provider, model, latency)
        await self.state.clear_cooldown(provider)

    async def record_failure(
        self,
        provider: str,
        model: str,
        category: ProviderErrorCategory,
        retry_after: float | None = None,
    ) -> None:
        self._health.track_failure(provider, model, category, retry_after)
        remaining = self._health.get_effective_cooldown_remaining(provider, model)
        if remaining > 0:
            await self.state.set_cooldown(provider, remaining)

    async def _seconds_until_callable(self, name: str) -> float:
        remaining = 0.0
        ph = self._health.get_provider_health(name)
        if ph is not None:
            model_cooldown = max((mh.cooldown_until for mh in ph.models.values()), default=0.0)
            local = max(ph.cooldown_until, model_cooldown)
            remaining = max(remaining, local - time.monotonic())
        state_until = await self.state.get_cooldown_until(name)
        remaining = max(remaining, state_until - time.time())
        return max(0.0, remaining)

    async def _is_callable(self, name: str) -> bool:
        if await self._seconds_until_callable(name) > 0:
            return False
        provider = self._by_name.get(name)
        if provider is None or provider.rate_limiter is None:
            return True
        return provider.rate_limiter.wait_until_available() <= 0

    def _score(self, name: str) -> float:
        score = self._health.get_provider_score(name)
        if self.config.preference_provider and name == self.config.preference_provider:
            score += self.config.preference_weight
        return score

    async def _sleep_shared(self, seconds: float) -> None:
        """Sleep once for the group: waiters on the same wake-up share a gate."""
        fire = round(time.time() + seconds, 1)
        async with self._wait_gates_lock:
            gate = self._wait_gates.get(fire)
            if gate is None:
                gate = asyncio.Event()
                self._wait_gates[fire] = gate
        try:
            await asyncio.wait_for(gate.wait(), timeout=max(0.0, seconds + 0.05))
        except TimeoutError:
            pass
        finally:
            gate.set()
            async with self._wait_gates_lock:
                self._wait_gates.pop(fire, None)


def cooldown_duration_for(category: ProviderErrorCategory, retry_after: float | None = None) -> float:
    """Flat cooldown fallback for a failure category (no registry available)."""
    if retry_after is not None:
        return max(0.0, cast(float, retry_after))
    return _COOLDOWN_BY_CATEGORY.get(category, _DEFAULT_COOLDOWN_SECONDS)
