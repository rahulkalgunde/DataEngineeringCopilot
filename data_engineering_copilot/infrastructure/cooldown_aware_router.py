"""Cooldown-aware embedding router.

Wraps ``ProviderFallbackChain`` so that, when all external embedding providers
are rate-limited or in cooldown, the router waits for the shortest cooldown
instead of immediately degrading to the local fallback. Local embedding is
used only after the wait budget is exhausted.

This keeps gen-build and other batch embedding jobs on the fast external
path whenever possible, rather than blocking on local CPUs while providers
cool down.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from data_engineering_copilot.domain.models import EmbeddingRequest
from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry


class CooldownAwareEmbeddingRouter:
    """Wait for external provider cooldown before degrading to local embedding.

    The wrapper is transparent to callers: it implements the same
    ``embed_texts`` / ``embed_query`` surface as ``FallbackEmbedder``, and
    delegates to the wrapped ``ProviderFallbackChain`` whenever an external
    provider is available.

    Before each call, the router checks whether any external provider is
    available. If none are available, it sleeps for the shortest remaining
    cooldown (capped by ``max_cooldown_wait_s``) and rechecks. Only when the
    wait budget is exhausted does it call the chain and let it fall back to
    the degraded local provider.
    """

    def __init__(
        self,
        chain: ProviderFallbackChain,
        health: ProviderHealthRegistry,
        max_cooldown_wait_s: int = 60,
    ) -> None:
        self._chain = chain
        self._health = health
        self._max_cooldown_wait_s = max_cooldown_wait_s

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with cooldown-aware routing."""
        await self._wait_for_external_if_cooldown()

        return await self._chain.execute(EmbeddingRequest(input_type="passage", texts=list(texts)))

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        results = await self.embed_texts([text])
        if not results or results[0] is None:
            raise ValueError("embed_query returned no embedding")
        return results[0]

    async def close(self) -> None:
        """Propagate close to the wrapped chain."""
        if hasattr(self._chain, "close"):
            await self._chain.close()

    @property
    def inner(self) -> Any:
        """Expose the wrapped chain for introspection/testing."""
        return self._chain

    async def _wait_for_external_if_cooldown(self) -> None:
        """Wait for the shortest external cooldown, up to the configured budget."""
        shortest = self._shortest_external_cooldown()
        if shortest <= 0:
            return

        wait_time = min(shortest, self._max_cooldown_wait_s)
        await asyncio.sleep(wait_time)

    def _shortest_external_cooldown(self) -> float:
        """Return the shortest remaining cooldown among external providers."""
        shortest = float("inf")
        for provider in self._chain._config.providers:
            if provider.name.lower() == "ollama":
                continue
            health = self._health.get_provider_health(provider.name)
            if health is None:
                continue
            for model_health in health.models.values():
                remaining = max(0.0, model_health.cooldown_until - time.monotonic())
                if remaining > 0:
                    shortest = min(shortest, remaining)
        return shortest if shortest != float("inf") else 0.0
