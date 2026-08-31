"""Offline bulk embedding wait controller.

Wraps a ``ProviderFallbackChain`` for the offline batch path (gen-build /
PinnedIndexBuilder / SparkIndexBuilder).  Semantics:

* **Collective gate:** before *each* ``embed_texts`` call, check whether
  *any* of the allowed offline providers (default ``nvidia|openrouter|
  huggingface``) is currently callable — i.e. health is not in cooldown
  *and* ``rate_limiter.wait_until_available() == 0``.  If at least one is
  callable, delegate to the wrapped chain immediately — no pre-sleep before
  trying fallbacks within the call.  The chain itself probes
  ``nvidia → openrouter → hf`` via ``try_acquire``; callers should not sleep
  before the probe.

* **Wait only when none callable.**  When the collective check is
  ``False``, enter an exponential backoff loop with jitter.  Cumulative
  *wait-time only* (not execution time) is capped by
  ``offline_embedding_max_wait_s`` (default 3600).  When the budget is
  exhausted, raise ``OfflineEmbeddingPaused`` after flushing checkpoint
  state — the builder will catch it and exit gracefully for resume.

Online query embeddings (``AsyncRagService``) do NOT use this controller —
they keep fail-fast behaviour.

Fail-open for auxiliary verifiers is unchanged.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
from data_engineering_copilot.domain.models import EmbeddingRequest
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry


class OfflineEmbeddingPaused(RuntimeError):
    """Raised when offline wait budget exhausted — caller should checkpoint and exit."""

    def __init__(self, waited_s: float, max_wait_s: float) -> None:
        super().__init__(f"Offline embedding paused after {waited_s:.1f}s wait (budget {max_wait_s:.1f}s)")
        self.waited_s = waited_s
        self.max_wait_s = max_wait_s


class OfflineEmbeddingWaitController:
    """Collective-gate wait controller for offline bulk embedding.

    Implements ``EmbedderProtocol`` (``embed_texts`` / ``embed_query`` /
    ``close`` / ``inner``) by delegating to the wrapped
    ``ProviderFallbackChain``.  Only sleeps when *none* of the offline pool is
    callable; otherwise calls through instantly.
    """

    def __init__(
        self,
        chain: Any,
        health: ProviderHealthRegistry,
        app_settings: AppSettings | None = None,
        *,
        max_wait_s: float | None = None,
        backoff_base_s: float | None = None,
        backoff_cap_s: float | None = None,
        jitter: float | None = None,
        rpd_wait: bool | None = None,
    ) -> None:
        self._chain = chain
        self._health = health
        # Late import to avoid cycle; AppSettings already imported but keep
        # fallback if caller passes None (tests).
        if app_settings is None:
            # Use defaults aligned with AppSettings defaults.
            self._max_wait_s = max_wait_s if max_wait_s is not None else 3600.0
            self._backoff_base_s = backoff_base_s if backoff_base_s is not None else 10.0
            self._backoff_cap_s = backoff_cap_s if backoff_cap_s is not None else 60.0
            self._jitter = jitter if jitter is not None else 0.2
            self._rpd_wait = rpd_wait if rpd_wait is not None else True
        else:
            self._max_wait_s = (
                max_wait_s if max_wait_s is not None else float(app_settings.offline_embedding_max_wait_s)
            )
            self._backoff_base_s = (
                backoff_base_s if backoff_base_s is not None else float(app_settings.offline_embedding_backoff_base_s)
            )
            self._backoff_cap_s = (
                backoff_cap_s if backoff_cap_s is not None else float(app_settings.offline_embedding_backoff_cap_s)
            )
            self._jitter = jitter if jitter is not None else float(app_settings.offline_embedding_jitter)
            self._rpd_wait = rpd_wait if rpd_wait is not None else bool(app_settings.offline_embedding_rpd_wait)

    @property
    def inner(self) -> Any:
        return self._chain

    async def close(self) -> None:
        if hasattr(self._chain, "close"):
            await self._chain.close()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._execute_with_collective_wait(EmbeddingRequest(input_type="passage", texts=list(texts)))

    async def embed_query(self, text: str) -> list[float]:
        results = await self._execute_with_collective_wait(EmbeddingRequest(input_type="query", texts=[text]))
        if not results or results[0] is None:
            raise ValueError("embed_query returned no embedding")
        return results[0]

    # Internal

    def _is_provider_callable(self, provider_name: str) -> tuple[bool, float]:
        """Return (callable, wait_seconds) for one provider.

        Callable requires both health and rate-limiter to be clear.
        """
        # Health gate (cooldown)
        ph = self._health.get_provider_health(provider_name)
        if ph is not None:
            if not ph.is_available:
                # provider-level cooldown
                wait = max(0.0, ph.cooldown_until - time.monotonic())
                return False, wait
            # model-level cooldown — if all models are cooling, not callable
            model_cooldowns = [mh.cooldown_until for mh in ph.models.values()]
            if model_cooldowns:
                max_cooldown = max(model_cooldowns)
                # If any model is available, provider is callable (pool size 1: only one model)
                any_available = any(mh.is_available for mh in ph.models.values())
                if not any_available:
                    wait = max(0.0, max_cooldown - time.monotonic())
                    return False, wait
        # Rate limiter gate
        # Find the ProviderConfig to read its limiter. Chain config keeps them.
        cfg = getattr(self._chain, "_config", None)
        rl = None
        if cfg is not None:
            for pc in list(getattr(cfg, "providers", [])) + (
                [cfg.degraded_fallback] if getattr(cfg, "degraded_fallback", None) else []
            ):
                if pc.name.lower() == provider_name.lower():
                    rl = pc.rate_limiter
                    break
        if rl is not None:
            wait = rl.wait_until_available()
            # RPD path — if caller disallows RPD waits, treat as hard-exhausted
            if not self._rpd_wait and rl._rpd_limit > 0 and rl._daily_count >= rl._rpd_limit:
                # Return large wait so collective gate sees "none callable"
                return False, wait if wait > 0 else 86400.0
            if wait > 0:
                return False, wait
        return True, 0.0

    def _any_callable(self) -> tuple[bool, float]:
        """Check collective gate.

        Returns (any_callable, min_wait_until_any).  When any_callable is
        True, caller should try immediately.  Otherwise min_wait_until_any
        is the smallest wait among providers (seconds until *some* one frees).
        """
        cfg = getattr(self._chain, "_config", None)
        providers = getattr(cfg, "providers", []) if cfg is not None else []
        if not providers:
            return False, 0.0
        any_callable = False
        min_wait = float("inf")
        for pc in providers:
            callable_, wait = self._is_provider_callable(pc.name)
            if callable_:
                any_callable = True
                break
            min_wait = min(min_wait, wait)
        if any_callable:
            return True, 0.0
        # None callable — return the earliest freeing provider's wait.
        # If all waits are 0 but still none callable (e.g. degraded-fallback-only edge), return 0.
        if min_wait == float("inf"):
            return False, 0.0
        return False, max(0.0, min_wait)

    def _backoff_step(self, attempt: int) -> float:
        base = self._backoff_base_s * (2**attempt)
        capped = min(base, self._backoff_cap_s)
        if self._jitter and self._jitter > 0:
            # ± jitter fraction; jitter 0.2 => 0.8..1.2
            jitter_factor = random.uniform(1.0 - self._jitter, 1.0 + self._jitter)
            capped *= jitter_factor
        return max(0.5, capped)

    async def _execute_with_collective_wait(self, request: Any) -> Any:
        waited = 0.0
        attempt = 0
        while True:
            any_callable, min_wait = self._any_callable()
            if any_callable:
                # Try immediately — chain will probe nvidia→or→hf via try_acquire
                try:
                    return await self._chain.execute(request)
                except Exception as exc:
                    # Only wait on rate-limit / temporary categories; others (dimension, permanent) fail fast.
                    from data_engineering_copilot.domain.exceptions import ProviderError  # local import

                    cat = None
                    if isinstance(exc, ProviderError):
                        cat = exc.category
                    else:
                        # Fallback chain wraps as LLMClientError whose cause is ProviderError; unwrap
                        cause = getattr(exc, "__cause__", None)
                        if isinstance(cause, ProviderError):
                            cat = cause.category
                    retryable_cats = {
                        ProviderErrorCategory.RATE_LIMITED,
                        ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
                        ProviderErrorCategory.RETRYABLE,
                        ProviderErrorCategory.QUOTA_EXCEEDED,
                    }
                    if cat not in retryable_cats:
                        raise
                    # This provider is now on cooldown; loop will re-check collective gate and potentially sleep.
                    # Don't sleep here — go back to collective gate to see if a sibling is still free.
                    # Small yield to avoid tight loop.
                    await asyncio.sleep(0)
                    any_callable2, _ = self._any_callable()
                    if any_callable2:
                        continue
                    # else fall through to backoff sleep below
                else:
                    # Should not reach here (return already in try)
                    pass
            # None callable — need to sleep with exponential backoff (wait-time only budget)
            # Determine desired sleep: min(backoff, min_wait_until_any)
            desired = min(self._backoff_step(attempt), min_wait if min_wait > 0 else self._backoff_cap_s)
            if waited + desired > self._max_wait_s:
                raise OfflineEmbeddingPaused(waited_s=waited, max_wait_s=self._max_wait_s)
            await asyncio.sleep(desired)
            waited += desired
            attempt += 1
