"""Unified provider fallback chain for any async client protocol.

Replaces:
- AdaptiveLLMRouter (llm-specific logic moved here)
- AdaptiveRagasEmbeddings._embed_with_failover (duplicate logic)

Supports both LLM and Embedding providers via protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, cast

import structlog

from data_engineering_copilot.domain.exceptions import ProviderError, ProviderErrorCategory
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = structlog.get_logger(__name__)

T = TypeVar("T")  # Request type
R = TypeVar("R")  # Response type


class ProviderClient(Protocol):
    """Minimal protocol for any provider client (LLM or Embedding)."""

    @property
    def model(self) -> str: ...

    async def call(self, request: Any) -> Any: ...

    async def close(self) -> None: ...

    @property
    def last_usage(self) -> Any: ...


class ErrorCategorizer(Protocol):
    """Maps provider exceptions to ProviderErrorCategory."""

    def __call__(self, exc: Exception, provider: str, model: str) -> ProviderError: ...


@dataclass(slots=True)
class ProviderConfig:
    """Declarative provider configuration."""

    name: str
    client: ProviderClient
    rate_limiter: SlidingWindowRateLimiter | None = None


@dataclass(slots=True)
class FallbackChainConfig:
    """Declarative fallback chain configuration."""

    providers: list[ProviderConfig] = field(default_factory=list)
    degraded_fallback: ProviderConfig | None = None
    max_degraded_consecutive_failures: int = 3
    error_categorizer: ErrorCategorizer | None = None


def _default_categorizer(exc: Exception, provider: str, model: str) -> ProviderError:
    """Default error categorizer — treats everything as RETRYABLE."""
    import httpx

    from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

    def _status_category(status: int) -> ProviderErrorCategory:
        if status == 429:
            return ProviderErrorCategory.RATE_LIMITED
        if status in (401, 403):
            return ProviderErrorCategory.AUTHENTICATION_ERROR
        if status in (400, 422):
            return ProviderErrorCategory.INVALID_REQUEST
        if status >= 500:
            return ProviderErrorCategory.TEMPORARY_UNAVAILABLE
        return ProviderErrorCategory.PERMANENT_ERROR

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        retry_after = SlidingWindowRateLimiter.parse_retry_after(dict(exc.response.headers))
        return ProviderError(
            _status_category(status),
            provider,
            model,
            retry_after=retry_after if status == 429 else None,
            original=exc,
        )

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, TimeoutError, OSError)):
        return ProviderError(ProviderErrorCategory.RETRYABLE, provider, model, original=exc)

    lower_msg = str(exc).lower()
    if "rate limit" in lower_msg or "429" in lower_msg or "too many requests" in lower_msg:
        return ProviderError(ProviderErrorCategory.RATE_LIMITED, provider, model, original=exc)
    if "quota" in lower_msg or "exceeded" in lower_msg:
        return ProviderError(ProviderErrorCategory.QUOTA_EXCEEDED, provider, model, original=exc)
    if "401" in lower_msg or "unauthorized" in lower_msg or "authentication" in lower_msg:
        return ProviderError(ProviderErrorCategory.AUTHENTICATION_ERROR, provider, model, original=exc)
    if "timed out" in lower_msg or "timeout" in lower_msg:
        return ProviderError(ProviderErrorCategory.RETRYABLE, provider, model, original=exc)
    if "could not reach" in lower_msg or "connection" in lower_msg:
        return ProviderError(ProviderErrorCategory.RETRYABLE, provider, model, original=exc)

    return ProviderError(ProviderErrorCategory.PERMANENT_ERROR, provider, model, original=exc)


class ProviderFallbackChain[T, R]:
    """Generic fail-fast, failover-first provider chain.

    Usage:
        chain = ProviderFallbackChain(
            providers=[ProviderConfig("openrouter", client1), ...],
            degraded_fallback=ProviderConfig("ollama", ollama_client),
            error_categorizer=categorize_llm_error,
        )
        result = await chain.execute(request)

    Behavior:
    1. Pre-flight gate: skip providers in cooldown or rate-limited
    2. Try each provider once (no same-provider retries)
    3. On failure: categorize error, set cooldown, continue to next
    4. If all external fail: try degraded_fallback (if available)
    5. Raise aggregated error if all exhausted
    """

    def __init__(
        self,
        config: FallbackChainConfig,
        health: ProviderHealthRegistry,
    ) -> None:
        if not config.providers and config.degraded_fallback is None:
            raise ValueError("Fallback chain requires at least one provider or a degraded fallback")
        self._config = config
        self._health = health
        self._error_categorizer = config.error_categorizer or _default_categorizer
        self._last_error: ProviderError | None = None

    async def execute(self, request: T) -> R:
        """Execute request through the fallback chain."""
        # Split main providers and degraded fallback
        main = [p for p in self._config.providers if p.name.lower() != "ollama"]
        degraded = self._config.degraded_fallback

        # If all providers are Ollama, treat them as main
        if not main and degraded:
            main = [degraded]
            degraded = None

        total = len(main) + (1 if degraded else 0)

        logger.info(
            "fallback_chain_started",
            candidates=[{"provider": p.name, "model": p.client.model} for p in main],
            degraded_fallback=degraded is not None,
            total=total,
        )

        attempted: list[dict[str, str]] = []
        self._last_error = None

        # Try each main provider
        for position, provider in enumerate(main, start=1):
            available, reason, available_in = self._provider_gate(provider)
            if not available:
                if available_in > 0:
                    self._health.mark_provider_cooldown(provider.name, available_in)
                logger.info(
                    "provider_skipped",
                    provider=provider.name,
                    model=provider.client.model,
                    position=f"{position}/{total}",
                    reason=reason,
                    available_in_seconds=round(available_in, 1),
                )
                attempted.append({"provider": provider.name, "outcome": f"skipped:{reason}"})
                continue

            logger.info(
                "provider_call",
                provider=provider.name,
                model=provider.client.model,
                position=f"{position}/{total}",
                reason="available",
            )

            try:
                return await self._call_with_health(provider, request)
            except ProviderError as p_err:
                self._last_error = p_err
                attempted.append({"provider": provider.name, "outcome": f"failed:{p_err.category.value}"})

        # Degraded fallback
        if degraded:
            logger.warning(
                "all_external_unavailable",
                skipped=attempted,
                fallback=degraded.name,
            )
            available, consecutive = self._degraded_available(degraded)
            if not available:
                logger.warning(
                    "degraded_fallback_skipped",
                    provider=degraded.name,
                    model=degraded.client.model,
                    reason="consecutive_failures",
                    consecutive_failures=consecutive,
                    max_consecutive_failures=self._config.max_degraded_consecutive_failures,
                )
                attempted.append({"provider": degraded.name, "outcome": f"skipped:degraded({consecutive})"})
            else:
                logger.info(
                    "provider_call",
                    provider=degraded.name,
                    model=degraded.client.model,
                    position=f"{len(main) + 1}/{total}",
                    reason="degraded_no_external",
                )
                try:
                    return await self._call_with_health(degraded, request)
                except ProviderError as p_err:
                    self._last_error = p_err
                    attempted.append({"provider": degraded.name, "outcome": f"failed:{p_err.category.value}"})

        logger.error(
            "all_providers_failed",
            attempts=attempted,
            last_error=str(self._last_error) if self._last_error else "no providers configured",
        )
        from data_engineering_copilot.infrastructure.llm_client import LLMClientError

        raise LLMClientError(
            f"All providers in fallback chain failed. Attempts: {attempted}. Last error: {self._last_error}",
            retry_after=self._last_error.retry_after if isinstance(self._last_error, ProviderError) else None,
            category=self._last_error.category if isinstance(self._last_error, ProviderError) else None,
        ) from self._last_error

    def _provider_gate(self, provider: ProviderConfig) -> tuple[bool, str, float]:
        """Pre-flight availability gate.

        Returns ``(available, skip_reason, available_in_seconds)``.
        """
        ph = self._health.get_provider_health(provider.name)
        if ph is not None:
            if not ph.is_available:
                return False, "cooldown", max(0.0, ph.cooldown_until - time.monotonic())
            if not ph.models or not any(mh.is_available for mh in ph.models.values()):
                model_cooldown = max((mh.cooldown_until for mh in ph.models.values()), default=0.0)
                return False, "cooldown", max(0.0, model_cooldown - time.monotonic())

        rl = provider.rate_limiter
        if rl is not None:
            wait = rl.wait_until_available()
            if wait > 0:
                return False, "rate_limit", wait

        return True, "", 0.0

    def _degraded_available(self, provider: ProviderConfig) -> tuple[bool, int]:
        """Whether the degraded fallback is worth attempting."""
        mh = self._health.get_model_health(provider.name, provider.client.model)
        consecutive = mh.consecutive_failures if mh is not None else 0
        return consecutive < self._config.max_degraded_consecutive_failures, consecutive

    async def _call_with_health(self, provider: ProviderConfig, request: T) -> R:
        """Single-attempt call. Records health outcome and returns the result."""
        start = time.monotonic()
        try:
            result = await provider.client.call(request)
            latency = time.monotonic() - start
            self._health.track_success(provider.name, provider.client.model, latency)
            logger.info(
                "provider_success",
                provider=provider.name,
                model=provider.client.model,
                latency_seconds=round(latency, 2),
            )
            return result
        except Exception as exc:
            p_err = self._error_categorizer(exc, provider.name, provider.client.model)
            self._health.track_failure(provider.name, provider.client.model, p_err.category, p_err.retry_after)
            logger.warning(
                "provider_failed",
                provider=provider.name,
                model=provider.client.model,
                category=p_err.category.value,
                retry_after=p_err.retry_after,
                error=str(exc),
            )
            raise p_err from exc

    @property
    def last_error(self) -> ProviderError | None:
        return self._last_error

    @property
    def model(self) -> str:
        """Return the first provider's model for backward compatibility."""
        all_providers = self._config.providers + (
            [self._config.degraded_fallback] if self._config.degraded_fallback else []
        )
        return all_providers[0].client.model if all_providers else ""

    @property
    def last_usage(self):
        """Return the last usage from the first provider."""
        all_providers = self._config.providers + (
            [self._config.degraded_fallback] if self._config.degraded_fallback else []
        )
        if all_providers and hasattr(all_providers[0].client, "last_usage"):
            return all_providers[0].client.last_usage
        from data_engineering_copilot.domain.models import LLMUsage

        return LLMUsage()

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Generate using the fallback chain (delegates to execute)."""
        return cast(str, await self.execute(cast(T, prompt)))

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ):
        """Stream tokens — not supported via fallback chain, falls back to generate."""
        result = await self.generate(prompt, temperature=temperature)
        yield result

    async def close(self) -> None:
        """Close all provider clients."""
        for provider in self._config.providers:
            if hasattr(provider.client, "close"):
                await provider.client.close()
        if self._config.degraded_fallback and hasattr(self._config.degraded_fallback.client, "close"):
            await self._config.degraded_fallback.client.close()
