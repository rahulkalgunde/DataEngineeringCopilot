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
    router_deadline_seconds: float = 45.0


class RouterLike(Protocol):
    """Minimal surface ``ProviderFallbackChain`` needs from the router.

    Declared as a protocol (not a direct import) to keep ``provider_fallback``
    free of a circular import with ``provider_selector``.
    """

    async def pick(self, exclude: set[str] | None = None) -> ProviderConfig | None: ...

    async def wait_for_availability(
        self,
        deadline_monotonic: float,
        exclude: set[str] | None = None,
    ) -> bool: ...

    async def record_success(self, provider: str, model: str, latency: float) -> None: ...

    async def record_failure(
        self,
        provider: str,
        model: str,
        category: ProviderErrorCategory,
        retry_after: float | None = None,
    ) -> None: ...


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
        router: RouterLike | None = None,
    ) -> None:
        if not config.providers and config.degraded_fallback is None:
            raise ValueError("Fallback chain requires at least one provider or a degraded fallback")
        self._config = config
        self._health = health
        self._router = router
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

        if self._router is not None and main:
            return await self._execute_with_router(request, main, degraded, total)

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
            entries, result = await self._try_degraded(degraded, request, len(main) + 1, total)
            attempted.extend(entries)
            if result is not None:
                return result

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

    async def _call_with_health(
        self,
        provider: ProviderConfig,
        request: T,
        router: RouterLike | None = None,
    ) -> R:
        """Single-attempt call. Records health outcome and returns the result."""
        start = time.monotonic()
        try:
            result = await provider.client.call(request)
            latency = time.monotonic() - start
            if router is not None:
                await router.record_success(provider.name, provider.client.model, latency)
            else:
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
            if router is not None:
                await router.record_failure(provider.name, provider.client.model, p_err.category, p_err.retry_after)
            else:
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

    async def _execute_with_router(
        self,
        request: T,
        main: list[ProviderConfig],
        degraded: ProviderConfig | None,
        total: int,
    ) -> R:
        """Router-driven execution: pick → call → re-pick on failure → wait when
        everything is down, re-looping until the deadline, then degrade."""
        attempted: list[dict[str, str]] = []
        self._last_error = None
        failed: set[str] = set()
        deadline = time.monotonic() + self._config.router_deadline_seconds
        assert self._router is not None

        while True:
            provider = await self._router.pick(exclude=failed)
            if provider is None:
                recovered = await self._router.wait_for_availability(deadline, exclude=failed)
                if not recovered:
                    break
                continue
            logger.info(
                "provider_call",
                provider=provider.name,
                model=provider.client.model,
                reason="router_selected",
            )
            try:
                return await self._call_with_health(provider, request, self._router)
            except ProviderError as p_err:
                self._last_error = p_err
                failed.add(provider.name)
                attempted.append({"provider": provider.name, "outcome": f"failed:{p_err.category.value}"})
                if time.monotonic() >= deadline:
                    break

        if degraded:
            logger.warning(
                "all_external_unavailable",
                skipped=attempted,
                fallback=degraded.name,
            )
            entries, result = await self._try_degraded(degraded, request, len(main) + 1, total)
            attempted.extend(entries)
            if result is not None:
                return result

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

    async def _try_degraded(
        self,
        degraded: ProviderConfig,
        request: T,
        position: int,
        total: int,
    ) -> tuple[list[dict[str, str]], R | None]:
        """Attempt the degraded fallback once.

        Returns ``(attempt_entries, result)`` — ``result`` is the response when
        the degraded provider succeeded, otherwise ``None``.
        """
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
            return [{"provider": degraded.name, "outcome": f"skipped:degraded({consecutive})"}], None
        logger.info(
            "provider_call",
            provider=degraded.name,
            model=degraded.client.model,
            position=f"{position}/{total}",
            reason="degraded_no_external",
        )
        try:
            result = await self._call_with_health(degraded, request, self._router)
            return [], result
        except ProviderError as p_err:
            self._last_error = p_err
            return [{"provider": degraded.name, "outcome": f"failed:{p_err.category.value}"}], None

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
        max_tokens: int | None = None,
    ) -> str:
        """Generate using the fallback chain (delegates to execute)."""
        return cast(str, await self.execute(cast(T, prompt)))

    async def _call_with_health_stream(
        self,
        provider: ProviderConfig,
        prompt: str,
        temperature: float | None = None,
        router: RouterLike | None = None,
    ):
        """Stream tokens from a single provider, tracking health on the way.

        Yields individual tokens from ``provider.client.generate_stream`` when
        available; falls back to a single non-streaming ``call`` result for
        clients that do not implement streaming. Records provider success after
        the stream completes and failure (with error categorization) on error.
        """
        start = time.monotonic()
        client = provider.client
        try:
            stream_method = getattr(client, "generate_stream", None)
            if stream_method is None:
                result = await client.call(prompt)
                latency = time.monotonic() - start
                if router is not None:
                    await router.record_success(provider.name, client.model, latency)
                else:
                    self._health.track_success(provider.name, client.model, latency)
                if result:
                    yield str(result)
                return
            async for token in stream_method(prompt, temperature=temperature):
                yield token
            latency = time.monotonic() - start
            if router is not None:
                await router.record_success(provider.name, client.model, latency)
            else:
                self._health.track_success(provider.name, client.model, latency)
        except Exception as exc:
            p_err = self._error_categorizer(exc, provider.name, client.model)
            if router is not None:
                await router.record_failure(provider.name, client.model, p_err.category, p_err.retry_after)
            else:
                self._health.track_failure(provider.name, client.model, p_err.category, p_err.retry_after)
            logger.warning(
                "provider_stream_failed",
                provider=provider.name,
                model=client.model,
                category=p_err.category.value,
                error=str(exc),
            )
            raise p_err from exc

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Stream tokens from the fallback chain.

        Mirrors ``execute``'s provider walk: skip providers in cooldown /
        rate-limited, stream from the first available provider, fall through to
        the degraded provider when all external providers fail. A provider that
        fails *before emitting any token* is skipped; a failure *after* tokens
        were emitted is re-raised — already-sent tokens cannot be retried.
        """
        main = [p for p in self._config.providers if p.name.lower() != "ollama"]
        degraded = self._config.degraded_fallback

        if not main and degraded:
            main = [degraded]
            degraded = None

        if self._router is not None and main:
            async for token in self._generate_stream_with_router(main, degraded, prompt, temperature):
                yield token
            return

        attempted: list[dict[str, str]] = []
        self._last_error = None

        for provider in main:
            available, reason, available_in = self._provider_gate(provider)
            if not available:
                attempted.append({"provider": provider.name, "outcome": f"skipped:{reason}"})
                continue
            emitted = False
            try:
                async for token in self._call_with_health_stream(provider, prompt, temperature):
                    emitted = True
                    yield token
                return  # stream completed on this provider
            except ProviderError as p_err:
                self._last_error = p_err
                attempted.append({"provider": provider.name, "outcome": f"failed:{p_err.category.value}"})
                if emitted:
                    raise

        if degraded:
            available, consecutive = self._degraded_available(degraded)
            if not available:
                attempted.append({"provider": degraded.name, "outcome": f"skipped:degraded({consecutive})"})
            else:
                emitted = False
                try:
                    async for token in self._call_with_health_stream(degraded, prompt, temperature):
                        emitted = True
                        yield token
                    return
                except ProviderError as p_err:
                    self._last_error = p_err
                    attempted.append({"provider": degraded.name, "outcome": f"failed:{p_err.category.value}"})
                    if emitted:
                        raise

        logger.error(
            "all_providers_failed_stream",
            attempts=attempted,
            last_error=str(self._last_error) if self._last_error else "no providers configured",
        )
        from data_engineering_copilot.infrastructure.llm_client import LLMClientError

        raise LLMClientError(
            f"All providers in fallback chain failed. Attempts: {attempted}. Last error: {self._last_error}",
            retry_after=self._last_error.retry_after if isinstance(self._last_error, ProviderError) else None,
            category=self._last_error.category if isinstance(self._last_error, ProviderError) else None,
        ) from self._last_error

    async def _generate_stream_with_router(
        self,
        main: list[ProviderConfig],
        degraded: ProviderConfig | None,
        prompt: str,
        temperature: float | None,
    ):
        """Router-driven streaming: pick → stream → re-pick on pre-token
        failure → wait when everything is down, then degrade."""
        attempted: list[dict[str, str]] = []
        self._last_error = None
        failed: set[str] = set()
        deadline = time.monotonic() + self._config.router_deadline_seconds
        assert self._router is not None

        while True:
            provider = await self._router.pick(exclude=failed)
            if provider is None:
                recovered = await self._router.wait_for_availability(deadline, exclude=failed)
                if not recovered:
                    break
                continue
            emitted = False
            try:
                async for token in self._call_with_health_stream(provider, prompt, temperature, self._router):
                    emitted = True
                    yield token
                return
            except ProviderError as p_err:
                self._last_error = p_err
                attempted.append({"provider": provider.name, "outcome": f"failed:{p_err.category.value}"})
                if emitted:
                    raise
                failed.add(provider.name)
                if time.monotonic() >= deadline:
                    break

        if degraded:
            available, consecutive = self._degraded_available(degraded)
            if not available:
                attempted.append({"provider": degraded.name, "outcome": f"skipped:degraded({consecutive})"})
            else:
                emitted = False
                try:
                    async for token in self._call_with_health_stream(degraded, prompt, temperature, self._router):
                        emitted = True
                        yield token
                    return
                except ProviderError as p_err:
                    self._last_error = p_err
                    attempted.append({"provider": degraded.name, "outcome": f"failed:{p_err.category.value}"})
                    if emitted:
                        raise

        logger.error(
            "all_providers_failed_stream",
            attempts=attempted,
            last_error=str(self._last_error) if self._last_error else "no providers configured",
        )
        from data_engineering_copilot.infrastructure.llm_client import LLMClientError

        raise LLMClientError(
            f"All providers in fallback chain failed. Attempts: {attempted}. Last error: {self._last_error}",
            retry_after=self._last_error.retry_after if isinstance(self._last_error, ProviderError) else None,
            category=self._last_error.category if isinstance(self._last_error, ProviderError) else None,
        ) from self._last_error

    async def close(self) -> None:
        """Close all provider clients."""
        for provider in self._config.providers:
            if hasattr(provider.client, "close"):
                await provider.client.close()
        if self._config.degraded_fallback and hasattr(self._config.degraded_fallback.client, "close"):
            await self._config.degraded_fallback.client.close()
