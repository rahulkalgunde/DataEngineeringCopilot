"""Fail-fast, failover-first LLM provider router.

Request flow
------------
1. Every provider — including the purpose-assigned one — passes an
   availability *gate* BEFORE it is called:
   - not in cooldown (``ProviderHealthRegistry``), and
   - per-provider rate limiter has a free slot (non-blocking).
2. The purpose-assigned provider is tried first, then the remaining
   external providers in ``llm_fallback_order``, each with a single attempt
   and no same-provider retries.
3. If no external provider is callable, the local Ollama client serves the
   request (degraded mode) instead of making the caller wait for cooldowns
   or rate-limit windows to elapse.

Every routing decision is emitted as a structured log event
(``llm_routing_started``, ``llm_provider_skipped``, ``llm_provider_call``,
``llm_provider_failed``, ``llm_all_external_unavailable``, ...) so the
decision trail is fully auditable from the logs alone.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import httpx
import structlog

from data_engineering_copilot.domain.exceptions import ProviderError, ProviderErrorCategory
from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.llm_client import LLMClient, LLMClientError
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = structlog.get_logger(__name__)

EXTERNAL_PROVIDERS = {"openrouter", "nvidia", "groq", "cerebras", "gemini"}


def _categorize_llm_error(exc: Exception, provider: str, model: str) -> ProviderError:
    """Categorise a provider failure, preferring structured error metadata.

    ``LLMClientError`` carries optional ``status_code`` / ``retry_after``
    attributes so the router does not depend on brittle message matching.
    """

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

    if isinstance(exc, LLMClientError) and exc.status_code is not None:
        return ProviderError(
            _status_category(exc.status_code),
            provider,
            model,
            retry_after=exc.retry_after,
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


class AdaptiveLLMRouter:
    """Routes a single LLM request across providers with fail-fast failover."""

    def __init__(
        self,
        clients: list[tuple[str, LLMClient]],
        health: ProviderHealthRegistry,
        rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
        ollama_max_consecutive_failures: int = 3,
    ) -> None:
        self._clients = clients
        self._client_map: dict[str, LLMClient] = {name: client for name, client in clients}
        self._health = health
        self._rate_limiters = rate_limiters or {}
        self._ollama_max_consecutive_failures = max(1, ollama_max_consecutive_failures)
        self._last_usage = LLMUsage()

    @property
    def model(self) -> str:
        return self._clients[0][1].model if self._clients else ""

    @property
    def last_usage(self) -> LLMUsage:
        return self._last_usage

    def _get_ollama_clients(self) -> list[tuple[str, LLMClient]]:
        return [(n, c) for n, c in self._clients if n.lower() == "ollama"]

    def _ollama_degraded_available(self, name: str, client: LLMClient) -> tuple[bool, int]:
        """Whether the degraded Ollama fallback is worth attempting.

        Once a local model has racked up ``ollama_max_consecutive_failures``
        in a row (e.g. repeated 120s timeouts), the router fails fast instead
        of stalling every request on it. Returns ``(available, consecutive)``.
        """
        mh = self._health.get_model_health(name, client.model)
        consecutive = mh.consecutive_failures if mh is not None else 0
        return consecutive < self._ollama_max_consecutive_failures, consecutive

    def _provider_gate(self, provider: str) -> tuple[bool, str, float]:
        """Pre-flight availability gate.

        Returns ``(available, skip_reason, available_in_seconds)``. A provider
        is unavailable when it is cooling down from a previous failure or when
        its rate limiter window is exhausted. Never blocks.
        """
        ph = self._health.get_provider_health(provider)
        if ph is not None:
            if not ph.is_available:
                return False, "cooldown", max(0.0, ph.cooldown_until - time.monotonic())
            if not ph.models or not any(mh.is_available for mh in ph.models.values()):
                model_cooldown = max((mh.cooldown_until for mh in ph.models.values()), default=0.0)
                return False, "cooldown", max(0.0, model_cooldown - time.monotonic())

        rl = self._rate_limiters.get(provider)
        if rl is not None:
            wait = rl.wait_until_available()
            if wait > 0:
                return False, "rate_limit", wait

        return True, "", 0.0

    async def _try_provider_once(
        self,
        name: str,
        client: LLMClient,
        prompt: str,
        temperature: float | None,
        num_predict: int | None,
        num_ctx: int | None,
    ) -> str:
        """Single-attempt call. Records health outcome and returns the text."""
        start = time.monotonic()
        try:
            text = await client.generate(
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                num_ctx=num_ctx,
            )
            latency = time.monotonic() - start
            self._health.track_success(name, client.model, latency)
            self._last_usage = client.last_usage
            logger.info(
                "llm_provider_success",
                provider=name,
                model=client.model,
                latency_seconds=round(latency, 2),
            )
            return text
        except Exception as exc:
            p_err = _categorize_llm_error(exc, name, client.model)
            self._health.track_failure(name, client.model, p_err.category, p_err.retry_after)
            model_health = self._health.get_model_health(name, client.model)
            logger.warning(
                "llm_provider_failed",
                provider=name,
                model=client.model,
                category=p_err.category.value,
                retry_after=p_err.retry_after,
                consecutive_failures=model_health.consecutive_failures if model_health else 0,
                error=str(exc),
            )
            raise p_err from exc

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        purpose_is_ollama = bool(self._clients) and self._clients[0][0].lower() == "ollama"
        main_candidates = self._clients if purpose_is_ollama else [c for c in self._clients if c[0].lower() != "ollama"]
        ollama_clients = self._get_ollama_clients() if not purpose_is_ollama else []

        total = len(main_candidates) + (1 if ollama_clients else 0)
        logger.info(
            "llm_routing_started",
            candidates=[{"provider": name, "model": client.model} for name, client in main_candidates],
            degraded_fallback=bool(ollama_clients),
            total=total,
        )

        attempted: list[dict] = []
        last_error: ProviderError | None = None

        for position, (name, client) in enumerate(main_candidates, start=1):
            available, reason, available_in = self._provider_gate(name)
            if not available:
                # Propagate rate-limiter cooldown to health registry so provider
                # is skipped on subsequent requests without re-checking the limiter
                if available_in > 0:
                    self._health.mark_provider_cooldown(name, available_in)
                logger.info(
                    "llm_provider_skipped",
                    provider=name,
                    model=client.model,
                    position=f"{position}/{total}",
                    reason=reason,
                    available_in_seconds=round(available_in, 1),
                )
                attempted.append({"provider": name, "outcome": f"skipped:{reason}"})
                continue

            logger.info(
                "llm_provider_call",
                provider=name,
                model=client.model,
                position=f"{position}/{total}",
                reason="available",
            )
            try:
                return await self._try_provider_once(
                    name,
                    client,
                    prompt,
                    temperature,
                    num_predict,
                    num_ctx,
                )
            except ProviderError as p_err:
                last_error = p_err
                attempted.append({"provider": name, "outcome": f"failed:{p_err.category.value}"})

        if ollama_clients:
            logger.warning(
                "llm_all_external_unavailable",
                skipped=attempted,
                fallback="ollama",
            )
            name, client = ollama_clients[0]
            available, consecutive = self._ollama_degraded_available(name, client)
            if not available:
                logger.warning(
                    "llm_ollama_degraded_skipped",
                    provider=name,
                    model=client.model,
                    reason="consecutive_failures",
                    consecutive_failures=consecutive,
                    max_consecutive_failures=self._ollama_max_consecutive_failures,
                )
                attempted.append({"provider": name, "outcome": f"skipped:degraded({consecutive})"})
            else:
                logger.info(
                    "llm_provider_call",
                    provider=name,
                    model=client.model,
                    position=f"{len(main_candidates) + 1}/{total}",
                    reason="degraded_no_external",
                )
                try:
                    return await self._try_provider_once(
                        name,
                        client,
                        prompt,
                        temperature,
                        num_predict,
                        num_ctx,
                    )
                except ProviderError as p_err:
                    last_error = p_err
                    attempted.append({"provider": name, "outcome": f"failed:{p_err.category.value}"})

        logger.error(
            "llm_all_providers_failed",
            attempts=attempted,
            last_error=str(last_error) if last_error else "no providers configured",
        )
        raise LLMClientError(
            f"All LLM providers in adaptive fallback chain failed. Attempts: {attempted}. Last error: {last_error}",
            retry_after=last_error.retry_after if isinstance(last_error, ProviderError) else None,
            category=last_error.category if isinstance(last_error, ProviderError) else None,
        ) from last_error

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        purpose_is_ollama = bool(self._clients) and self._clients[0][0].lower() == "ollama"
        main_candidates = self._clients if purpose_is_ollama else [c for c in self._clients if c[0].lower() != "ollama"]
        ollama_clients = self._get_ollama_clients() if not purpose_is_ollama else []

        total = len(main_candidates) + (1 if ollama_clients else 0)
        logger.info(
            "llm_routing_started_stream",
            candidates=[{"provider": name, "model": client.model} for name, client in main_candidates],
            degraded_fallback=bool(ollama_clients),
            total=total,
        )

        attempted: list[dict] = []

        for position, (name, client) in enumerate(main_candidates, start=1):
            available, reason, available_in = self._provider_gate(name)
            if not available:
                logger.info(
                    "llm_provider_skipped",
                    provider=name,
                    model=client.model,
                    position=f"{position}/{total}",
                    reason=reason,
                    available_in_seconds=round(available_in, 1),
                )
                attempted.append({"provider": name, "outcome": f"skipped:{reason}"})
                continue

            logger.info(
                "llm_provider_call",
                provider=name,
                model=client.model,
                position=f"{position}/{total}",
                reason="available",
            )
            try:
                async for token in client.generate_stream(prompt=prompt, temperature=temperature):
                    yield token
                logger.info("llm_provider_success", provider=name, model=client.model, mode="stream")
                return
            except Exception as exc:
                p_err = _categorize_llm_error(exc, name, client.model)
                # Streaming failures do not set a provider cooldown: they fall
                # back to a non-streaming generate() attempt below, which owns
                # the health bookkeeping. A provider whose streaming endpoint is
                # flaky but whose chat endpoint works should not be penalised.
                logger.warning(
                    "llm_provider_failed",
                    provider=name,
                    model=client.model,
                    category=p_err.category.value,
                    error=str(exc),
                )
                attempted.append({"provider": name, "outcome": f"failed:{p_err.category.value}"})

        if ollama_clients:
            logger.warning(
                "llm_all_external_unavailable",
                skipped=attempted,
                fallback="ollama",
            )
            name, client = ollama_clients[0]
            available, consecutive = self._ollama_degraded_available(name, client)
            if not available:
                logger.warning(
                    "llm_ollama_degraded_skipped",
                    provider=name,
                    model=client.model,
                    reason="consecutive_failures",
                    consecutive_failures=consecutive,
                    max_consecutive_failures=self._ollama_max_consecutive_failures,
                )
                attempted.append({"provider": name, "outcome": f"skipped:degraded({consecutive})"})
            else:
                logger.info(
                    "llm_provider_call",
                    provider=name,
                    model=client.model,
                    reason="degraded_no_external",
                )
                try:
                    async for token in client.generate_stream(prompt=prompt, temperature=temperature):
                        yield token
                    return
                except Exception as exc:
                    p_err = _categorize_llm_error(exc, name, client.model)
                    logger.warning(
                        "llm_provider_failed",
                        provider=name,
                        model=client.model,
                        category=p_err.category.value,
                        error=str(exc),
                    )
                    attempted.append({"provider": name, "outcome": f"failed:{p_err.category.value}"})

        logger.warning(
            "llm_all_streaming_failed",
            attempts=attempted,
            falling_back_to="generate",
        )
        result = await self.generate(prompt=prompt, temperature=temperature)
        yield result

    async def close(self) -> None:
        for _, client in self._clients:
            await client.close()
