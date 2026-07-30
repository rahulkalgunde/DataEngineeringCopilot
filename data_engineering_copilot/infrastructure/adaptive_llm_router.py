from __future__ import annotations

import asyncio
import logging
import random
import time as time_module
from collections.abc import AsyncIterator

import httpx

from data_engineering_copilot.domain.exceptions import ProviderError, ProviderErrorCategory
from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.llm_client import CircuitBreakerError, LLMClient, LLMClientError
from data_engineering_copilot.infrastructure.provider_health import ProviderHealthRegistry
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

EXTERNAL_PROVIDERS = {"openrouter", "nvidia", "groq", "cerebras", "gemini"}

LLAMA_POOL = {"llama", "deepseek", "llama3", "qwen", "gemma", "mistral", "mixtral"}
SMART_POOL = {"smart", "coder", "instruct-110b", "oss-120b"}


def _categorize_llm_error(exc: Exception, provider: str, model: str) -> ProviderError:
    msg = str(exc)

    if isinstance(exc, CircuitBreakerError):
        return ProviderError(
            ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
            provider,
            model,
            original=exc,
        )

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return ProviderError(
                ProviderErrorCategory.RATE_LIMITED,
                provider,
                model,
                original=exc,
            )
        if status in (401, 403):
            return ProviderError(
                ProviderErrorCategory.AUTHENTICATION_ERROR,
                provider,
                model,
                original=exc,
            )
        if status in (400, 422):
            return ProviderError(
                ProviderErrorCategory.INVALID_REQUEST,
                provider,
                model,
                original=exc,
            )
        if status >= 500:
            return ProviderError(
                ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
                provider,
                model,
                original=exc,
            )
        return ProviderError(
            ProviderErrorCategory.PERMANENT_ERROR,
            provider,
            model,
            original=exc,
        )

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, TimeoutError, OSError)):
        return ProviderError(
            ProviderErrorCategory.RETRYABLE,
            provider,
            model,
            original=exc,
        )

    lower_msg = msg.lower()
    if "rate limit" in lower_msg or "429" in lower_msg or "too many requests" in lower_msg:
        return ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            provider,
            model,
            original=exc,
        )
    if "quota" in lower_msg or "exceeded" in lower_msg:
        return ProviderError(
            ProviderErrorCategory.QUOTA_EXCEEDED,
            provider,
            model,
            original=exc,
        )
    if "401" in lower_msg or "unauthorized" in lower_msg or "authentication" in lower_msg:
        return ProviderError(
            ProviderErrorCategory.AUTHENTICATION_ERROR,
            provider,
            model,
            original=exc,
        )
    if "timed out" in lower_msg or "timeout" in lower_msg:
        return ProviderError(
            ProviderErrorCategory.RETRYABLE,
            provider,
            model,
            original=exc,
        )
    if "could not reach" in lower_msg or "connection" in lower_msg:
        return ProviderError(
            ProviderErrorCategory.RETRYABLE,
            provider,
            model,
            original=exc,
        )
    if "circuit breaker" in lower_msg:
        return ProviderError(
            ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
            provider,
            model,
            original=exc,
        )

    return ProviderError(
        ProviderErrorCategory.PERMANENT_ERROR,
        provider,
        model,
        original=exc,
    )


class AdaptiveLLMRouter:
    def __init__(
        self,
        clients: list[tuple[str, LLMClient]],
        health: ProviderHealthRegistry,
        rate_limiters: dict[str, SlidingWindowRateLimiter] | None = None,
        max_retries: int = 3,
        backoff_min: float = 1.0,
        backoff_max: float = 30.0,
        backoff_multiplier: float = 2.0,
        jitter_factor: float = 0.1,
        load_balance_strategy: str = "least_used",
        quota_near_limit_threshold: float = 0.1,
    ) -> None:
        self._clients = clients
        self._client_map: dict[str, LLMClient] = {name: client for name, client in clients}
        self._health = health
        self._rate_limiters = rate_limiters or {}
        self._max_retries = max_retries
        self._backoff_min = backoff_min
        self._backoff_max = backoff_max
        self._backoff_multiplier = backoff_multiplier
        self._jitter_factor = jitter_factor
        self._load_balance_strategy = load_balance_strategy
        self._quota_near_limit_threshold = quota_near_limit_threshold
        self._last_usage = LLMUsage()

    @property
    def model(self) -> str:
        return self._clients[0][1].model if self._clients else ""

    @property
    def last_usage(self) -> LLMUsage:
        return self._last_usage

    def _get_external_providers(self) -> list[str]:
        return [name for name, _ in self._clients if name.lower() in EXTERNAL_PROVIDERS]

    def _get_ollama_clients(self) -> list[tuple[str, LLMClient]]:
        return [(n, c) for n, c in self._clients if n.lower() == "ollama"]

    def _provider_is_near_quota(self, provider: str) -> bool:
        rl = self._rate_limiters.get(provider)
        if rl is None:
            return False
        threshold = self._quota_near_limit_threshold
        rpd_remaining = rl.remaining_rpd
        rpd_limit = rl.stats.get("rpd_limit", 1000)
        if rpd_limit > 0 and rpd_remaining / rpd_limit < threshold:
            return True
        rpm_remaining = rl.remaining_rpm
        rpm_limit = rl.stats.get("rpm_limit", 60)
        return bool(rpm_limit > 0 and rpm_remaining / rpm_limit < threshold)

    async def _select_best_model(
        self,
        exclude: set[str] | None = None,
        prefer_external: bool = True,
    ) -> tuple[str, str, LLMClient] | None:
        exclude = exclude or set()
        candidates = (
            self._get_external_providers() if prefer_external else [n for n, _ in self._clients if n not in exclude]
        )
        candidates = [p for p in candidates if p not in exclude]

        if not candidates:
            return None

        healthy = [p for p in candidates if self._health.provider_is_healthy(p)]
        if not healthy:
            return None

        not_near_quota = [p for p in healthy if not self._provider_is_near_quota(p)]
        pool = not_near_quota or healthy

        if self._load_balance_strategy == "least_used":
            provider = self._health.get_least_recently_selected(pool)
        else:
            provider = pool[0]
        if provider is None:
            return None

        models = self._health.get_healthy_models(provider)
        for model_name, _score in models:
            for p_name, client in self._clients:
                if p_name == provider and client.model == model_name:
                    self._health.mark_selected(provider)
                    logger.info(
                        "Selected provider=%s model=%s score=%.3f",
                        provider,
                        model_name,
                        _score,
                    )
                    return provider, model_name, client

            for p_name, client in self._clients:
                if p_name == provider:
                    self._health.mark_selected(provider)
                    logger.info(
                        "Selected provider=%s model=%s (fallback to client model)",
                        provider,
                        client.model,
                    )
                    return provider, client.model, client
        return None

    async def _try_model(
        self,
        client: LLMClient,
        provider: str,
        model: str,
        prompt: str,
        temperature: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        last_error: ProviderError | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                start = time_module.monotonic()
                text = await client.generate(
                    prompt=prompt,
                    temperature=temperature,
                    num_predict=num_predict,
                    num_ctx=num_ctx,
                )
                latency = time_module.monotonic() - start
                self._health.track_success(provider, client.model, latency)
                self._last_usage = client.last_usage
                return text
            except Exception as exc:
                p_err = _categorize_llm_error(exc, provider, client.model)
                last_error = p_err
                self._health.track_failure(provider, client.model, p_err.category, p_err.retry_after)

                is_transient = p_err.category in (
                    ProviderErrorCategory.RETRYABLE,
                    ProviderErrorCategory.RATE_LIMITED,
                )
                if attempt >= self._max_retries or not is_transient:
                    self._health.mark_model_cooldown(provider, client.model)
                    logger.warning(
                        "Model failed provider=%s model=%s attempt=%d/%d category=%s",
                        provider,
                        client.model,
                        attempt,
                        self._max_retries,
                        p_err.category.value,
                    )
                    raise p_err from None

                if p_err.category == ProviderErrorCategory.RATE_LIMITED:
                    other_healthy = [
                        p
                        for p in self._get_external_providers()
                        if p != provider and self._health.provider_is_healthy(p)
                    ]
                    if other_healthy:
                        logger.info(
                            "Rate limited on %s/%s, failover to another provider available",
                            provider,
                            client.model,
                        )
                        raise p_err from None

                    wait = p_err.retry_after or 5.0
                    logger.info(
                        "Rate limited on %s/%s, waiting %.1fs before retry %d/%d",
                        provider,
                        client.model,
                        wait,
                        attempt,
                        self._max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue

                backoff = min(
                    self._backoff_min * (self._backoff_multiplier ** (attempt - 1)),
                    self._backoff_max,
                )
                jitter = random.uniform(0, backoff * self._jitter_factor)
                total_wait = backoff + jitter
                logger.info(
                    "Retrying provider=%s model=%s attempt=%d/%d wait=%.1fs category=%s",
                    provider,
                    client.model,
                    attempt,
                    self._max_retries,
                    total_wait,
                    p_err.category.value,
                )
                await asyncio.sleep(total_wait)

        raise LLMClientError(
            f"Provider {provider} model {client.model} failed after {self._max_retries} attempts. Last error: {last_error}"
        ) from last_error

    async def _try_provider(
        self,
        provider_name: str,
        prompt: str,
        temperature: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str | None:
        client = self._client_map.get(provider_name)
        if client is None:
            return None

        models = self._health.get_healthy_models(provider_name)
        tried_models = set()

        for model_name, _score in models:
            if model_name in tried_models:
                continue
            tried_models.add(model_name)
            try:
                mock_client = self._client_map.get(provider_name)
                if mock_client is None:
                    continue
                return await self._try_model(
                    mock_client,
                    provider_name,
                    model_name,
                    prompt,
                    temperature,
                    num_predict,
                    num_ctx,
                )
            except ProviderError as p_err:
                if p_err.category in (
                    ProviderErrorCategory.AUTHENTICATION_ERROR,
                    ProviderErrorCategory.INVALID_REQUEST,
                    ProviderErrorCategory.PERMANENT_ERROR,
                    ProviderErrorCategory.QUOTA_EXCEEDED,
                ):
                    raise
                continue

        return None

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        attempted_providers: set[str] = set()

        while True:
            selected = await self._select_best_model(
                exclude=attempted_providers,
                prefer_external=True,
            )

            if selected is not None:
                provider_name, model_name, client = selected
                logger.info(
                    "Routing request to provider=%s model=%s",
                    provider_name,
                    model_name,
                )
                try:
                    return await self._try_model(
                        client,
                        provider_name,
                        model_name,
                        prompt,
                        temperature,
                        num_predict,
                        num_ctx,
                    )
                except ProviderError as p_err:
                    attempted_providers.add(provider_name)
                    self._health.mark_provider_cooldown(provider_name)
                    logger.warning(
                        "Provider %s failed with %s, marking cooldown and trying next",
                        provider_name,
                        p_err.category.value,
                    )
                    if p_err.category in (
                        ProviderErrorCategory.AUTHENTICATION_ERROR,
                        ProviderErrorCategory.INVALID_REQUEST,
                        ProviderErrorCategory.PERMANENT_ERROR,
                        ProviderErrorCategory.QUOTA_EXCEEDED,
                    ):
                        remaining = [
                            p
                            for p in self._get_external_providers()
                            if p not in attempted_providers and self._health.provider_is_healthy(p)
                        ]
                        if not remaining:
                            break
                    continue

            external_remaining = [p for p in self._get_external_providers() if p not in attempted_providers]
            if not external_remaining:
                ollama_clients = self._get_ollama_clients()
                if ollama_clients:
                    logger.warning("All external providers exhausted, falling back to Ollama")
                    name, client = ollama_clients[0]
                    try:
                        return await self._try_model(
                            client,
                            name,
                            client.model,
                            prompt,
                            temperature,
                            num_predict,
                            num_ctx,
                        )
                    except ProviderError as p_err:
                        raise LLMClientError(f"All providers (including Ollama) failed. Last error: {p_err}") from p_err
                break

            await asyncio.sleep(0.5)

        raise LLMClientError("All LLM providers in adaptive fallback chain failed.")

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        selected = await self._select_best_model(prefer_external=True)
        if selected is not None:
            provider_name, _model_name, client = selected
            try:
                async for token in client.generate_stream(prompt=prompt, temperature=temperature):
                    yield token
                return
            except Exception as exc:
                p_err = _categorize_llm_error(exc, provider_name, client.model)
                logger.warning(
                    "Streaming failed on %s/%s: %s, trying fallback",
                    provider_name,
                    client.model,
                    p_err.category.value,
                )

        for name, client in self._clients:
            if name == selected[0] if selected else False:
                continue
            try:
                async for token in client.generate_stream(prompt=prompt, temperature=temperature):
                    yield token
                return
            except Exception as exc:
                p_err = _categorize_llm_error(exc, name, client.model)
                logger.warning(
                    "Fallback stream %s failed: %s",
                    name,
                    p_err.category.value,
                )

        logger.warning("All streaming providers failed, falling back to generate()")
        result = await self.generate(prompt=prompt, temperature=temperature)
        yield result

    async def close(self) -> None:
        for _, client in self._clients:
            await client.close()
