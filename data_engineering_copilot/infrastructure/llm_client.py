"""Unified async LLM client for any OpenAI-compatible provider.

Supports Ollama, OpenRouter, NVIDIA NIM, OpenAI, etc. via a single
parametrized class. All providers use the ``/v1/chat/completions``
endpoint; differences are captured in constructor parameters.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.async_client import SafeAsyncClientMixin
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)


def _make_http_post_retry(wait: wait_base | None = None):
    """Build the tenacity retry decorator for LLM HTTP posts.

    ``wait`` defaults to the production exponential backoff; tests pass a zero
    wait so retry assertions don't sleep for real backoff durations.
    """
    return retry(
        stop=stop_after_attempt(5),
        wait=wait or wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(LLMClient._is_retryable_llm_error),  # type: ignore[arg-type]
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )


class CircuitBreakerError(RuntimeError):
    """Raised when the circuit breaker is open and the request is rejected."""


class CircuitBreaker:
    """Fail-fast circuit breaker for LLM provider calls.

    After ``failure_threshold`` consecutive failures the circuit opens
    and all subsequent calls are rejected immediately (without waiting
    for a timeout) for ``recovery_timeout`` seconds. After that period,
    a single test request is allowed (half-open). If it succeeds the
    circuit closes; if it fails the circuit re-opens.
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0, call_timeout: float = 60.0) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._call_timeout = call_timeout
        self._failures = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()
        self._state = "closed"  # closed | open | half-open

    async def call(self, coro_factory):
        async with self._lock:
            if self._state == "open":
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = "half-open"
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker open for {self._recovery_timeout:.0f}s "
                        f"after {self._failure_threshold} failures"
                    )
        try:
            result = await asyncio.wait_for(coro_factory(), timeout=self._call_timeout)
        except TimeoutError as exc:
            async with self._lock:
                self._failures += 1
                self._last_failure_time = time.monotonic()
                if self._state == "half-open" or self._failures >= self._failure_threshold:
                    self._state = "open"
            raise exc
        except Exception as exc:
            async with self._lock:
                self._failures += 1
                self._last_failure_time = time.monotonic()
                if self._state == "half-open" or self._failures >= self._failure_threshold:
                    self._state = "open"
            raise exc
        async with self._lock:
            self._failures = 0
            self._state = "closed"
        return result


class LLMClientError(RuntimeError):
    """Raised when the LLM provider cannot return an answer."""


class LLMClient(SafeAsyncClientMixin):
    """Unified async client for any OpenAI-compatible Chat Completions API.

    Differences between providers are handled purely through constructor
    parameters — no subclassing required.

    Includes a fail-fast circuit breaker: after ``circuit_breaker_threshold``
    consecutive failures, subsequent requests are immediately rejected for
    ``circuit_breaker_timeout`` seconds instead of waiting for the provider
    timeout.

    Parameters
    ----------
    base_url:
        Base URL of the provider API (e.g. ``"http://localhost:11434/v1"``
        for Ollama or ``"https://openrouter.ai/api/v1"`` for OpenRouter).
    model:
        Model name to use for generation.
    api_key:
        API key for authentication. Empty string for providers that do not
        require auth (e.g. local Ollama).
    timeout_seconds:
        HTTP request timeout.
    temperature:
        Sampling temperature.
    max_retries:
        Number of retry attempts for transient failures (timeout, connect,
        5xx). 429s are retried up to this limit as well.
    endpoint_path:
        API endpoint path (default ``"/chat/completions"``).
    extra_body:
        Additional fields merged into the request body (e.g.
        ``{"options": {"num_ctx": 4096}}`` for Ollama-specific options).
    extra_headers:
        Additional HTTP headers sent with every request (e.g.
        ``{"HTTP-Referer": "https://..."}`` for OpenRouter).
    rate_limiter:
        Optional shared rate limiter for providers that enforce RPM/RPD
        limits (OpenRouter, NVIDIA NIM, etc.).
    circuit_breaker_threshold:
        Consecutive failures before circuit opens (default 3).
    circuit_breaker_timeout:
        Seconds to keep circuit open (default 30).
    retry_wait:
        Override for the tenacity retry wait strategy (defaults to exponential
        backoff of 1-10s). Used by tests to eliminate real backoff sleeps.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: int = 120,
        temperature: float = 0.05,
        max_retries: int = 3,
        endpoint_path: str = "/chat/completions",
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_timeout: float = 30.0,
        retry_wait: wait_base | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_retries = max_retries
        self._endpoint_path = endpoint_path
        self._extra_body = extra_body or {}
        self._extra_headers = extra_headers or {}
        self._usage = LLMUsage()
        self._rate_limiter = rate_limiter
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_timeout,
            call_timeout=timeout_seconds,
        )
        self._http_post = _make_http_post_retry(retry_wait)(self._http_post)

    @property
    def last_usage(self) -> LLMUsage:
        return self._usage

    def _make_client_kwargs(self) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._extra_headers:
            headers.update(self._extra_headers)
        return {"headers": headers} if headers else {}

    async def _get_client(self) -> httpx.AsyncClient:
        return await self._get_safe_client()

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self._temperature
        logger.info(
            "LLM generation started model=%s prompt_chars=%s temperature=%.2f",
            self.model,
            len(prompt),
            temp,
        )

        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
        }
        if self._extra_body:
            payload.update(self._extra_body)

        options: dict = {}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if options:
            payload.setdefault("options", {}).update(options)

        # Fast-fail circuit breaker: fail immediately if provider is unhealthy
        try:
            body = await self._circuit_breaker.call(lambda: self._http_post(payload))
        except CircuitBreakerError:
            logger.warning(
                "Circuit breaker open for model=%s, failing fast instead of waiting %ss",
                self.model,
                self.timeout_seconds,
            )
            raise LLMClientError(
                f"LLM provider {self.model} is temporarily unavailable (circuit breaker open after repeated failures)."
            ) from None
        except TimeoutError as exc:
            logger.warning(
                "LLM generation timed out (circuit breaker call timeout) model=%s timeout=%ss",
                self.model,
                self._circuit_breaker._call_timeout,
            )
            raise LLMClientError(
                f"LLM provider timed out after {self._circuit_breaker._call_timeout} seconds."
            ) from exc
        except httpx.TimeoutException as exc:
            logger.exception("LLM generation timed out timeout_seconds=%s", self.timeout_seconds)
            raise LLMClientError(f"LLM provider timed out after {self.timeout_seconds} seconds.") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.exception("Rate limit persistently exceeded after retries.")
                raise LLMClientError("Rate limit exceeded after all retries. Try again later.") from exc
            logger.exception("LLM provider HTTP error: %s", exc)
            raise LLMClientError(f"LLM provider returned HTTP {exc.response.status_code}.") from exc
        except (httpx.ConnectError, httpx.HTTPError) as exc:
            logger.exception("LLM provider connection failed")
            raise LLMClientError("Could not reach LLM provider. Check your network and API key.") from exc

        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage_data = body.get("usage", {})

        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)

        self._usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=body.get("model", self.model),
        )

        clean_text = self._extract_final_response(content)

        logger.info(
            "LLM generation completed model=%s response_chars=%s final_chars=%s prompt_tokens=%d completion_tokens=%d",
            self.model,
            len(content),
            len(clean_text),
            prompt_tokens,
            completion_tokens,
        )

        return clean_text

    @staticmethod
    def _is_retryable_llm_error(exc: BaseException) -> bool:
        """Retry on transient errors only. 429 is handled by rate limiter, not retried."""
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, OSError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        return False

    async def _http_post(self, payload: dict) -> dict:
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        response = await (await self._get_client()).post(self._endpoint_path, json=payload)
        if response.status_code == 429:
            if self._rate_limiter is not None:
                await self._rate_limiter.handle_429(dict(response.headers))
            raise httpx.HTTPStatusError("Rate limited by provider", request=response.request, response=response)
        if response.status_code == 401:
            raise LLMClientError("LLM provider returned 401 Unauthorized. Check your API key.")
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_final_response(response: str) -> str:
        response = response.strip()
        if not response:
            return ""
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
        if response.lower().startswith("<think>"):
            return ""
        return response

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM via SSE chunks.

        Yields individual token strings as they arrive from the provider.
        Falls back to non-streaming ``generate()`` if streaming fails.
        """
        temp = temperature if temperature is not None else self._temperature
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "stream": True,
        }
        if self._extra_body:
            payload.update(self._extra_body)

        try:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()
            client = await self._get_client()
            async with client.stream("POST", self._endpoint_path, json=payload) as response:
                if response.status_code == 429:
                    if self._rate_limiter is not None:
                        await self._rate_limiter.handle_429(dict(response.headers))
                    raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError):
            logger.warning("Streaming failed, falling back to non-streaming generate()")
            result = await self.generate(prompt, temperature=temperature)
            yield result


class FallbackLLMClient:
    """Wrapper that tries multiple LLM providers in sequence on failure.

    Each provider is a separate ``LLMClient`` instance with its own circuit
    breaker and rate limiter.  On ``LLMClientError`` the next provider in the
    chain is tried.  If all fail the last error is re-raised.

    Shares the same public interface as ``LLMClient`` so consumers do not
    need to distinguish between them.
    """

    def __init__(self, clients: list[tuple[str, LLMClient]]) -> None:
        self._clients = clients
        self._last_usage = LLMUsage()

    @property
    def model(self) -> str:
        return self._clients[0][1].model if self._clients else ""

    @property
    def last_usage(self) -> LLMUsage:
        return self._last_usage

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        for name, client in self._clients:
            try:
                text = await client.generate(
                    prompt=prompt,
                    temperature=temperature,
                    num_predict=num_predict,
                    num_ctx=num_ctx,
                )
                self._last_usage = client.last_usage
                return text
            except LLMClientError as e:
                logger.warning("Fallback: %s failed (%s), trying next provider", name, e)
                last_error = e
        raise LLMClientError(f"All LLM providers in fallback chain failed. Last error: {last_error}") from last_error

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        for name, client in self._clients:
            try:
                async for token in client.generate_stream(prompt=prompt, temperature=temperature):
                    yield token
                return
            except (LLMClientError, Exception) as e:
                logger.warning("Fallback stream: %s failed (%s), trying next provider", name, e)
        result = await self._clients[-1][1].generate(prompt=prompt, temperature=temperature)
        yield result

    async def close(self) -> None:
        for _, client in self._clients:
            await client.close()
