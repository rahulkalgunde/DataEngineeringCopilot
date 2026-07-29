"""Unified async LLM client for any OpenAI-compatible provider.

Supports Ollama, OpenRouter, NVIDIA NIM, OpenAI, etc. via a single
parametrized class. All providers use the ``/v1/chat/completions``
endpoint; differences are captured in constructor parameters.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.async_client import SafeAsyncClientMixin
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the LLM provider cannot return an answer."""


class LLMClient(SafeAsyncClientMixin):
    """Unified async client for any OpenAI-compatible Chat Completions API.

    Differences between providers are handled purely through constructor
    parameters — no subclassing required.

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

        try:
            body = await self._http_post(payload)
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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_llm_error),  # type: ignore[arg-type]
        reraise=True,
    )
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
