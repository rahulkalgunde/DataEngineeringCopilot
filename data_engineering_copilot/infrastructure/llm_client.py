"""Unified async LLM client for any OpenAI-compatible provider.

Supports Ollama, OpenRouter, NVIDIA NIM, OpenAI, etc. via a single
parametrized class. All providers use the ``/v1/chat/completions``
endpoint; differences are captured in constructor parameters.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator

import httpx

from data_engineering_copilot.domain.exceptions import CoreDomainException, ProviderErrorCategory
from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.async_client import SafeAsyncClientMixin
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

# Separator between the system-instruction block and the user-facing data in a
# compiled prompt. Prompt templates (see ``langfuse_prompts.SEED_PROMPTS`` and
# each service's fallback template) embed this literal so ``generate()`` can
# split a single compiled string into a proper ``system`` + ``user`` message
# pair. Providers follow system instructions more reliably when they arrive in
# the ``system`` role instead of being buried in a ``user`` message.
#
# Prompts that do NOT contain the separator are sent unchanged as a single
# ``user`` message (backwards compatible with already-seeded Langfuse prompts).
SYSTEM_BLOCK_SEPARATOR = "\n\n<<<SYSTEM_BLOCK_END>>>\n\n"


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    """Build chat ``messages`` from a compiled prompt.

    When *prompt* embeds :data:`SYSTEM_BLOCK_SEPARATOR`, the text before it
    becomes the ``system`` message and the text after it the ``user`` message.
    Otherwise (and whenever either half is empty) the whole prompt is sent as a
    single ``user`` message so behavior is identical to the previous format.
    """
    if SYSTEM_BLOCK_SEPARATOR in prompt:
        system_part, user_part = prompt.split(SYSTEM_BLOCK_SEPARATOR, 1)
        if system_part.strip() and user_part.strip():
            return [
                {"role": "system", "content": system_part.strip()},
                {"role": "user", "content": user_part.strip()},
            ]
    return [{"role": "user", "content": prompt}]


class LLMClientError(CoreDomainException):
    """Raised when the LLM provider cannot return an answer.

    Carries optional structured metadata so the adaptive router can make
    failover decisions without relying on message matching.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        category: ProviderErrorCategory | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.category = category


class LLMClient(SafeAsyncClientMixin):
    """Unified async client for any OpenAI-compatible Chat Completions API.

    Differences between providers are handled purely through constructor
    parameters — no subclassing required.

    This client performs a single HTTP attempt per call: no retry loop and
    no circuit breaker. Provider-level resilience is owned by the adaptive
    router (fail fast, fail over to the next available provider). The rate
    limiter is used as a non-blocking pre-flight gate so an over-limit
    provider is never called.

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
    endpoint_path:
        API endpoint path (default ``"/chat/completions"``).
    max_tokens:
        Max output tokens sent with every request. ``None`` omits the field
        (provider default applies — avoid: Cloudflare truncates at 256, NVIDIA
        at 1024). Use 0 only where a provider treats 0 as "no limit".
    max_tokens_field:
        Body field name for the output cap — ``"max_tokens"`` (NVIDIA,
        Cloudflare, Ollama, OpenRouter, Gemini) or ``"max_completion_tokens"``
        (Groq, Cerebras; NVIDIA's schema rejects unknown fields with 422).
    extra_body:
        Additional fields merged into the request body (e.g.
        ``{"options": {"num_ctx": 4096}}`` for Ollama-specific options).
    extra_headers:
        Additional HTTP headers sent with every request (e.g.
        ``{"HTTP-Referer": "https://..."}`` for OpenRouter).
    rate_limiter:
        Optional shared rate limiter for providers that enforce RPM/RPD
        limits (OpenRouter, NVIDIA NIM, etc.). When set, a slot is acquired
        non-blocking before each request; exhaustion raises a 429-class
        ``LLMClientError`` instead of blocking.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: int = 120,
        temperature: float = 0.05,
        endpoint_path: str = "/chat/completions",
        max_tokens: int | None = None,
        max_tokens_field: str = "max_tokens",
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        keep_alive: str | int | None = None,
        connect_timeout_seconds: int | float | None = None,
        pool_timeout_seconds: int | float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._endpoint_path = endpoint_path
        self._max_tokens = max_tokens
        self._max_tokens_field = max_tokens_field
        self._extra_body = extra_body or {}
        self._extra_headers = extra_headers or {}
        self._keep_alive = keep_alive
        self._usage = LLMUsage()
        self._rate_limiter = rate_limiter
        self.connect_timeout_seconds = connect_timeout_seconds
        self.pool_timeout_seconds = pool_timeout_seconds

    @property
    def last_usage(self) -> LLMUsage:
        return self._usage

    # ProviderClient protocol method
    async def call(self, request: str) -> str:
        """Unified call interface for ProviderFallbackChain."""
        return await self.generate(prompt=request)

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
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self._temperature
        if max_tokens is None:
            max_tokens = self._max_tokens
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        logger.info(
            "LLM generation started request_id=%s model=%s prompt_chars=%s temperature=%.2f max_tokens=%s",
            request_id,
            self.model,
            len(prompt),
            temp,
            max_tokens,
        )

        payload: dict = {
            "model": self.model,
            "messages": build_chat_messages(prompt),
            "temperature": temp,
        }
        if max_tokens is not None and max_tokens > 0:
            payload[self._max_tokens_field] = max_tokens
        if self._extra_body:
            payload.update(self._extra_body)
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive

        # Single attempt: no retry loop, no circuit breaker. Failures carry
        # structured status_code / retry_after so the router can fail over fast.
        try:
            body = await self._http_post(payload)
        except TimeoutError as exc:
            logger.warning(
                "LLM generation timed out request_id=%s model=%s timeout=%ss duration_ms=%.0f",
                request_id,
                self.model,
                self.timeout_seconds,
                (time.perf_counter() - started) * 1000,
            )
            raise LLMClientError(f"LLM provider timed out after {self.timeout_seconds} seconds.") from exc
        except httpx.TimeoutException as exc:
            logger.exception(
                "LLM generation timed out request_id=%s timeout_seconds=%s duration_ms=%.0f",
                request_id,
                self.timeout_seconds,
                (time.perf_counter() - started) * 1000,
            )
            raise LLMClientError(f"LLM provider timed out after {self.timeout_seconds} seconds.") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            retry_after = None
            if self._rate_limiter is not None:
                retry_after = self._rate_limiter.parse_retry_after(dict(exc.response.headers))
            logger.warning(
                "LLM provider HTTP error model=%s status=%s retry_after=%s",
                self.model,
                status,
                retry_after,
            )
            raise LLMClientError(
                f"LLM provider returned HTTP {status}.",
                status_code=status,
                retry_after=retry_after,
            ) from exc
        except (httpx.HTTPError, OSError) as exc:
            logger.exception("LLM provider connection failed")
            raise LLMClientError("Could not reach LLM provider. Check your network and API key.") from exc

        # Some providers return message.content: null (or omit usage) on an
        # otherwise 200 response; treat that as empty so downstream parsing
        # (e.g. _extract_final_response -> .strip()) never sees None.
        content = (body.get("choices", [{}])[0].get("message", {}).get("content", "")) or ""
        usage_data = body.get("usage") or {}

        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)

        self._usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=body.get("model", self.model),
        )

        clean_text = self._extract_final_response(content)

        logger.info(
            "LLM generation completed request_id=%s model=%s response_chars=%s final_chars=%s prompt_tokens=%d completion_tokens=%d duration_ms=%.0f",
            request_id,
            self.model,
            len(content),
            len(clean_text),
            prompt_tokens,
            completion_tokens,
            (time.perf_counter() - started) * 1000,
        )

        return clean_text

    async def _http_post(self, payload: dict) -> dict:
        if self._rate_limiter is not None and not await self._rate_limiter.try_acquire():
            raise LLMClientError(
                "Rate limit window exhausted; provider not called.",
                status_code=429,
                retry_after=self._rate_limiter.wait_until_available(),
            )
        response = await (await self._get_client()).post(self._endpoint_path, json=payload)
        if response.status_code == 429:
            if self._rate_limiter is not None:
                retry_after = self._rate_limiter.parse_retry_after(dict(response.headers))
                logger.warning(
                    "rate_limit_captured model=%s retry_after=%s",
                    self.model,
                    retry_after,
                )
            raise httpx.HTTPStatusError("Rate limited by provider", request=response.request, response=response)
        if response.status_code == 401:
            raise LLMClientError("LLM provider returned 401 Unauthorized. Check your API key.", status_code=401)
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
            self._client_loop = None

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
            "messages": build_chat_messages(prompt),
            "temperature": temp,
            "stream": True,
        }
        if self._max_tokens is not None and self._max_tokens > 0:
            payload[self._max_tokens_field] = self._max_tokens
        if self._extra_body:
            payload.update(self._extra_body)

        try:
            if self._rate_limiter is not None and not await self._rate_limiter.try_acquire():
                raise LLMClientError(
                    "Rate limit window exhausted; provider not called.",
                    status_code=429,
                    retry_after=self._rate_limiter.wait_until_available(),
                )
            client = await self._get_client()
            async with client.stream("POST", self._endpoint_path, json=payload) as response:
                if response.status_code == 429:
                    if self._rate_limiter is not None:
                        self._rate_limiter.parse_retry_after(dict(response.headers))
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
