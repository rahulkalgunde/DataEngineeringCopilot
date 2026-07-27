"""Async OpenRouter generation client using httpx.AsyncClient.

Provides an LLMProvider-compatible interface for OpenRouter's
OpenAI-compatible Chat Completions API at /api/v1/chat/completions.
"""

from __future__ import annotations

import logging
import re

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.async_client import SafeAsyncClientMixin
from data_engineering_copilot.infrastructure.rate_limiter import OpenRouterRateLimiter

logger = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot return an answer."""


class OpenRouterLLMClient(SafeAsyncClientMixin):
    """Async OpenRouter LLM client using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.05,
        rate_limiter: OpenRouterRateLimiter | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._usage = LLMUsage()
        self._rate_limiter = rate_limiter

    @property
    def last_usage(self) -> LLMUsage:
        return self._usage

    @property
    def _base_url(self) -> str:
        return self.base_url

    def _make_client_kwargs(self) -> dict:
        return {
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://data-engineering-copilot.local",
            }
        }

    async def _get_client(self) -> httpx.AsyncClient:
        return await self._get_safe_client()

    async def generate(self, prompt: str, temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self._temperature
        logger.info(
            "OpenRouter generation started model=%s prompt_chars=%s temperature=%.2f",
            self.model,
            len(prompt),
            temp,
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
        }

        try:
            body = await self._http_post(payload)
        except httpx.TimeoutException as exc:
            logger.exception("OpenRouter generation timed out timeout_seconds=%s", self.timeout_seconds)
            raise OpenRouterError(f"OpenRouter timed out after {self.timeout_seconds} seconds.") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.exception("OpenRouter rate limit persistently exceeded after retries.")
                raise OpenRouterError("OpenRouter rate limit exceeded after all retries. Try again later.") from exc
            logger.exception("OpenRouter HTTP error: %s", exc)
            raise OpenRouterError(f"OpenRouter returned HTTP {exc.response.status_code}.") from exc
        except (httpx.ConnectError, httpx.HTTPError) as exc:
            logger.exception("OpenRouter connection failed")
            raise OpenRouterError("Could not reach OpenRouter. Check your network and API key.") from exc

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
            "OpenRouter generation completed model=%s response_chars=%s final_chars=%s "
            "prompt_tokens=%d completion_tokens=%d",
            self.model,
            len(content),
            len(clean_text),
            prompt_tokens,
            completion_tokens,
        )

        return clean_text

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, OSError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def _http_post(self, payload: dict) -> dict:
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        response = await (await self._get_client()).post("/chat/completions", json=payload)
        if response.status_code == 429:
            if self._rate_limiter is not None:
                await self._rate_limiter.handle_429(dict(response.headers))
            raise httpx.HTTPStatusError("Rate limited by OpenRouter", request=response.request, response=response)
        if response.status_code == 401:
            raise OpenRouterError("OpenRouter returned 401 Unauthorized. Check your API key.")
        response.raise_for_status()
        return response.json()

    def _extract_final_response(self, response: str) -> str:
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
