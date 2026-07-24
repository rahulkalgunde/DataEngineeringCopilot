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

logger = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot return an answer."""


class OpenRouterLLMClient:
    """Async OpenRouter LLM client using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.05,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._client: httpx.AsyncClient | None = None
        self._loop_id: int | None = None
        self._usage = LLMUsage()

    @property
    def last_usage(self) -> LLMUsage:
        return self._usage

    def _get_client(self) -> httpx.AsyncClient:
        import asyncio

        current_loop = id(asyncio.get_running_loop())
        if self._client is not None and self._loop_id != current_loop:
            self._client = None
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://data-engineering-copilot.local",
                },
            )
            self._loop_id = current_loop
        return self._client

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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, OSError)),
        reraise=True,
    )
    async def _http_post(self, payload: dict) -> dict:
        response = await self._get_client().post("/chat/completions", json=payload)
        if response.status_code == 401:
            raise OpenRouterError("OpenRouter returned 401 Unauthorized. Check your API key.")
        if response.status_code == 429:
            raise OpenRouterError("OpenRouter rate limit exceeded (429). Try again later.")
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
