"""Async Ollama generation client using httpx.AsyncClient.

Provides the same interface as OllamaClient but with native async/await support,
eliminating the need for ThreadPoolExecutor offloading for LLM generation calls.
"""

from __future__ import annotations

import logging
import re

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class AsyncOllamaError(RuntimeError):
    """Raised when async Ollama cannot return an answer."""


class AsyncOllamaClient:
    """Async Ollama generation client using httpx.AsyncClient."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        num_ctx: int,
        num_predict: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def generate(self, prompt: str, num_predict: int | None = None, num_ctx: int | None = None) -> str:
        if num_predict is None:
            num_predict = self.num_predict
        if num_ctx is None:
            num_ctx = self.num_ctx
        logger.info(
            "Async Ollama generation started model=%s prompt_chars=%s num_ctx=%s num_predict=%s",
            self.model,
            len(prompt),
            num_ctx,
            num_predict,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "top_p": 0.8,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        try:
            body = await self._http_post(payload)
        except httpx.TimeoutException as exc:
            logger.exception("Async Ollama generation timed out timeout_seconds=%s", self.timeout_seconds)
            raise AsyncOllamaError(
                f"Ollama timed out after {self.timeout_seconds} seconds. "
                "Try again after Ollama finishes loading the model, or reduce the configured context/output limits."
            ) from exc
        except (httpx.ConnectError, httpx.HTTPError) as exc:
            logger.exception("Async Ollama connection failed base_url=%s", self.base_url)
            raise AsyncOllamaError(f"Could not reach Ollama. Start Ollama and run: ollama pull {self.model}") from exc

        response = self._extract_final_response(str(body.get("response", "")))
        done_reason = body.get("done_reason", "unknown")
        logger.info(
            "Async Ollama generation completed done_reason=%s response_chars=%s final_chars=%s",
            done_reason,
            len(str(body.get("response", ""))),
            len(response),
        )
        if not response:
            logger.warning("Async Ollama returned no final answer done_reason=%s", done_reason)
            raise AsyncOllamaError(
                "Ollama returned no final answer. "
                f"Generation stopped with reason `{done_reason}`. "
                "The model likely spent its output budget on reasoning. "
                "Try again, or increase `ollama_num_predict` in settings.py."
            )
        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, OSError)),
        reraise=True,
    )
    async def _http_post(self, payload: dict) -> dict:
        response = await self._client.post("/api/generate", json=payload)
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
        """Close the httpx client."""
        await self._client.aclose()
