"""Async Ollama generation client using httpx.AsyncClient.

Provides the same interface as OllamaClient but with native async/await support,
eliminating the need for ThreadPoolExecutor offloading for LLM generation calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_engineering_copilot.domain.models import LLMUsage

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Structured output from Ollama generation."""

    text: str
    prompt_eval_count: int = 0
    eval_count: int = 0
    duration_ms: int = 0
    tokens_per_second: float = 0.0
    model: str = ""


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
        self._client: httpx.AsyncClient | None = None
        self._loop_id: int | None = None
        self._usage: LLMUsage = LLMUsage()

    @property
    def last_usage(self) -> LLMUsage:
        return self._usage

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the httpx client on the current event loop.

        Re-creates the client if the event loop has changed (e.g. across
        pytest-asyncio test functions with function-scoped loop scope).
        """
        import asyncio

        current_loop = id(asyncio.get_running_loop())
        if self._client is not None and self._loop_id != current_loop:
            import warnings

            warnings.warn("Recreating httpx client for new event loop", stacklevel=2)
            self._client = None
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_seconds),
            )
            self._loop_id = current_loop
        return self._client

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

        # Track token usage
        prompt_eval_count = body.get("prompt_eval_count", 0)
        eval_count = body.get("eval_count", 0)
        total_duration_ns = body.get("total_duration", 0)
        duration_ms = int(total_duration_ns / 1_000_000) if total_duration_ns else 0
        tokens_per_second = (eval_count / (duration_ms / 1000)) if duration_ms > 0 else 0.0

        self._usage = LLMUsage(
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            model=self.model,
            duration_ms=duration_ms,
            tokens_per_second=round(tokens_per_second, 2),
        )

        logger.info(
            "Async Ollama generation completed done_reason=%s response_chars=%s final_chars=%s "
            "prompt_tokens=%d completion_tokens=%d duration_ms=%d tok/s=%.1f",
            done_reason,
            len(str(body.get("response", ""))),
            len(response),
            prompt_eval_count,
            eval_count,
            duration_ms,
            tokens_per_second,
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
        response = await self._get_client().post("/api/generate", json=payload)
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
        """Close the httpx client if it was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
