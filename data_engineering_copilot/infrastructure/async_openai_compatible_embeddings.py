"""Async OpenAI-compatible embedding provider using httpx.AsyncClient.

Provides an EmbedderProtocol-compatible interface for any OpenAI-compatible
/v1/embeddings endpoint. Used by OpenRouter, NVIDIA NIM, Gemini, and others.
"""

from __future__ import annotations

import logging

import httpx
import tiktoken
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base

from data_engineering_copilot.domain.exceptions import EmbeddingError
from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.async_client import SafeAsyncClientMixin
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

# Retryable network errors — these should propagate to the @retry decorator
_RETRYABLE_ERRORS = (httpx.TimeoutException, httpx.ConnectError, OSError)

# Fallback token encoder (matches common OpenAI-compatible tokenizer ratio)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_SAFE_TOKENS = 3800  # Safe buffer below OpenRouter's 4096 model limit


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


class OpenAICompatibleEmbeddings(SafeAsyncClientMixin):
    """Async embedding provider for any OpenAI-compatible /v1/embeddings endpoint.

    Works with OpenRouter, NVIDIA NIM, Gemini, and other OpenAI-compatible APIs.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "nvidia/nemotron-3-embed-1b",
        base_url: str = "https://openrouter.ai/api/v1",
        embedding_dimension: int = 2048,
        batch_size: int = 32,
        timeout_seconds: int = 120,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        include_provider_param: bool = True,
        retry_wait: wait_base | None = None,
        max_tokens_per_input: int = MAX_SAFE_TOKENS,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._embedding_dimension = embedding_dimension
        self._batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self._rate_limiter = rate_limiter
        self._include_provider_param = include_provider_param
        self._max_tokens_per_input = max_tokens_per_input
        logger.info("Using embedding model %s at %s", model_name, self.base_url)
        self._request_embeddings = retry(
            stop=stop_after_attempt(5),
            wait=retry_wait or wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, OSError, httpx.HTTPStatusError)),
            reraise=True,
        )(self._request_embeddings)

    def _make_client_kwargs(self) -> dict:
        return {
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
            }
        }

    async def _get_client(self) -> httpx.AsyncClient:
        return await self._get_safe_client()

    def _slice_texts_into_batches(self, texts: list[str], batch_size: int) -> list[list[str]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def _reject_over_budget(self, texts: list[str]) -> None:
        """Raise ``EmbeddingError`` when any input exceeds the token budget.

        Over-budget input is a caller bug (chunks must be split losslessly
        before embedding). Failing loudly beats silently losing content.
        """
        for text in texts:
            token_count = _count_tokens(text)
            if token_count > self._max_tokens_per_input:
                raise EmbeddingError(
                    f"Embedding input exceeds budget: model={self.model_name} "
                    f"provider={self.base_url} tokens={token_count} "
                    f"allowed={self._max_tokens_per_input}. "
                    "Split the text losslessly before embedding."
                )

    async def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Acquire rate limiter slot before making the request
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

        # Never truncate input to fit the provider limit — reject over-budget
        # text so silent content loss can never corrupt the index.
        self._reject_over_budget(texts)

        payload: dict = {
            "model": self.model_name,
            "input": texts,
        }
        if self._include_provider_param:
            payload["provider"] = {}

        response = await (await self._get_client()).post(
            "/embeddings",
            json=payload,
        )

        # Handle 429 rate limit — parse Retry-After and retry after backoff
        if response.status_code == 429:
            if self._rate_limiter is not None:
                await self._rate_limiter.handle_429(dict(response.headers))
            raise httpx.HTTPStatusError(  # tenacity will retry this
                "Rate limited by embedding provider", request=response.request, response=response
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(f"Failed to get embeddings: {exc}") from exc
        resp_data = response.json()

        # Check for provider error body (200 OK with embedded error)
        if isinstance(resp_data, dict) and "error" in resp_data:
            err_details = resp_data["error"]
            err_msg = err_details.get("message", str(resp_data)) if isinstance(err_details, dict) else str(err_details)
            err_code = err_details.get("code", "UNKNOWN") if isinstance(err_details, dict) else "UNKNOWN"
            raise EmbeddingError(f"Embedding API returned error [Code {err_code}]: {err_msg}")

        if "data" not in resp_data:
            raise EmbeddingError(
                f"Embeddings response missing 'data' key. "
                f"Response keys: {sorted(resp_data.keys()) if isinstance(resp_data, dict) else 'invalid'}. "
                f"Response: {str(resp_data)[:500]}"
            )

        data_list = resp_data["data"]
        if not isinstance(data_list, list):
            raise EmbeddingError(
                f"Embeddings response 'data' value is not a list. Got type {type(data_list).__name__}."
            )

        embeddings = [item["embedding"] for item in sorted(data_list, key=lambda x: x.get("index", 0))]

        if len(embeddings) != len(texts):
            raise EmbeddingError(f"Provider returned {len(embeddings)} embeddings for {len(texts)} input texts.")

        self._validate_embedding_dimensions(embeddings, texts)
        return embeddings

    async def _embed_with_batching(self, texts: list[str]) -> list[list[float]]:
        batches = self._slice_texts_into_batches(texts, self._batch_size)
        if len(batches) == 1:
            return await self._request_embeddings(texts)

        logger.info("Processing %d texts in %d batches (batch_size=%d)", len(texts), len(batches), self._batch_size)
        all_embeddings: list[list[float]] = []
        for batch_idx, batch_texts in enumerate(batches, start=1):
            logger.debug("Processing batch %d/%d with %d texts", batch_idx, len(batches), len(batch_texts))
            batch_embeddings = await self._request_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)

        logger.info("Successfully embedded all %d texts in %d batches", len(texts), len(batches))
        return all_embeddings

    def _validate_embedding_dimensions(self, embeddings: list[list[float]], texts: list[str]) -> None:
        expected_dim = self._embedding_dimension
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list):
                raise EmbeddingError(
                    f"Embedding {i} is not a list. Got type {type(emb).__name__}. Text: {texts[i][:100]!r}"
                )
            if len(emb) == 0:
                raise EmbeddingError(f"Embedding {i} is empty (dimension 0). Expected dimension {expected_dim}.")
            if len(emb) != expected_dim:
                raise EmbeddingError(f"Embedding {i} has dimension {len(emb)}, expected {expected_dim}.")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await self._embed_with_batching(texts)
        logger.info("Embedded texts count=%s", len(texts))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        if not results or results[0] is None:
            raise EmbeddingError(f"Embedding returned empty result for query: {text[:80]!r}")
        return results[0]

    # ProviderClient protocol method
    async def call(self, request: list[str]) -> list[list[float]]:
        """Unified call interface for ProviderFallbackChain."""
        return await self.embed_texts(request)

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def last_usage(self) -> LLMUsage:
        return LLMUsage()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
