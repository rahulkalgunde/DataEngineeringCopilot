"""Async OpenRouter embedding provider using httpx.AsyncClient.

Provides an EmbedderProtocol-compatible interface for OpenRouter's
embedding API at /api/v1/embeddings. Supports models like
nvidia/nemotron-3-embed-1b:free.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import tiktoken
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_engineering_copilot.domain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Fallback token encoder (matches OpenAI/OpenRouter common tokenizer tokenization ratio)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_SAFE_TOKENS = 3800  # Safe buffer below OpenRouter's 4096 model limit


def _truncate_to_safe_tokens(text: str, max_tokens: int = MAX_SAFE_TOKENS) -> str:
    """Truncates text to stay safely under OpenRouter's token limit."""
    tokens = _TOKENIZER.encode(text)
    if len(tokens) > max_tokens:
        logger.warning(
            "Text length (%d tokens) exceeds max limit (%d). Truncating before sending to OpenRouter.",
            len(tokens),
            max_tokens,
        )
        return _TOKENIZER.decode(tokens[:max_tokens])
    return text


class OpenRouterEmbeddings:
    """Async OpenRouter embedding provider using the /api/v1/embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "nvidia/nemotron-3-embed-1b:free",
        base_url: str = "https://openrouter.ai/api/v1",
        embedding_dimension: int = 2048,
        batch_size: int = 32,
        timeout_seconds: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._embedding_dimension = embedding_dimension
        self._batch_size = batch_size
        self._timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._loop_id: int | None = None
        logger.info("Using OpenRouter embedding model %s at %s", model_name, self._base_url)

    def _get_client(self) -> httpx.AsyncClient:
        current_loop = id(asyncio.get_running_loop())
        if self._client is not None and self._loop_id != current_loop:
            self._client = None
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://data-engineering-copilot.local",
                },
            )
            self._loop_id = current_loop
        return self._client

    def _slice_texts_into_batches(self, texts: list[str], batch_size: int) -> list[list[str]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, OSError)),
        reraise=True,
    )
    async def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Pre-emptively truncate all texts to ensure no text exceeds 3800 tokens
        safe_texts = [_truncate_to_safe_tokens(t) for t in texts]

        try:
            response = await self._get_client().post(
                "/embeddings",
                json={
                    "model": self.model_name,
                    "input": safe_texts,
                    "provider": {"truncate": "END"},  # Instruct OpenRouter upstream to truncate if still oversized
                },
            )
            response.raise_for_status()
            resp_data = response.json()
        except Exception as exc:
            raise EmbeddingError(f"Failed to get embeddings from OpenRouter: {exc}") from exc

        # FIX: Check for OpenRouter 200 OK Embedded Error Body
        if isinstance(resp_data, dict) and "error" in resp_data:
            err_details = resp_data["error"]
            err_msg = err_details.get("message", str(resp_data)) if isinstance(err_details, dict) else str(err_details)
            err_code = err_details.get("code", "UNKNOWN") if isinstance(err_details, dict) else "UNKNOWN"
            raise EmbeddingError(f"OpenRouter API returned error [Code {err_code}]: {err_msg}")

        if "data" not in resp_data:
            raise EmbeddingError(
                f"OpenRouter embeddings response missing 'data' key. "
                f"Response keys: {sorted(resp_data.keys()) if isinstance(resp_data, dict) else 'invalid'}. "
                f"Response: {str(resp_data)[:500]}"
            )

        data_list = resp_data["data"]
        if not isinstance(data_list, list):
            raise EmbeddingError(f"OpenRouter 'data' value is not a list. Got type {type(data_list).__name__}.")

        embeddings = [item["embedding"] for item in sorted(data_list, key=lambda x: x.get("index", 0))]

        if len(embeddings) != len(safe_texts):
            raise EmbeddingError(f"OpenRouter returned {len(embeddings)} embeddings for {len(safe_texts)} input texts.")

        self._validate_embedding_dimensions(embeddings, safe_texts)
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

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
