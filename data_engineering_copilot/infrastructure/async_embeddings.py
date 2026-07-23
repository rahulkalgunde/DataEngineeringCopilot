"""Async Ollama embedding provider using httpx.AsyncClient.

Provides the same interface as OllamaEmbeddings but with native async/await support,
eliminating the need for ThreadPoolExecutor offloading for embedding API calls.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class AsyncOllamaEmbeddings:
    """Async Ollama embedding provider using the /api/embed endpoint with httpx."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.ollama_base_url = settings.ollama_base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._loop_id: int | None = None
        logger.info("Using async Ollama embedding model %s at %s", model_name, self.ollama_base_url)

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
                base_url=self.ollama_base_url,
                timeout=httpx.Timeout(settings.ollama_timeout_seconds),
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
    async def _aollama_embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed for a single batch asynchronously."""
        try:
            response = await self._get_client().post(
                "/api/embed",
                json={"model": self.model_name, "input": texts},
            )
            response.raise_for_status()
            resp_data = response.json()
        except Exception as exc:
            raise EmbeddingError(f"Failed to get embeddings from Ollama: {exc}") from exc

        if "embeddings" not in resp_data:
            raise EmbeddingError(
                f"Ollama embeddings response missing 'embeddings' key. "
                f"Response keys: {sorted(resp_data.keys()) if isinstance(resp_data, dict) else 'invalid'}. "
                f"Response: {str(resp_data)[:500]}"
            )

        embeddings = resp_data["embeddings"]
        if not isinstance(embeddings, list):
            raise EmbeddingError(f"Ollama 'embeddings' value is not a list. Got type {type(embeddings).__name__}.")

        if len(embeddings) != len(texts):
            raise EmbeddingError(f"Ollama returned {len(embeddings)} embeddings for {len(texts)} input texts.")

        self._validate_embedding_dimensions(embeddings, texts)
        return embeddings

    async def _aollama_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with automatic batch slicing, fully async."""
        batch_size = settings.embedding_batch_size
        batches = self._slice_texts_into_batches(texts, batch_size)

        if len(batches) == 1:
            return await self._aollama_embed_single_batch(texts)

        logger.info("Processing %d texts in %d async batches (batch_size=%d)", len(texts), len(batches), batch_size)
        all_embeddings: list[list[float]] = []
        for batch_idx, batch_texts in enumerate(batches, start=1):
            logger.debug("Processing async batch %d/%d with %d texts", batch_idx, len(batches), len(batch_texts))
            batch_embeddings = await self._aollama_embed_single_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

        logger.info("Successfully async embedded all %d texts in %d batches", len(texts), len(batches))
        return all_embeddings

    def _validate_embedding_dimensions(self, embeddings: list[list[float]], texts: list[str]) -> None:
        expected_dim = settings.embedding_dimension
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
        """Embed a batch of texts using Ollama asynchronously."""
        vectors = await self._aollama_embed(texts)
        logger.info("Async embedded texts count=%s", len(texts))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string asynchronously."""
        results = await self.embed_texts([text])
        if not results or results[0] is None:
            raise EmbeddingError(f"Embedding returned empty result for query: {text[:80]!r}")
        return results[0]

    async def close(self) -> None:
        """Close the httpx client if it was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
