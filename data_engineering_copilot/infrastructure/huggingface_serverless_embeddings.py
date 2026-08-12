"""Async Hugging Face serverless embedding provider (native feature-extraction).

Hugging Face Inference Providers serve ``nvidia/Nemotron-3-Embed-1B-BF16``
through the ``feature-extraction`` pipeline of the ``hf-inference`` provider.
The OpenAI-compatible ``https://router.huggingface.co/v1`` surface is
**chat-completions only** — ``POST /v1/embeddings`` returns 404 even with a
valid token, and ``/v1/models`` lists no embedding models. Embeddings must go
through the native per-model pipeline route:

    POST https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction
    {"inputs": ["text", ...]}
    -> [[float, ...], ...]   # raw array-of-arrays, not an OpenAI wrapper

The serverless backend does **not** apply the model's ``query:``/``passage:``
prefixes (the ``prompt_name`` parameter is ignored, verified empirically), so
this client prepends them itself — exactly like
``LocalSentenceTransformerEmbeddings``. Verified: embeddings match the local
``nvidia/Nemotron-3-Embed-1B-BF16`` model at cosine ~1.0, so the provider can
share a Qdrant collection with the NVIDIA/OpenRouter/local-hf chain.

Free tier: every HF account gets $0.10/month of Inference Provider credits
(routed via ``router.huggingface.co``); ``hf-inference`` mostly runs CPU
inference for embedding models, so batch requests are cheap.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
import tiktoken
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base

from data_engineering_copilot.domain.exceptions import EmbeddingError
from data_engineering_copilot.domain.models import EmbeddingRequest, LLMUsage
from data_engineering_copilot.infrastructure.async_client import SafeAsyncClientMixin
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

# Native feature-extraction route for the ``hf-inference`` serverless provider.
DEFAULT_BASE_URL = "https://router.huggingface.co/hf-inference"
# Pipeline path appended to the model id (model id keeps its ``org/repo`` slash).
FEATURE_EXTRACTION_PATH = "/models/{model}/pipeline/feature-extraction"
# HTTP statuses that mean "try again shortly" (the fallback chain also sees
# these after the tenacity retries are exhausted).
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# Fallback token encoder (matches the OpenAI-compatible tokenizer ratio; the
# real model tokenizer is threaded in via ``token_counter`` when available).
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


class HuggingFaceServerlessEmbeddings(SafeAsyncClientMixin):
    """Async embedding provider for the HF Inference Provider feature-extraction route.

    Satisfies ``EmbedderProtocol`` and the ``ProviderClient`` shape used by the
    fallback chain (``call``, ``model``, ``last_usage``, ``close``). Applies
    the ``query:``/``passage:`` prefixes client-side since the serverless
    backend ignores ``prompt_name``.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "nvidia/Nemotron-3-Embed-1B-BF16",
        base_url: str = DEFAULT_BASE_URL,
        embedding_dimension: int = 2048,
        batch_size: int = 64,
        timeout_seconds: int = 120,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        retry_wait: wait_base | None = None,
        max_tokens_per_input: int = 3800,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._embedding_dimension = embedding_dimension
        self._batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self._rate_limiter = rate_limiter
        self._max_tokens_per_input = max_tokens_per_input
        self._token_counter = token_counter or _count_tokens
        self._request_feature_extraction = retry(
            stop=stop_after_attempt(4),
            wait=retry_wait or wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, OSError, httpx.HTTPStatusError)),
            reraise=True,
        )(self._request_feature_extraction)

    def _make_client_kwargs(self) -> dict:
        return {
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
            }
        }

    async def _get_client(self) -> httpx.AsyncClient:
        return await self._get_safe_client()

    def _pipeline_url(self) -> str:
        return FEATURE_EXTRACTION_PATH.format(model=self.model_name)

    async def _request_feature_extraction(self, texts: list[str], prefix: str | None) -> list[list[float]]:
        # Non-blocking rate-limit gate: when the RPM/RPD window is exhausted,
        # fail fast so the fallback chain skips to the next provider.
        if self._rate_limiter is not None and not await self._rate_limiter.try_acquire():
            raise EmbeddingError("Rate limit window exhausted; embedding provider not called.")

        # Never truncate input to fit the provider limit — reject over-budget
        # text so silent content loss can never corrupt the index.
        for text in texts:
            token_count = self._token_counter(text)
            if token_count > self._max_tokens_per_input:
                raise EmbeddingError(
                    f"Embedding input exceeds budget: model={self.model_name} tokens={token_count} "
                    f"allowed={self._max_tokens_per_input}. Split the text losslessly before embedding."
                )

        prepared = [f"{prefix}: {t}" if prefix else t for t in texts]
        response = await (await self._get_client()).post(self._pipeline_url(), json={"inputs": prepared})

        if response.status_code == 429:
            if self._rate_limiter is not None:
                await self._rate_limiter.handle_429(dict(response.headers))
            raise httpx.HTTPStatusError(  # tenacity will retry this
                "Rate limited by Hugging Face serverless embedding provider",
                request=response.request,
                response=response,
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _RETRYABLE_STATUSES:
                raise  # tenacity retries; fallback categorizer maps 5xx -> TEMPORARY_UNAVAILABLE
            raise EmbeddingError(f"Failed to get embeddings from Hugging Face: {exc}") from exc

        try:
            vectors = response.json()
        except ValueError as exc:
            raise EmbeddingError(f"Hugging Face embeddings returned non-JSON body: {response.text[:200]!r}") from exc

        if not isinstance(vectors, list):
            raise EmbeddingError(
                f"Hugging Face embeddings response is not a list. Got {type(vectors).__name__}: {str(vectors)[:200]}"
            )
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Hugging Face returned {len(vectors)} embeddings for {len(texts)} input texts.")
        self._validate_dimensions(vectors, texts)
        return vectors

    def _validate_dimensions(self, embeddings: list[list[float]], texts: list[str]) -> None:
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list):
                raise EmbeddingError(f"Embedding {i} is not a list. Got {type(emb).__name__}. Text: {texts[i][:100]!r}")
            if len(emb) != self._embedding_dimension:
                raise EmbeddingError(f"Embedding {i} has dimension {len(emb)}, expected {self._embedding_dimension}.")

    async def _embed_with_batching(self, texts: list[str], prefix: str | None) -> list[list[float]]:
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        all_embeddings: list[list[float]] = []
        for batch in batches:
            all_embeddings.extend(await self._request_feature_extraction(batch, prefix=prefix))
        return all_embeddings

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._embed_with_batching(texts, prefix="passage")

    async def embed_query(self, text: str) -> list[float]:
        results = await self._embed_with_batching([text], prefix="query")
        if not results or results[0] is None:
            raise EmbeddingError(f"Embedding returned empty result for query: {text[:80]!r}")
        return results[0]

    # ProviderClient protocol method
    async def call(self, request: list[str] | EmbeddingRequest) -> list[list[float]]:
        """Unified call interface for ``ProviderFallbackChain``."""
        if isinstance(request, EmbeddingRequest):
            prefix = {"query": "query", "passage": "passage"}.get(request.input_type) if request.input_type else None
            texts = request.texts
        else:
            prefix = "passage"
            texts = list(request)
        if not texts:
            return []
        return await self._embed_with_batching(texts, prefix=prefix)

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
            self._client_loop = None
