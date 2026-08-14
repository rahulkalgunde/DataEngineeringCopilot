"""Async rerank provider clients for the unified fallback chain.

Three cloud providers implement the ``ProviderClient`` shape consumed by
``ProviderFallbackChain[RerankRequest, RerankResult]``:

- OpenRouter ``/api/v1/rerank`` (NVIDIA nemotron rerank models, e.g.
  ``nvidia/llama-nemotron-rerank-vl-1b-v2:free``) — relevance scores already
  in ``[0, 1]``.
- NVIDIA ``/v1/retrieval/nvidia/reranking`` (``nv-rerank-qa-mistral-4b:1``) —
  raw ``logit`` values normalized with sigmoid.
- Hugging Face ``text-classification`` pipeline (``BAAI/bge-reranker-v2-m3``).
  The serverless backend has no dedicated rerank task (dedicated endpoints
  only), so the model is invoked as a cross-encoder classifier over
  ``"<query> <passage>"`` inputs. The ``hf-inference`` backend returns a flat
  ``[{"label":..., "score":...}, ...]`` list in input order (one entry per
  document), so scores map back by position.

All clients normalize scores to ``[0, 1]`` so the rerank confidence gate
(``reranker_confidence_threshold``) keeps the same meaning across providers.
Each mirrors the ``HuggingFaceServerlessEmbeddings`` HTTP discipline: shared
``SafeAsyncClientMixin`` client, tenacity retries, a non-blocking rate-limit
gate (``try_acquire``), and 429 handling that records ``Retry-After`` via
``handle_429`` before raising an ``httpx.HTTPStatusError`` the fallback chain
categorizes as ``RATE_LIMITED`` (honoring the header).
"""

from __future__ import annotations

import logging
import math

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base

from data_engineering_copilot.domain.exceptions import RerankError
from data_engineering_copilot.domain.models import LLMUsage, RerankRequest, RerankResult
from data_engineering_copilot.infrastructure.async_client import SafeAsyncClientMixin
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

# HTTP statuses that mean "try again shortly" (the fallback chain also sees
# these after the tenacity retries are exhausted).
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


class _BaseRerankClient(SafeAsyncClientMixin):
    """Shared HTTP plumbing for rerank provider clients.

    Requirements on ``self``:
        - ``self.api_key`` — str
        - ``self.model_name`` — str
        - ``self.base_url`` — str (full endpoint for OpenRouter/NVIDIA; the
          ``hf-inference`` root for Hugging Face)
        - ``self.timeout_seconds`` — int | float
        - ``self._rate_limiter`` — SlidingWindowRateLimiter | None
        - ``self.extra_headers`` — dict
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        timeout_seconds: int | float = 60,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        extra_headers: dict | None = None,
        retry_wait: wait_base | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._rate_limiter = rate_limiter
        self._extra_headers = extra_headers or {}
        # Retry transient failures a few times before the chain fails over.
        self._http_request = retry(
            stop=stop_after_attempt(4),
            wait=retry_wait or wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, OSError, httpx.HTTPStatusError)),
            reraise=True,
        )(self._http_request)

    def _make_client_kwargs(self) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        headers.update(self._extra_headers)
        return {"headers": headers}

    async def _get_client(self) -> httpx.AsyncClient:
        return await self._get_safe_client()

    async def _http_request(self, url: str, payload: dict) -> httpx.Response:
        # Non-blocking rate-limit gate: when the RPM/RPD window is exhausted,
        # fail fast so the fallback chain skips to the next provider.
        if self._rate_limiter is not None and not await self._rate_limiter.try_acquire():
            raise RerankError("Rate limit window exhausted; rerank provider not called.")

        response = await (await self._get_client()).post(url, json=payload)

        if response.status_code == 429:
            if self._rate_limiter is not None:
                await self._rate_limiter.handle_429(dict(response.headers))
            raise httpx.HTTPStatusError(  # tenacity retries; chain maps 429 -> RATE_LIMITED
                "Rate limited by rerank provider",
                request=response.request,
                response=response,
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _RETRYABLE_STATUSES:
                raise  # tenacity retries; chain maps 5xx -> TEMPORARY_UNAVAILABLE
            raise RerankError(f"Rerank request failed: {exc}") from exc

        return response

    async def _post_and_parse(self, url: str, payload: dict) -> dict:
        response = await self._http_request(url, payload)
        try:
            data = response.json()
        except ValueError as exc:
            raise RerankError(f"Rerank provider returned non-JSON body: {response.text[:200]!r}") from exc
        if not isinstance(data, dict):
            raise RerankError(
                f"Rerank provider response is not an object. Got {type(data).__name__}: {str(data)[:200]}"
            )
        return data

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


class OpenRouterRerankClient(_BaseRerankClient):
    """OpenRouter ``/api/v1/rerank`` client (NVIDIA nemotron rerank models)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        base_url: str = "https://openrouter.ai/api/v1/rerank",
        timeout_seconds: int | float = 60,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        retry_wait: wait_base | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            rate_limiter=rate_limiter,
            extra_headers={
                # OpenRouter encourages provenance headers for free-tier models.
                "HTTP-Referer": "http://localhost",
                "X-Title": "data-engineering-copilot",
            },
            retry_wait=retry_wait,
        )

    async def call(self, request: RerankRequest) -> RerankResult:
        """Rerank ``request.documents`` against ``request.query``."""
        if not request.documents:
            return RerankResult()
        payload = {
            "model": self.model_name,
            "query": request.query,
            "documents": request.documents,
            "top_n": request.top_n,
        }
        data = await self._post_and_parse(self.base_url, payload)
        results = data.get("results")
        if not isinstance(results, list):
            raise RerankError(f"OpenRouter rerank response missing 'results' list: {str(data)[:200]}")
        rankings: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or not isinstance(score, (int, float)):
                continue
            rankings.append((index, float(score)))
        rankings.sort(key=lambda pair: pair[1], reverse=True)
        return RerankResult(rankings=tuple(rankings))


class NvidiaRerankClient(_BaseRerankClient):
    """NVIDIA NIM reranking endpoint client (``nv-rerank-qa-mistral-4b:1``)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "nv-rerank-qa-mistral-4b:1",
        base_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        timeout_seconds: int | float = 60,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        retry_wait: wait_base | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            rate_limiter=rate_limiter,
            retry_wait=retry_wait,
        )

    async def call(self, request: RerankRequest) -> RerankResult:
        """Rerank ``request.documents`` against ``request.query``."""
        if not request.documents:
            return RerankResult()
        payload = {
            "model": self.model_name,
            "query": {"text": request.query},
            "passages": [{"text": doc} for doc in request.documents],
            "truncate": "END",
        }
        data = await self._post_and_parse(self.base_url, payload)
        rankings_raw = data.get("rankings")
        if not isinstance(rankings_raw, list):
            raise RerankError(f"NVIDIA rerank response missing 'rankings' list: {str(data)[:200]}")
        rankings: list[tuple[int, float]] = []
        for item in rankings_raw:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            logit = item.get("logit")
            if not isinstance(index, int) or not isinstance(logit, (int, float)):
                continue
            rankings.append((index, _sigmoid(float(logit))))
        rankings.sort(key=lambda pair: pair[1], reverse=True)
        return RerankResult(rankings=tuple(rankings))


class HuggingFaceRerankClient(_BaseRerankClient):
    """Hugging Face serverless rerank client (text-classification pipeline).

    The serverless ``hf-inference`` backend serves ``BAAI/bge-reranker-v2-m3``
    through the ``text-classification`` pipeline (no dedicated serverless rerank
    task exists). ``query`` and ``passage`` are concatenated with a space,
    matching the model's training format. The backend returns a flat list of
    ``{"label":..., "score":...}`` entries in input order; scores map back to
    documents by position.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "https://router.huggingface.co/hf-inference",
        timeout_seconds: int | float = 120,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        retry_wait: wait_base | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            rate_limiter=rate_limiter,
            retry_wait=retry_wait,
        )

    def _pipeline_url(self) -> str:
        return f"/models/{self.model_name}/pipeline/text-classification"

    async def call(self, request: RerankRequest) -> RerankResult:
        """Rerank ``request.documents`` against ``request.query``."""
        if not request.documents:
            return RerankResult()
        inputs = [f"{request.query} {doc}" for doc in request.documents]
        payload = {"inputs": inputs}
        response = await self._http_request(self.base_url + self._pipeline_url(), payload)
        try:
            body = response.json()
        except ValueError as exc:
            raise RerankError(f"Hugging Face rerank returned non-JSON body: {response.text[:200]!r}") from exc

        scores = self._parse_classification_scores(body, len(inputs))
        rankings = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return RerankResult(rankings=tuple(rankings[: request.top_n]))

    def _parse_classification_scores(self, body: object, expected: int) -> list[float]:
        """Parse the text-classification response into one score per input.

        Accepts both the serverless flat shape ``[[{label,score}, ...]]`` (one
        entry per input, in order) and the standard per-input shape
        ``[[{label,score}, ...], ...]`` (labels per input). When a standard
        multi-label list is seen, the max label score is taken.
        """
        if not isinstance(body, list) or not body:
            raise RerankError(f"Hugging Face rerank response is not a list: {str(body)[:200]!r}")

        # Flat serverless shape: single outer list, one label dict per input.
        if len(body) == 1 and isinstance(body[0], list) and len(body[0]) == expected:
            return [float(item.get("score", 0.0)) for item in body[0] if isinstance(item, dict)]

        # Standard shape: one inner list per input; take the max label score.
        if len(body) == expected:
            out: list[float] = []
            for entry in body:
                if not isinstance(entry, list) or not entry:
                    raise RerankError(f"Hugging Face rerank malformed per-input entry: {str(entry)[:100]!r}")
                scores = [float(item.get("score", 0.0)) for item in entry if isinstance(item, dict)]
                if not scores:
                    raise RerankError(f"Hugging Face rerank entry has no labels: {str(entry)[:100]!r}")
                out.append(max(scores))
            return out

        raise RerankError(
            f"Hugging Face rerank response shape mismatch: got {len(body)} outer entries for "
            f"{expected} inputs. Body: {str(body)[:200]!r}"
        )


class LocalRerankerClient:
    """Adapter exposing the local cross-encoder reranker as a ``ProviderClient``.

    Used as the ``degraded_fallback`` in the rerank fallback chain so the local
    ``CrossEncoderReranker`` remains the last resort after all cloud providers
    are skipped or fail.
    """

    def __init__(self, reranker) -> None:
        self._reranker = reranker
        self._model_name = getattr(reranker, "model_name", "local-crossencoder")

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def last_usage(self) -> LLMUsage:
        return LLMUsage()

    async def call(self, request: RerankRequest) -> RerankResult:
        """Score documents with the local cross-encoder (sigmoid-normalized)."""
        if not request.documents:
            return RerankResult()
        scores = await self._reranker.score_documents(request.query, request.documents)
        rankings = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return RerankResult(rankings=tuple(rankings[: request.top_n]))

    async def close(self) -> None:
        if hasattr(self._reranker, "close"):
            await self._reranker.close()
