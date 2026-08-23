"""Single source of truth for provider error categorization.

One canonical classifier shared by the LLM, embedding, and rerank fallback
chains (previously triplicated in ``provider_fallback._default_categorizer``
and two factory copies, which had already drifted). Behavior is the union of
all three former implementations:

- ``LLMClientError`` structured metadata (``status_code`` / ``retry_after`` /
  ``response_body``) is preferred over message matching.
- A 401/403 whose body indicates a model-not-supported error (some
  OpenAI-compatible gateways report wrong-model as 401) is categorised as
  ``INVALID_REQUEST``, not an auth failure, so the provider is not marked
  down for auth.
- httpx transport errors and timeouts map to RETRYABLE; keyword fallbacks
  cover providers that raise bare exceptions.
"""

from __future__ import annotations

import httpx

from data_engineering_copilot.domain.exceptions import (
    ProviderError,
    ProviderErrorCategory,
)
from data_engineering_copilot.infrastructure.llm_client import LLMClientError
from data_engineering_copilot.infrastructure.rate_limiter import SlidingWindowRateLimiter


def _is_model_not_supported_text(text: str) -> bool:
    """Heuristic for model-not-supported error bodies (e.g. opencodego's
    ``ModelError: "Model X is not supported"``)."""
    lowered = (text or "").lower()
    return (
        "not supported" in lowered
        or "modelerror" in lowered
        or "does not exist" in lowered
        or "unknown model" in lowered
        or ("model" in lowered and "not found" in lowered)
    )


def _response_body(exc: Exception) -> str:
    if isinstance(exc, LLMClientError) and exc.response_body:
        return exc.response_body
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.text or ""
    return ""


def _status_category(status: int, body: str) -> ProviderErrorCategory:
    if body and _is_model_not_supported_text(body) and status in (401, 403):
        return ProviderErrorCategory.INVALID_REQUEST
    if status == 429:
        return ProviderErrorCategory.RATE_LIMITED
    if status in (401, 403):
        return ProviderErrorCategory.AUTHENTICATION_ERROR
    if status in (400, 422):
        return ProviderErrorCategory.INVALID_REQUEST
    if status >= 500:
        return ProviderErrorCategory.TEMPORARY_UNAVAILABLE
    return ProviderErrorCategory.PERMANENT_ERROR


def categorize_provider_error(exc: Exception, provider: str, model: str) -> ProviderError:
    """Categorise a provider failure into a :class:`ProviderError`."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        retry_after = SlidingWindowRateLimiter.parse_retry_after(dict(exc.response.headers))
        return ProviderError(
            _status_category(status, exc.response.text),
            provider,
            model,
            retry_after=retry_after if status == 429 else None,
            original=exc,
        )

    if isinstance(exc, LLMClientError) and exc.status_code is not None:
        return ProviderError(
            _status_category(exc.status_code, exc.response_body or ""),
            provider,
            model,
            retry_after=exc.retry_after,
            original=exc,
        )

    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.TransportError,
            httpx.ProtocolError,
            httpx.DecodingError,
            TimeoutError,
            OSError,
        ),
    ):
        # TransportError/ProtocolError cover mid-stream drops (ReadError,
        # RemoteProtocolError, ...) that often carry EMPTY messages — these
        # are transient connection failures, not permanent provider faults
        # (2026-08-23: NVIDIA rerank benched on ReadError("") misclassification).
        return ProviderError(ProviderErrorCategory.RETRYABLE, provider, model, original=exc)

    lower_msg = str(exc).lower()
    if "rate limit" in lower_msg or "429" in lower_msg or "too many requests" in lower_msg:
        return ProviderError(ProviderErrorCategory.RATE_LIMITED, provider, model, original=exc)
    if "quota" in lower_msg or "exceeded" in lower_msg:
        return ProviderError(ProviderErrorCategory.QUOTA_EXCEEDED, provider, model, original=exc)
    if _is_model_not_supported_text(lower_msg) and ("401" in lower_msg or "unauthorized" in lower_msg):
        return ProviderError(ProviderErrorCategory.INVALID_REQUEST, provider, model, original=exc)
    if "401" in lower_msg or "unauthorized" in lower_msg or "authentication" in lower_msg:
        return ProviderError(ProviderErrorCategory.AUTHENTICATION_ERROR, provider, model, original=exc)
    if "timed out" in lower_msg or "timeout" in lower_msg:
        return ProviderError(ProviderErrorCategory.RETRYABLE, provider, model, original=exc)
    if "could not reach" in lower_msg or "connection" in lower_msg:
        return ProviderError(ProviderErrorCategory.RETRYABLE, provider, model, original=exc)

    return ProviderError(ProviderErrorCategory.PERMANENT_ERROR, provider, model, original=exc)
