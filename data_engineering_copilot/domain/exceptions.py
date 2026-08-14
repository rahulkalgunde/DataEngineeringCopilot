"""Domain exception hierarchy.

All domain exceptions inherit from ``CoreDomainException``.
"""

from __future__ import annotations

from enum import Enum


class CoreDomainException(RuntimeError):
    """Base exception for all domain-level errors."""


class DataCopilotError(CoreDomainException):
    """Base exception for legacy/application errors."""


class EmbeddingError(CoreDomainException):
    """Raised when embedding generation fails."""


class RerankError(CoreDomainException):
    """Raised when provider reranking fails."""


class VectorStoreError(CoreDomainException):
    """Raised when a vector store operation fails."""


class IngestionError(CoreDomainException):
    """Raised when the ingestion pipeline fails."""


class CrawlError(CoreDomainException):
    """Raised when a single page crawl fails (non-fatal, page is skipped)."""


class RetrievalError(CoreDomainException):
    """Raised when document retrieval fails during RAG pipeline."""


class AuthorizationError(CoreDomainException):
    """Raised when a caller lacks permission for a requested resource.

    Mapped to HTTP 403 by the API layer. Never leaks internal detail.
    """


class LLMGenerationError(CoreDomainException):
    """Raised when LLM text generation fails."""


class ProviderErrorCategory(Enum):
    """Categorised outcome of a provider call.

    Used by the adaptive router to decide whether to retry, fail over,
    or abort.
    """

    SUCCESS = "success"
    RETRYABLE = "retryable"
    RATE_LIMITED = "rate_limited"
    TEMPORARY_UNAVAILABLE = "temporary_unavailable"
    PERMANENT_ERROR = "permanent_error"
    AUTHENTICATION_ERROR = "authentication_error"
    INVALID_REQUEST = "invalid_request"
    QUOTA_EXCEEDED = "quota_exceeded"


class ProviderError(CoreDomainException):
    """An error from an LLM provider that includes a categorised reason.

    Attributes
    ----------
    category:
        The category of error that occurred.
    provider:
        The provider name (e.g. ``"openrouter"``).
    model:
        The model name that was being called.
    original:
        The original exception, if any.
    retry_after:
        Suggested wait time (seconds) from a ``Retry-After`` header, or
        ``None``.
    """

    def __init__(
        self,
        category: ProviderErrorCategory,
        provider: str,
        model: str,
        message: str = "",
        original: Exception | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.category = category
        self.provider = provider
        self.model = model
        self.original = original
        self.retry_after = retry_after
        super().__init__(message or f"[{provider}/{model}] {category.value}")
