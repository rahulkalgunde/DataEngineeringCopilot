"""Domain exception hierarchy.

All domain exceptions inherit from ``CoreDomainException``.
"""


class CoreDomainException(RuntimeError):
    """Base exception for all domain-level errors."""


class DataCopilotError(CoreDomainException):
    """Base exception for legacy/application errors."""


class EmbeddingError(CoreDomainException):
    """Raised when embedding generation fails."""


class VectorStoreError(CoreDomainException):
    """Raised when a vector store operation fails."""


class IngestionError(CoreDomainException):
    """Raised when the ingestion pipeline fails."""


class CrawlError(CoreDomainException):
    """Raised when a single page crawl fails (non-fatal, page is skipped)."""


class RetrievalError(CoreDomainException):
    """Raised when document retrieval fails during RAG pipeline."""


class LLMGenerationError(CoreDomainException):
    """Raised when LLM text generation fails."""
