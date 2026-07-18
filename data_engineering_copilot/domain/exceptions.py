"""Domain exception hierarchy.

All application-specific exceptions inherit from ``DataCopilotError``.
"""


class DataCopilotError(RuntimeError):
    """Base exception for all DataEngineeringCopilot errors."""


class EmbeddingError(DataCopilotError):
    """Raised when embedding generation fails."""


class VectorStoreError(DataCopilotError):
    """Raised when a vector store operation fails."""


class IngestionError(DataCopilotError):
    """Raised when the ingestion pipeline fails."""


class CrawlError(DataCopilotError):
    """Raised when a single page crawl fails (non-fatal, page is skipped)."""
