"""Tests for domain exception hierarchy."""

from __future__ import annotations

from data_engineering_copilot.domain.exceptions import (
    CoreDomainException,
    EmbeddingError,
    IngestionError,
    LLMGenerationError,
    RetrievalError,
    VectorStoreError,
)


def test_exception_hierarchy():
    assert issubclass(RetrievalError, CoreDomainException)
    assert issubclass(LLMGenerationError, CoreDomainException)
    assert issubclass(EmbeddingError, CoreDomainException)
    assert issubclass(VectorStoreError, CoreDomainException)
    assert issubclass(IngestionError, CoreDomainException)


def test_exceptions_can_be_instantiated_with_messages():
    err = RetrievalError("Failed to retrieve chunks")
    assert str(err) == "Failed to retrieve chunks"
    assert isinstance(err, CoreDomainException)
