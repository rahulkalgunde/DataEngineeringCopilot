"""Tests for circuit breaker and bulkhead integration with infrastructure services."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.infrastructure.resilience import (
    Bulkhead,
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerState,
)


class TestEmbeddingsCircuitBreaker:
    @patch("data_engineering_copilot.infrastructure.embeddings.urllib.request.urlopen")
    @patch("data_engineering_copilot.infrastructure.embeddings.settings")
    def test_circuit_breaker_on_embedding_endpoint(self, mock_settings, mock_urlopen):
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.ollama_timeout_seconds = 30
        mock_settings.embedding_dimension = 768
        mock_settings.embedding_batch_size = 32

        from data_engineering_copilot.infrastructure.embeddings import OllamaEmbeddings

        emb = OllamaEmbeddings(model_name="nomic-embed-text")
        assert hasattr(emb, "_circuit_breaker")
        assert isinstance(emb._circuit_breaker, CircuitBreaker)

    @patch("data_engineering_copilot.infrastructure.embeddings.urllib.request.urlopen")
    @patch("data_engineering_copilot.infrastructure.embeddings.settings")
    def test_bulkhead_on_embedding_endpoint(self, mock_settings, mock_urlopen):
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.ollama_timeout_seconds = 30
        mock_settings.embedding_dimension = 768
        mock_settings.embedding_batch_size = 32

        from data_engineering_copilot.infrastructure.embeddings import OllamaEmbeddings

        emb = OllamaEmbeddings(model_name="nomic-embed-text")
        assert hasattr(emb, "_bulkhead")
        assert isinstance(emb._bulkhead, Bulkhead)

    @patch("data_engineering_copilot.infrastructure.embeddings.urllib.request.urlopen")
    @patch("data_engineering_copilot.infrastructure.embeddings.settings")
    def test_circuit_opens_after_repeated_failures(self, mock_settings, mock_urlopen):
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.ollama_timeout_seconds = 30
        mock_settings.embedding_dimension = 768
        mock_settings.embedding_batch_size = 32
        mock_urlopen.side_effect = ConnectionError("refused")

        from data_engineering_copilot.domain.exceptions import EmbeddingError
        from data_engineering_copilot.infrastructure.embeddings import OllamaEmbeddings

        emb = OllamaEmbeddings(model_name="nomic-embed-text")
        emb._circuit_breaker._failure_threshold = 2

        with pytest.raises(EmbeddingError):
            emb._ollama_embed_single_batch(["hello"])
        assert emb._circuit_breaker.failure_count == 1
        assert emb._circuit_breaker.state == CircuitBreakerState.CLOSED

        with pytest.raises(EmbeddingError):
            emb._ollama_embed_single_batch(["hello"])
        assert emb._circuit_breaker.state == CircuitBreakerState.OPEN

        with pytest.raises(CircuitBreakerOpen):
            emb._ollama_embed_single_batch(["hello"])


class TestQdrantCircuitBreaker:
    def test_circuit_breaker_on_vector_store(self):
        with patch("data_engineering_copilot.infrastructure.qdrant_store.QdrantClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_cls.return_value = mock_client

            from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore

            store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
            assert hasattr(store, "_circuit_breaker")
            assert isinstance(store._circuit_breaker, CircuitBreaker)

    def test_bulkhead_on_vector_store(self):
        with patch("data_engineering_copilot.infrastructure.qdrant_store.QdrantClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_cls.return_value = mock_client

            from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore

            store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
            assert hasattr(store, "_bulkhead")
            assert isinstance(store._bulkhead, Bulkhead)


class TestOllamaClientCircuitBreaker:
    def test_circuit_breaker_on_client(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient

        client = OllamaClient(
            base_url="http://localhost:11434",
            model="llama3.2:3b",
            timeout_seconds=120,
            num_ctx=4096,
            num_predict=512,
        )
        assert hasattr(client, "_circuit_breaker")
        assert isinstance(client._circuit_breaker, CircuitBreaker)

    def test_bulkhead_on_client(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient

        client = OllamaClient(
            base_url="http://localhost:11434",
            model="llama3.2:3b",
            timeout_seconds=120,
            num_ctx=4096,
            num_predict=512,
        )
        assert hasattr(client, "_bulkhead")
        assert isinstance(client._bulkhead, Bulkhead)
