"""Integration tests for OllamaClient and OllamaEmbeddings.

Tests embedding generation, text embedding batches, and LLM generation
against a real Ollama instance.

Run with: pytest tests/integration/test_ollama_integration.py -v -m integration
"""

from __future__ import annotations

import asyncio

import pytest

# ---------------------------------------------------------------------------
# Embeddings integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.ollama
class TestOllamaEmbeddings:
    def test_embed_query_returns_vector(self, embeddings_provider):
        """embed_query should return a 768-dim vector."""
        vec = asyncio.run(embeddings_provider.embed_query("What is Apache Spark?"))
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)

    def test_embed_texts_batch(self, embeddings_provider):
        """embed_texts should return one vector per input text."""
        texts = [
            "Apache Spark is a unified analytics engine.",
            "Delta Lake provides ACID transactions.",
            "Apache Airflow orchestrates workflows.",
        ]
        vecs = asyncio.run(embeddings_provider.embed_texts(texts))
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 768

    def test_embed_query_deterministic(self, embeddings_provider):
        """Same input should produce very similar (nearly identical) embeddings."""
        text = "Deterministic embedding test"
        v1 = asyncio.run(embeddings_provider.embed_query(text))
        v2 = asyncio.run(embeddings_provider.embed_query(text))
        # Cosine similarity should be very close to 1.0
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        similarity = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        assert similarity > 0.99, f"Expected near-identical embeddings, got similarity={similarity:.4f}"

    def test_embed_texts_large_batch(self, embeddings_provider):
        """Embedding a batch of 32 texts should work without OOM."""
        texts = [f"This is test document number {i} about data engineering." for i in range(32)]
        vecs = asyncio.run(embeddings_provider.embed_texts(texts))
        assert len(vecs) == 32
        for v in vecs:
            assert len(v) == 768


# ---------------------------------------------------------------------------
# OllamaClient integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.ollama
class TestOllamaClientGeneration:
    def test_generate_returns_nonempty(self, ollama_client):
        """generate() should return a non-empty string."""
        answer = asyncio.run(ollama_client.generate("What is 2 + 2? Answer with just the number."))
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_generate_with_custom_num_predict(self, ollama_client):
        """generate() should respect num_predict override."""
        answer = asyncio.run(
            ollama_client.generate(
                "Write a single sentence about data engineering.",
                num_predict=100,
            )
        )
        assert len(answer) > 0
        # num_predict limits tokens, not words; just verify it returns something
        assert len(answer) < 2000

    def test_generate_prompt_formatting(self, ollama_client):
        """The internal prompt formatter should wrap user input."""
        raw = "What is Spark?"
        formatted = ollama_client._format_raw_chat_prompt(raw)
        assert "DataEngineeringCopilot" in formatted
        assert raw in formatted
        assert "SYSTEM" in formatted
