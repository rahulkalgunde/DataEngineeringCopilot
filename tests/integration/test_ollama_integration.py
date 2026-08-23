"""Integration tests for OllamaClient and OllamaEmbeddings.

Tests LLM generation against a live Ollama daemon (embeddings moved to local-hf)
against a real Ollama instance.

Run with: pytest tests/integration/test_ollama_integration.py -v -m integration
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Embeddings integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.ollama
class TestOllamaClientGeneration:
    @pytest.mark.asyncio
    async def test_generate_returns_nonempty(self, ollama_client):
        """generate() should return a non-empty string."""
        answer = await ollama_client.generate("What is 2 + 2? Answer with just the number.")
        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_generate_with_custom_max_tokens(self, ollama_client):
        """generate() should respect max_tokens override."""
        answer = await ollama_client.generate(
            "Write a single sentence about data engineering.",
            max_tokens=100,
        )
        assert len(answer) > 0
        assert len(answer) < 2000

    @pytest.mark.asyncio
    async def test_generate_prompt_passthrough(self, ollama_client):
        """Client passes prompt as-is to Ollama (formatting lives in PromptBuilder)."""
        raw = "What is Spark? Answer briefly."
        answer = await ollama_client.generate(raw, max_tokens=50)
        assert len(answer) > 0
