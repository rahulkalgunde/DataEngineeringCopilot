"""Tests for provider factory — build_llm_client and build_embedder."""

from __future__ import annotations

import pytest

from data_engineering_copilot.config.settings import AppSettings


def _make_settings(**overrides) -> AppSettings:
    defaults = {
        "llm_provider": "ollama",
        "embedding_provider": "ollama",
        "openrouter_api_key": "",
        "openrouter_model": "anthropic/claude-3.5-sonnet",
        "openrouter_embedding_model": "nvidia/nemotron-3-embed-1b:free",
        "openrouter_embedding_dimension": 2048,
        "openai_api_key": "",
        "openai_embedding_model": "text-embedding-3-small",
        "openai_embedding_base_url": "https://api.openai.com",
        "openai_embedding_dimension": 1536,
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama3.2:3b",
        "ollama_timeout_seconds": 300,
        "ollama_num_ctx": 4096,
        "ollama_num_predict": 512,
        "embedding_model_name": "nomic-embed-text",
        "embedding_batch_size": 32,
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


class TestBuildLLMClient:
    def test_ollama_default(self):
        from data_engineering_copilot.factory import build_llm_client
        from data_engineering_copilot.infrastructure.async_ollama_client import AsyncOllamaClient

        s = _make_settings(llm_provider="ollama")
        client = build_llm_client(s)
        assert isinstance(client, AsyncOllamaClient)
        assert client.model == "llama3.2:3b"

    def test_openrouter(self):
        from data_engineering_copilot.factory import build_llm_client
        from data_engineering_copilot.infrastructure.async_openrouter_client import OpenRouterLLMClient

        s = _make_settings(
            llm_provider="openrouter",
            openrouter_api_key="sk-or-v1-test",
            openrouter_model="anthropic/claude-3.5-sonnet",
        )
        client = build_llm_client(s)
        assert isinstance(client, OpenRouterLLMClient)
        assert client.model == "anthropic/claude-3.5-sonnet"

    def test_openrouter_missing_api_key_raises(self):
        from data_engineering_copilot.factory import build_llm_client

        s = _make_settings(llm_provider="openrouter", openrouter_api_key="")
        with pytest.raises(ValueError, match="openrouter_api_key is required"):
            build_llm_client(s)

    def test_unsupported_provider_raises(self):
        from data_engineering_copilot.factory import build_llm_client

        s = _make_settings(llm_provider="bedrock")
        with pytest.raises(ValueError, match="Unsupported llm_provider"):
            build_llm_client(s)


class TestBuildEmbedder:
    def test_ollama_default(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

        s = _make_settings(embedding_provider="ollama")
        embedder = build_embedder(s)
        assert isinstance(embedder, AsyncOllamaEmbeddings)
        assert embedder.model_name == "nomic-embed-text"

    def test_openai(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.async_openai_embeddings import OpenAIEmbeddings

        s = _make_settings(
            embedding_provider="openai",
            openai_api_key="sk-test",
            openai_embedding_model="text-embedding-3-small",
        )
        embedder = build_embedder(s)
        assert isinstance(embedder, OpenAIEmbeddings)
        assert embedder.model_name == "text-embedding-3-small"

    def test_openai_missing_api_key_raises(self):
        from data_engineering_copilot.factory import build_embedder

        s = _make_settings(embedding_provider="openai", openai_api_key="")
        with pytest.raises(ValueError, match="openai_api_key is required"):
            build_embedder(s)

    def test_openrouter(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.async_openrouter_embeddings import OpenRouterEmbeddings

        s = _make_settings(
            embedding_provider="openrouter",
            openrouter_api_key="sk-or-v1-test",
            openrouter_embedding_model="nvidia/nemotron-3-embed-1b:free",
        )
        embedder = build_embedder(s)
        assert isinstance(embedder, OpenRouterEmbeddings)
        assert embedder.model_name == "nvidia/nemotron-3-embed-1b:free"

    def test_openrouter_missing_api_key_raises(self):
        from data_engineering_copilot.factory import build_embedder

        s = _make_settings(embedding_provider="openrouter", openrouter_api_key="")
        with pytest.raises(ValueError, match="openrouter_api_key is required"):
            build_embedder(s)

    def test_unsupported_provider_raises(self):
        from data_engineering_copilot.factory import build_embedder

        s = _make_settings(embedding_provider="voyage")
        with pytest.raises(ValueError, match="Unsupported embedding_provider"):
            build_embedder(s)
