"""Tests for provider factory — build_global_llm_client, _build_purpose_llm_client, and build_embedder."""

from __future__ import annotations

import pytest

from data_engineering_copilot.config.settings import AppSettings


def _make_settings(**overrides) -> AppSettings:
    defaults = {
        "llm_provider": "ollama",
        "llm_model": "llama3.2:3b",
        "embedding_provider": "ollama",
        "openrouter_api_key": "",
        "openrouter_model": "anthropic/claude-3.5-sonnet",
        "openrouter_embedding_model": "nvidia/nemotron-3-embed-1b:free",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama3.2:3b",
        "ollama_timeout_seconds": 300,
        "ollama_num_ctx": 4096,
        "ollama_num_predict": 512,
        "embedding_model_name": "nomic-embed-text",
        "embedding_batch_size": 32,
    }
    # Clear all purpose overrides and API keys to prevent .env from leaking in
    for key in (
        "answer_llm_provider",
        "rewrite_llm_provider",
        "groundedness_llm_provider",
        "intent_llm_provider",
        "enrichment_llm_provider",
        "evaluation_llm_provider",
        "code_llm_provider",
        "nvidia_nim_api_key",
    ):
        defaults.setdefault(key, "")
    defaults.update(overrides)
    return AppSettings(**defaults)


def _make_settings_empty_key(provider: str, key_type: str = "llm") -> AppSettings:
    """Create settings with a provider set and a placeholder key, then clear the key.

    This bypasses pydantic's model_validator to let the factory function
    be the one that raises on missing keys.
    """
    if key_type == "embedding":
        s = _make_settings(embedding_provider=provider, openrouter_api_key="sk-placeholder")
    else:
        s = _make_settings(llm_provider=provider, openrouter_api_key="sk-placeholder", llm_model="test")
    object.__setattr__(s, "openrouter_api_key", AppSettings.model_fields["openrouter_api_key"].annotation(""))
    return s


class TestBuildGlobalLLMClient:
    def test_ollama_default(self):
        from data_engineering_copilot.factory import build_global_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(llm_provider="ollama", llm_model="llama3.2:3b")
        client = build_global_llm_client(s)
        assert isinstance(client, LLMClient)
        assert client.model == "llama3.2:3b"

    def test_openrouter(self):
        from data_engineering_copilot.factory import build_global_llm_client

        s = _make_settings(
            llm_provider="openrouter",
            llm_model="anthropic/claude-3.5-sonnet",
            openrouter_api_key="sk-or-v1-test",
        )
        client = build_global_llm_client(s)
        assert client is not None
        assert client.model == "anthropic/claude-3.5-sonnet"

    def test_unsupported_provider_falls_back(self):
        from data_engineering_copilot.factory import build_global_llm_client

        s = _make_settings(llm_provider="bedrock", llm_model="test")
        client = build_global_llm_client(s)
        assert client is not None
        assert client.model == "test"


class TestBuildPurposeLLMClient:
    def test_empty_provider_returns_client_with_global_defaults(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings()
        client = _build_purpose_llm_client(provider="", model="", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "llama3.2:3b"

    def test_nvidia(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_model="qwen/qwen2.5-coder-32b-instruct",
            nvidia_nim_api_key="nvapi-test",
        )
        client = _build_purpose_llm_client(
            provider="nvidia",
            model="qwen/qwen2.5-coder-32b-instruct",
            app_settings=s,
        )
        assert isinstance(client, LLMClient)
        assert client.model == "qwen/qwen2.5-coder-32b-instruct"

    def test_nvidia_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            nvidia_nim_api_key="nvapi-placeholder",
        )
        object.__setattr__(s, "nvidia_nim_api_key", AppSettings.model_fields["nvidia_nim_api_key"].annotation(""))
        with pytest.raises(ValueError, match="NVIDIA_NIM_API_KEY is required"):
            _build_purpose_llm_client(provider="nvidia", model="test", app_settings=s)

    def test_ollama(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(llm_provider="ollama", llm_model="llama3.2:3b")
        client = _build_purpose_llm_client(
            provider="",
            model="",
            app_settings=s,
        )
        assert isinstance(client, LLMClient)
        assert client.model == "llama3.2:3b"

    def test_openrouter_delegates(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="openrouter",
            llm_model="anthropic/claude-3.5-sonnet",
            openrouter_api_key="sk-or-v1-test",
        )
        client = _build_purpose_llm_client(
            provider="",
            model="",
            app_settings=s,
        )
        assert isinstance(client, LLMClient)
        assert client.model == "anthropic/claude-3.5-sonnet"

    def test_unsupported_provider_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(llm_provider="ollama", llm_model="llama3.2:3b")
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            _build_purpose_llm_client(provider="bedrock", model="test", app_settings=s)

    def test_nvidia_creates_separate_rate_limiter(self):
        from data_engineering_copilot.factory import _build_provider_rate_limiters, _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            nvidia_nim_api_key="nvapi-test",
            nvidia_nim_rpm_limit=80,
        )
        rate_limiters = _build_provider_rate_limiters(s)
        client = _build_purpose_llm_client(
            provider="nvidia",
            model="qwen/qwen2.5-coder-32b-instruct",
            app_settings=s,
            provider_rate_limiters=rate_limiters,
        )
        assert isinstance(client, LLMClient)
        assert client.model == "qwen/qwen2.5-coder-32b-instruct"


class TestBuildEmbedder:
    def test_ollama_default(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.async_embeddings import AsyncOllamaEmbeddings

        s = _make_settings(embedding_provider="ollama")
        embedder = build_embedder(s)
        assert isinstance(embedder, AsyncOllamaEmbeddings)
        assert embedder.model_name == "nomic-embed-text"

    def test_openrouter(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
            OpenAICompatibleEmbeddings,
        )

        s = _make_settings(
            embedding_provider="openrouter",
            openrouter_api_key="sk-or-v1-test",
            openrouter_embedding_model="nvidia/nemotron-3-embed-1b:free",
        )
        embedder = build_embedder(s)
        assert isinstance(embedder, OpenAICompatibleEmbeddings)

    def test_openrouter_missing_api_key_raises(self):
        from data_engineering_copilot.factory import build_embedder

        s = _make_settings_empty_key("openrouter", key_type="embedding")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
            build_embedder(s)

    def test_unsupported_provider_raises(self):
        from data_engineering_copilot.factory import build_embedder

        s = _make_settings(embedding_provider="voyage")
        with pytest.raises(ValueError, match="Unsupported embedding_provider"):
            build_embedder(s)

    def test_nvidia(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
            OpenAICompatibleEmbeddings,
        )

        s = _make_settings(
            embedding_provider="nvidia",
            nvidia_embedding_model="nvidia/nemotron-3-embed-1b",
            nvidia_nim_api_key="nvapi-test",
        )
        embedder = build_embedder(s)
        assert isinstance(embedder, OpenAICompatibleEmbeddings)
        assert embedder.model_name == "nvidia/nemotron-3-embed-1b"
        assert embedder.base_url == "https://integrate.api.nvidia.com/v1"

    def test_nvidia_missing_api_key_raises(self):
        from data_engineering_copilot.factory import build_embedder

        s = _make_settings(
            embedding_provider="nvidia",
            nvidia_nim_api_key="nvapi-placeholder",
        )
        object.__setattr__(s, "nvidia_nim_api_key", AppSettings.model_fields["nvidia_nim_api_key"].annotation(""))
        with pytest.raises(ValueError, match="NVIDIA_NIM_API_KEY is required"):
            build_embedder(s)
