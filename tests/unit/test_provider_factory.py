"""Tests for provider factory — build_global_llm_client, _build_purpose_llm_client, and build_embedder."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from data_engineering_copilot.config.settings import AppSettings
from tests.conftest import make_settings

_PROVIDER_FIELDS = frozenset(
    {
        "llm_provider",
        "embedding_provider",
        "code_llm_provider",
        "answer_llm_provider",
        "rewrite_llm_provider",
        "groundedness_llm_provider",
        "intent_llm_provider",
        "enrichment_llm_provider",
        "evaluation_llm_provider",
    }
)


def _make_settings(**overrides) -> AppSettings:
    """Hermetic test settings via ``tests.conftest.make_settings``.

    Auto-opts into non-Ollama provider routing (``_test_allow_non_ollama``)
    whenever a provider field explicitly requests a non-Ollama provider — this
    file deliberately exercises the factory's provider-routing logic, always
    with placeholder API keys.
    """
    if any(k in _PROVIDER_FIELDS and v not in ("", "ollama") for k, v in overrides.items()):
        overrides.setdefault("_test_allow_non_ollama", True)
    return make_settings(**overrides)


def _make_settings_empty_key(provider: str, key_type: str = "llm") -> AppSettings:
    """Build settings with a placeholder key, then clear it post-validation so
    the factory's own missing-key check is what raises."""
    if key_type == "embedding":
        s = _make_settings(embedding_provider=provider, openrouter_api_key="sk-placeholder")
    else:
        s = _make_settings(llm_provider=provider, openrouter_api_key="sk-placeholder", llm_model="test")
    object.__setattr__(s, "openrouter_api_key", SecretStr(""))
    return s


class TestBuildGlobalLLMClient:
    def test_ollama_default(self):
        from data_engineering_copilot.factory import build_global_llm_client
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain

        s = _make_settings(llm_provider="ollama", llm_model="llama3.2:3b")
        client = build_global_llm_client(s)
        assert isinstance(client, ProviderFallbackChain)

    def test_openrouter(self):
        from data_engineering_copilot.factory import build_global_llm_client

        s = _make_settings(
            llm_provider="openrouter",
            llm_model="anthropic/claude-3.5-sonnet",
            openrouter_model="anthropic/claude-3.5-sonnet",
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
        assert client.model == "llama3.2:3b"


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
            nvidia_api_key="nvapi-test",
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
            nvidia_api_key="nvapi-placeholder",
        )
        object.__setattr__(s, "nvidia_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
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

    def test_ollama_sends_max_tokens_not_ignored_options(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(llm_provider="ollama", llm_model="llama3.2:3b")
        client = _build_purpose_llm_client(provider="", model="", app_settings=s)
        assert client is not None
        assert client._max_tokens == s.ollama_num_predict == 512
        assert client._max_tokens_field == "max_tokens"
        # options.num_ctx / options.num_predict / keep_alive are ignored on the
        # OpenAI-compat endpoint and must not be sent.
        assert client._extra_body == {}
        assert client._keep_alive is None

    def test_nvidia_per_purpose_max_tokens(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(nvidia_api_key="nvapi-test")
        answer = _build_purpose_llm_client(
            provider="nvidia", model="meta/llama-3.1-8b-instruct", purpose="answer", app_settings=s
        )
        rewrite = _build_purpose_llm_client(
            provider="nvidia", model="meta/llama-3.1-8b-instruct", purpose="rewrite", app_settings=s
        )
        assert answer is not None and rewrite is not None
        assert answer._max_tokens == s.purpose_max_tokens["answer"] == 4096
        assert rewrite._max_tokens == s.purpose_max_tokens["rewrite"] == 768
        assert answer._max_tokens_field == "max_tokens"

    def test_groq_cerebras_use_max_completion_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(groq_api_key="gsk-test", cerebras_api_key="cb-test")
        groq = _build_purpose_llm_client(
            provider="groq", model="llama-3.1-8b-instant", purpose="answer", app_settings=s
        )
        cerebras = _build_purpose_llm_client(
            provider="cerebras", model="gpt-oss-120b", purpose="answer", app_settings=s
        )
        assert groq is not None and cerebras is not None
        assert groq._max_tokens_field == "max_completion_tokens"
        assert cerebras._max_tokens_field == "max_completion_tokens"
        assert groq._max_tokens == 4096

    def test_opencodezen_uses_base_url_model_and_max_completion_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="opencodezen",
            opencodezen_api_key="oc-test",
        )
        client = _build_purpose_llm_client(provider="opencodezen", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "deepseek-v4-flash-free"
        assert client.base_url == "https://opencode.ai/zen/v1"
        assert client._max_tokens_field == "max_completion_tokens"
        assert client._max_tokens == 4096

    def test_opencodezen_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="opencodezen",
            opencodezen_api_key="oc-placeholder",
        )
        object.__setattr__(s, "opencodezen_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="OPENCODEZEN_API_KEY is required"):
            _build_purpose_llm_client(provider="opencodezen", model="test", app_settings=s)

    def test_opencodego_uses_base_url_model_and_max_completion_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="opencodego",
            opencodego_api_key="oc-go-test",
        )
        client = _build_purpose_llm_client(provider="opencodego", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "deepseek-v4-flash"
        assert client.base_url == "https://opencode.ai/zen/go/v1"
        assert client._max_tokens_field == "max_completion_tokens"
        assert client._max_tokens == 4096

    def test_opencodego_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="opencodego",
            opencodego_api_key="oc-go-placeholder",
        )
        object.__setattr__(s, "opencodego_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="OPENCODEGO_API_KEY is required"):
            _build_purpose_llm_client(provider="opencodego", model="test", app_settings=s)

    def test_sambanova_uses_base_url_model_and_max_completion_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="sambanova",
            sambanova_api_key="sn-test",
        )
        client = _build_purpose_llm_client(provider="sambanova", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "Meta-Llama-3.3-70B-Instruct"
        assert client.base_url == "https://api.sambanova.ai/v1"
        assert client._max_tokens_field == "max_completion_tokens"
        assert client._max_tokens == 4096

    def test_sambanova_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="sambanova",
            sambanova_api_key="sn-placeholder",
        )
        object.__setattr__(s, "sambanova_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="SAMBANOVA_API_KEY is required"):
            _build_purpose_llm_client(provider="sambanova", model="test", app_settings=s)

    def test_mistral_uses_base_url_model_and_max_completion_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="mistral",
            mistral_api_key="mistral-test-key",
        )
        client = _build_purpose_llm_client(provider="mistral", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "mistral-small-latest"
        assert client.base_url == "https://api.mistral.ai/v1"
        assert client._max_tokens_field == "max_completion_tokens"
        assert client._max_tokens == 4096

    def test_mistral_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="mistral",
            mistral_api_key="mistral-placeholder",
        )
        object.__setattr__(s, "mistral_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="MISTRAL_API_KEY is required"):
            _build_purpose_llm_client(provider="mistral", model="test", app_settings=s)

    def test_deepseek_uses_base_url_model_and_max_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="deepseek",
            deepseek_api_key="ds-test-key",
        )
        client = _build_purpose_llm_client(provider="deepseek", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "deepseek-chat"
        assert client.base_url == "https://api.deepseek.com/v1"
        assert client._max_tokens_field == "max_tokens"
        assert client._max_tokens == 4096

    def test_deepseek_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="deepseek",
            deepseek_api_key="ds-placeholder",
        )
        object.__setattr__(s, "deepseek_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is required"):
            _build_purpose_llm_client(provider="deepseek", model="test", app_settings=s)

    def test_zai_uses_base_url_model_and_max_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="zai",
            zai_api_key="zai-test-key",
        )
        client = _build_purpose_llm_client(provider="zai", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "glm-4.7-flash"
        assert client.base_url == "https://open.bigmodel.cn/api/paas/v4"
        assert client._max_tokens_field == "max_tokens"
        assert client._max_tokens == 4096

    def test_zai_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="zai",
            zai_api_key="zai-placeholder",
        )
        object.__setattr__(s, "zai_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="ZAI_API_KEY is required"):
            _build_purpose_llm_client(provider="zai", model="test", app_settings=s)

    def test_siliconflow_uses_base_url_model_and_max_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="siliconflow",
            siliconflow_api_key="sf-test-key",
        )
        client = _build_purpose_llm_client(provider="siliconflow", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "Qwen/Qwen3-8B"
        assert client.base_url == "https://api.siliconflow.com/v1"
        assert client._max_tokens_field == "max_tokens"
        assert client._max_tokens == 4096

    def test_siliconflow_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="siliconflow",
            siliconflow_api_key="sf-placeholder",
        )
        object.__setattr__(s, "siliconflow_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="SILICONFLOW_API_KEY is required"):
            _build_purpose_llm_client(provider="siliconflow", model="test", app_settings=s)

    def test_together_uses_base_url_model_and_max_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="together",
            together_api_key="together-test-key",
        )
        client = _build_purpose_llm_client(provider="together", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert client.base_url == "https://api.together.xyz/v1"
        assert client._max_tokens_field == "max_tokens"
        assert client._max_tokens == 4096

    def test_together_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="together",
            together_api_key="together-placeholder",
        )
        object.__setattr__(s, "together_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="TOGETHER_API_KEY is required"):
            _build_purpose_llm_client(provider="together", model="test", app_settings=s)

    def test_fireworks_uses_base_url_model_and_max_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="fireworks",
            fireworks_api_key="fw-test-key",
        )
        client = _build_purpose_llm_client(provider="fireworks", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "accounts/fireworks/models/llama-v3p3-70b-instruct"
        assert client.base_url == "https://api.fireworks.ai/inference/v1"
        assert client._max_tokens_field == "max_tokens"
        assert client._max_tokens == 4096

    def test_fireworks_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="fireworks",
            fireworks_api_key="fw-placeholder",
        )
        object.__setattr__(s, "fireworks_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="FIREWORKS_API_KEY is required"):
            _build_purpose_llm_client(provider="fireworks", model="test", app_settings=s)

    def test_fallback_provider_uses_own_default_model_not_purpose_override(self):
        """A fallback provider in a chain must resolve its own ``{provider}_model``
        (never the per-provider purpose override pinned for the primary), so a
        purpose model the fallback does not host is never forwarded."""
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="opencodego",
            opencodego_api_key="oc-go-test",
            opencodego_model="deepseek-v4-flash",
            groq_api_key="gsk-test",
            groq_model="openai/gpt-oss-20b",
            groq_answer_llm_model="mistral-large-latest",
        )

        # Purpose override suppressed (fallback) ⇒ resolves to provider default.
        fallback = _build_purpose_llm_client(
            provider="groq", model="", purpose="answer", app_settings=s, apply_purpose_model_override=False
        )
        assert fallback is not None
        assert fallback.model == "openai/gpt-oss-20b"

        # Purpose override applied (primary) ⇒ per-provider purpose override wins.
        primary = _build_purpose_llm_client(
            provider="groq", model="", purpose="answer", app_settings=s, apply_purpose_model_override=True
        )
        assert primary is not None
        assert primary.model == "mistral-large-latest"

    def test_openrouter_delegates(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="openrouter",
            llm_model="anthropic/claude-3.5-sonnet",
            openrouter_model="anthropic/claude-3.5-sonnet",
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
            nvidia_api_key="nvapi-test",
            nvidia_rpm_limit=80,
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


class TestBuildFallbackChain:
    def test_enrichment_empty_model_resolves_provider_default(self):
        from data_engineering_copilot.factory import build_llm_fallback_chain

        s = _make_settings(
            llm_provider="ollama",
            llm_model="llama3.2:3b",
            enrichment_llm_provider="gemini",
            gemini_api_key="sk-gemini-test",
        )
        client = build_llm_fallback_chain(
            purpose="enrichment",
            app_settings=s,
            purpose_provider="gemini",
            purpose_model="",
        )
        assert client is not None
        assert client.model == "gemini-2.5-flash"

    def test_answer_purpose_explicit_model(self):
        from data_engineering_copilot.factory import build_llm_fallback_chain

        s = _make_settings(
            llm_provider="ollama",
            llm_model="llama3.2:3b",
            answer_llm_provider="groq",
            answer_llm_model="llama-3.3-70b-versatile",
            groq_api_key="gsk-test",
        )
        client = build_llm_fallback_chain(
            purpose="answer",
            app_settings=s,
            purpose_provider="groq",
            purpose_model="llama-3.3-70b-versatile",
        )
        assert client is not None
        assert client.model == "llama-3.3-70b-versatile"

    def test_evaluation_empty_model_resolves_provider_default(self):
        from data_engineering_copilot.factory import build_llm_fallback_chain

        s = _make_settings(
            llm_provider="ollama",
            llm_model="llama3.2:3b",
            evaluation_llm_provider="groq",
            groq_api_key="gsk-test",
        )
        client = build_llm_fallback_chain(
            purpose="evaluation",
            app_settings=s,
            purpose_provider="groq",
            purpose_model="",
        )
        assert client is not None
        assert client.model == "openai/gpt-oss-20b"

    def test_empty_purpose_provider_uses_fallback_order(self):
        from data_engineering_copilot.factory import build_llm_fallback_chain

        s = _make_settings(llm_provider="ollama", llm_model="llama3.2:3b")
        # When purpose_provider is empty, it uses llm_fallback_order (adaptive)
        client = build_llm_fallback_chain(
            purpose="answer",
            app_settings=s,
            purpose_provider="",
            purpose_model="",
        )
        assert client is not None  # Returns adaptive chain from llm_fallback_order
        assert hasattr(client, "model")


class TestBuildEmbedder:
    def test_local_hf_default(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import (
            LocalSentenceTransformerEmbeddings,
        )

        s = _make_settings(embedding_provider="local-hf")
        embedder = build_embedder(s)
        assert isinstance(embedder, LocalSentenceTransformerEmbeddings)
        assert embedder.model_name == "nvidia/Nemotron-3-Embed-1B-BF16"

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

    def test_huggingface(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.huggingface_serverless_embeddings import (
            HuggingFaceServerlessEmbeddings,
        )

        s = _make_settings(
            embedding_provider="huggingface",
            huggingface_api_key="hf-test",
            huggingface_embedding_model="nvidia/Nemotron-3-Embed-1B-BF16",
        )
        embedder = build_embedder(s)
        assert isinstance(embedder, HuggingFaceServerlessEmbeddings)
        assert embedder.model_name == "nvidia/Nemotron-3-Embed-1B-BF16"
        assert s.get_embedding_dimension() == 2048

    def test_huggingface_missing_api_key_raises(self):
        from data_engineering_copilot.factory import build_embedder

        s = _make_settings(
            embedding_provider="huggingface",
            huggingface_api_key="hf-placeholder",
        )
        object.__setattr__(s, "huggingface_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="HF_TOKEN is required"):
            build_embedder(s)

    def test_nvidia(self):
        from data_engineering_copilot.factory import build_embedder
        from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
            OpenAICompatibleEmbeddings,
        )

        s = _make_settings(
            embedding_provider="nvidia",
            nvidia_embedding_model="nvidia/nemotron-3-embed-1b",
            nvidia_api_key="nvapi-test",
        )
        embedder = build_embedder(s)
        assert isinstance(embedder, OpenAICompatibleEmbeddings)
        assert embedder.model_name == "nvidia/nemotron-3-embed-1b"
        assert embedder.base_url == "https://integrate.api.nvidia.com/v1"

    def test_nvidia_missing_api_key_raises(self):
        from data_engineering_copilot.factory import build_embedder

        s = _make_settings(
            embedding_provider="nvidia",
            nvidia_api_key="nvapi-placeholder",
        )
        object.__setattr__(s, "nvidia_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
            build_embedder(s)


class TestBuildRerankFallbackChain:
    def test_no_cloud_keys_returns_none(self):
        from data_engineering_copilot.factory import build_rerank_fallback_chain

        s = _make_settings(llm_rerank_enabled=True)
        assert build_rerank_fallback_chain(s) is None

    def test_openrouter_key_builds_chain(self):
        from data_engineering_copilot.factory import build_rerank_fallback_chain
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain

        s = _make_settings(
            llm_rerank_enabled=True,
            openrouter_api_key="sk-or-v1-test",
        )
        chain = build_rerank_fallback_chain(s)
        assert isinstance(chain, ProviderFallbackChain)

    def test_all_three_keys_builds_ordered_chain(self):
        from data_engineering_copilot.factory import build_rerank_fallback_chain

        s = _make_settings(
            llm_rerank_enabled=True,
            openrouter_api_key="sk-or-v1-test",
            nvidia_api_key="nvapi-test",
            huggingface_api_key="hf-test",
        )
        chain = build_rerank_fallback_chain(s)
        assert chain is not None
        providers = [p.name for p in chain._config.providers]
        assert providers == ["openrouter", "nvidia", "huggingface"]

    def test_degraded_fallback_attached_when_local_reranker_provided(self):
        from data_engineering_copilot.factory import build_rerank_fallback_chain
        from data_engineering_copilot.infrastructure.rerank_clients import LocalRerankerClient

        class _StubReranker:
            model_name = "stub-crossencoder"

        s = _make_settings(
            llm_rerank_enabled=True,
            openrouter_api_key="sk-or-v1-test",
        )
        chain = build_rerank_fallback_chain(s, local_reranker=_StubReranker())
        assert chain is not None
        assert chain._config.degraded_fallback is not None
        assert isinstance(chain._config.degraded_fallback.client, LocalRerankerClient)

    def test_no_degraded_fallback_without_local_reranker(self):
        from data_engineering_copilot.factory import build_rerank_fallback_chain

        s = _make_settings(
            llm_rerank_enabled=True,
            openrouter_api_key="sk-or-v1-test",
        )
        chain = build_rerank_fallback_chain(s)
        assert chain is not None
        assert chain._config.degraded_fallback is None

    def test_rerank_fallback_order_skips_provider_without_key(self):
        from data_engineering_copilot.factory import build_rerank_fallback_chain

        s = _make_settings(
            llm_rerank_enabled=True,
            openrouter_api_key="sk-or-v1-test",
            nvidia_api_key="",
            huggingface_api_key="",
        )
        chain = build_rerank_fallback_chain(s)
        assert chain is not None
        providers = [p.name for p in chain._config.providers]
        assert providers == ["openrouter"]

    def test_settings_validation_rejects_unknown_rerank_provider(self):
        from pydantic import ValidationError

        s = _make_settings()
        object.__setattr__(s, "rerank_fallback_order", ["bogus"])
        with pytest.raises(ValidationError, match="unknown provider"):
            s.validate_all()


class TestRouterWiring:
    def test_router_wired_into_llm_chain(self):
        from typing import cast

        from data_engineering_copilot.factory import build_llm_fallback_chain
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain
        from data_engineering_copilot.infrastructure.provider_selector import (
            ProviderSelector,
            RedisRouterState,
        )

        s = _make_settings(
            llm_provider="openrouter",
            llm_model="openrouter/free",
            openrouter_api_key="sk-or-v1-test",
            groq_api_key="gsk-test",
            router_redis_sharing=False,
        )
        chain = build_llm_fallback_chain(purpose="answer", app_settings=s)
        assert isinstance(chain, ProviderFallbackChain)
        router = cast(ProviderSelector | None, chain._router)
        assert router is not None
        assert type(router.state) is not RedisRouterState
        assert router.config.purpose == "answer"

    def test_router_redis_sharing_enabled_by_default(self):
        from typing import cast

        from data_engineering_copilot.factory import build_llm_fallback_chain
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain
        from data_engineering_copilot.infrastructure.provider_selector import (
            ProviderSelector,
            RedisRouterState,
        )

        s = _make_settings(
            llm_provider="openrouter",
            llm_model="openrouter/free",
            openrouter_api_key="sk-or-v1-test",
            groq_api_key="gsk-test",
        )
        chain = build_llm_fallback_chain(purpose="answer", app_settings=s)
        assert isinstance(chain, ProviderFallbackChain)
        router = cast(ProviderSelector | None, chain._router)
        assert router is not None
        assert isinstance(router.state, RedisRouterState)

    def test_router_prefers_pinned_purpose_provider(self):
        from typing import cast

        from data_engineering_copilot.factory import build_llm_fallback_chain
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain
        from data_engineering_copilot.infrastructure.provider_selector import ProviderSelector

        s = _make_settings(
            llm_provider="openrouter",
            llm_model="openrouter/free",
            openrouter_api_key="sk-or-v1-test",
            groq_api_key="gsk-test",
            router_redis_sharing=False,
        )
        chain = build_llm_fallback_chain(
            purpose="answer",
            app_settings=s,
            purpose_provider="groq",
        )
        assert isinstance(chain, ProviderFallbackChain)
        router = cast(ProviderSelector | None, chain._router)
        assert router is not None
        assert router.config.preference_provider == "groq"
        assert router.config.preference_weight == s.router_purpose_preference_weight


class TestErrorCategorization:
    def test_embedding_401_model_not_supported_is_invalid_request(self):
        import httpx

        from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
        from data_engineering_copilot.factory import _categorize_embedding_error

        resp = httpx.Response(
            401,
            request=httpx.Request("POST", "http://example.com"),
            text='{"error": {"message": "Model X is not supported"}}',
        )
        exc = httpx.HTTPStatusError("401", request=resp.request, response=resp)
        err = _categorize_embedding_error(exc, "nvidia", "model-x")
        assert err.category == ProviderErrorCategory.INVALID_REQUEST

    def test_embedding_401_real_auth_stays_auth_error(self):
        import httpx

        from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
        from data_engineering_copilot.factory import _categorize_embedding_error

        resp = httpx.Response(
            401,
            request=httpx.Request("POST", "http://example.com"),
            text='{"error": {"message": "Invalid API key"}}',
        )
        exc = httpx.HTTPStatusError("401", request=resp.request, response=resp)
        err = _categorize_embedding_error(exc, "nvidia", "model")
        assert err.category == ProviderErrorCategory.AUTHENTICATION_ERROR

    def test_rerank_401_model_not_supported_is_invalid_request(self):
        import httpx

        from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
        from data_engineering_copilot.factory import _categorize_rerank_error

        resp = httpx.Response(
            403,
            request=httpx.Request("POST", "http://example.com"),
            text='ModelError: "rerank-model" is not supported',
        )
        exc = httpx.HTTPStatusError("403", request=resp.request, response=resp)
        err = _categorize_rerank_error(exc, "nvidia", "rerank-model")
        assert err.category == ProviderErrorCategory.INVALID_REQUEST

    def test_rerank_401_real_auth_stays_auth_error(self):
        import httpx

        from data_engineering_copilot.domain.exceptions import ProviderErrorCategory
        from data_engineering_copilot.factory import _categorize_rerank_error

        resp = httpx.Response(
            401,
            request=httpx.Request("POST", "http://example.com"),
            text='{"error": {"message": "Invalid API key"}}',
        )
        exc = httpx.HTTPStatusError("401", request=resp.request, response=resp)
        err = _categorize_rerank_error(exc, "nvidia", "model")
        assert err.category == ProviderErrorCategory.AUTHENTICATION_ERROR


# ---------------------------------------------------------------------------
# Batch 2 providers: ollama_cloud, llm7, agnes
# ---------------------------------------------------------------------------


class TestOllamaCloudProvider:
    def test_ollama_cloud_uses_cloud_base_url_and_api_key(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="ollama_cloud",
            ollama_cloud_api_key="oc-key-123",
            _test_allow_non_ollama=True,
        )
        client = _build_purpose_llm_client(provider="ollama_cloud", model="test", app_settings=s)
        assert isinstance(client, LLMClient)
        assert "ollama.com" in client.base_url
        assert client.api_key == "oc-key-123"

    def test_ollama_cloud_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="ollama_cloud",
            llm_fallback_order=["ollama_cloud"],
            ollama_cloud_api_key="placeholder",
        )
        object.__setattr__(s, "ollama_cloud_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="OLLAMA_API_KEY is required"):
            _build_purpose_llm_client(provider="ollama_cloud", model="test", app_settings=s)


class TestLLM7Provider:
    def test_llm7_uses_correct_base_url_and_model(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="llm7",
            llm7_api_key="llm7-key-123",
            _test_allow_non_ollama=True,
        )
        client = _build_purpose_llm_client(provider="llm7", model="test", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.base_url == "https://api.llm7.io/v1"
        assert client.model == "test"
        assert client.api_key == "llm7-key-123"

    def test_llm7_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="llm7",
            llm_fallback_order=["llm7"],
            llm7_api_key="placeholder",
        )
        object.__setattr__(s, "llm7_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="LLM7_API_KEY is required"):
            _build_purpose_llm_client(provider="llm7", model="test", app_settings=s)


class TestAgnesProvider:
    def test_agnes_uses_correct_base_url_and_model(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="agnes",
            agnes_api_key="agnes-key-123",
            _test_allow_non_ollama=True,
        )
        client = _build_purpose_llm_client(provider="agnes", model="test", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.base_url == "https://apihub.agnes-ai.com/v1"
        assert client.model == "test"
        assert client.api_key == "agnes-key-123"

    def test_agnes_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="agnes",
            llm_fallback_order=["agnes"],
            agnes_api_key="placeholder",
        )
        object.__setattr__(s, "agnes_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="AGNES_API_KEY is required"):
            _build_purpose_llm_client(provider="agnes", model="test", app_settings=s)


class TestHelyxProvider:
    def test_helyx_uses_correct_base_url_and_model(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="helyx",
            helyx_api_key="hx-key-123",
            _test_allow_non_ollama=True,
        )
        client = _build_purpose_llm_client(provider="helyx", model="test", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.base_url == "https://helyxai.space/v1"
        assert client.model == "test"
        assert client.api_key == "hx-key-123"

    def test_helyx_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="helyx",
            llm_fallback_order=["helyx"],
            helyx_api_key="placeholder",
        )
        object.__setattr__(s, "helyx_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="HELYX_API_KEY is required"):
            _build_purpose_llm_client(provider="helyx", model="test", app_settings=s)


class TestAnyAPIProvider:
    def test_anyapi_uses_correct_base_url_and_model(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="anyapi",
            anyapi_api_key="anyapi-key-123",
            _test_allow_non_ollama=True,
        )
        client = _build_purpose_llm_client(provider="anyapi", model="test", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.base_url == "https://api.anyapi.ai/v1"
        assert client.model == "test"
        assert client.api_key == "anyapi-key-123"

    def test_anyapi_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="anyapi",
            llm_fallback_order=["anyapi"],
            anyapi_api_key="placeholder",
        )
        object.__setattr__(s, "anyapi_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="ANYAPI_API_KEY is required"):
            _build_purpose_llm_client(provider="anyapi", model="test", app_settings=s)


def test_ollama_purpose_budget_respected_over_num_predict():
    """P5.1: degraded/ollama judge calls were capped at ollama_num_predict(512),
    truncating RAGAS judges (LLMDidNotFinish -> NaN). Purpose budget must win."""
    from data_engineering_copilot.factory import _build_purpose_llm_client

    s = _make_settings(llm_provider="ollama")
    client = _build_purpose_llm_client(provider="ollama", model="", purpose="evaluation", app_settings=s)
    assert client._max_tokens == 1536  # purpose_max_tokens["evaluation"], not 512


class TestBaiProvider:
    def test_bai_uses_base_url_model_and_max_tokens_field(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client
        from data_engineering_copilot.infrastructure.llm_client import LLMClient

        s = _make_settings(
            llm_provider="bai",
            bai_api_key="sk-bai-test",
            _test_allow_non_ollama=True,
        )
        client = _build_purpose_llm_client(provider="bai", model="", purpose="answer", app_settings=s)
        assert isinstance(client, LLMClient)
        assert client.model == "glm-5.3-flash"
        assert client.base_url == "https://api.b.ai/v1"
        assert client._max_tokens_field == "max_tokens"
        assert client._max_tokens == 4096

    def test_bai_missing_api_key_raises(self):
        from data_engineering_copilot.factory import _build_purpose_llm_client

        s = _make_settings(
            llm_provider="bai",
            llm_fallback_order=["bai"],
            bai_api_key="placeholder",
        )
        object.__setattr__(s, "bai_api_key", SecretStr(""))
        with pytest.raises(ValueError, match="BAI_API_KEY is required"):
            _build_purpose_llm_client(provider="bai", model="test", app_settings=s)
