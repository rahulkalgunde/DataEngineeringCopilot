import json
import os

import pytest
from pydantic import SecretStr

from data_engineering_copilot.config.settings import AppSettings, load_documentation_sources
from tests.conftest import make_settings


def test_load_documentation_sources_from_json(tmp_path):
    config_path = tmp_path / "documentation_sources.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": "Example Docs",
                    "start_urls": ["https://example.com/docs/"],
                    "allowed_domains": ["example.com"],
                    "url_prefixes": ["https://example.com/docs/"],
                }
            ]
        ),
        encoding="utf-8",
    )

    sources = load_documentation_sources(config_path)

    assert len(sources) == 1
    assert sources[0].name == "Example Docs"
    assert sources[0].start_urls == ("https://example.com/docs/",)
    assert sources[0].allowed_domains == ("example.com",)


def test_app_settings_default_logging_enabled() -> None:
    settings = make_settings()

    assert settings.logging_enabled is True


def test_app_settings_hybrid_search_defaults() -> None:
    settings = make_settings()
    assert settings.hybrid_search_enabled is True
    assert settings.hybrid_rrf_k == 60
    assert settings.context_compression_enabled is False
    assert settings.max_context_tokens == 4096
    assert settings.query_rewrite_enabled is True
    assert settings.groundedness_enabled is True


def test_app_settings_hybrid_search_overridable() -> None:
    settings = make_settings(
        hybrid_search_enabled=False,
        hybrid_rrf_k=100,
        context_compression_enabled=True,
        max_context_tokens=8192,
    )
    assert settings.hybrid_search_enabled is False
    assert settings.hybrid_rrf_k == 100
    assert settings.context_compression_enabled is True
    assert settings.max_context_tokens == 8192


def test_code_llm_defaults_empty() -> None:
    settings = make_settings()
    assert settings.code_llm_provider == ""
    assert settings.code_llm_model == ""
    assert settings.nvidia_rpm_limit == 40
    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"


def test_code_llm_overridable() -> None:
    settings = make_settings(
        code_llm_provider="nvidia",
        code_llm_model="qwen/qwen2.5-coder-32b-instruct",
        nvidia_api_key="nvapi-test",
        nvidia_rpm_limit=80,
        nvidia_base_url="https://custom.nvidia.com/v1",
        _test_allow_non_ollama=True,
    )
    assert settings.code_llm_provider == "nvidia"
    assert settings.code_llm_model == "qwen/qwen2.5-coder-32b-instruct"
    assert settings.nvidia_rpm_limit == 80
    assert settings.nvidia_base_url == "https://custom.nvidia.com/v1"


def test_embedding_model_dimensions_lookup() -> None:
    settings = make_settings()
    assert settings.get_embedding_dimension() == 768


def test_get_embedding_dimension_nvidia() -> None:
    s = make_settings(
        embedding_provider="nvidia",
        nvidia_embedding_model="nvidia/nemotron-3-embed-1b",
        nvidia_api_key=SecretStr("nvapi-test"),
        _test_allow_non_ollama=True,
    )
    assert s.get_embedding_dimension() == 2048


def test_get_embedding_dimension_unknown_model() -> None:
    s = make_settings(embedding_model_name="unknown-model")
    assert s.get_embedding_dimension() == s.default_embedding_dimension


def test_nvidia_nim_rpd_limit_default() -> None:
    settings = make_settings()
    assert settings.nvidia_rpd_limit == 1000


def test_embedding_provider_nvidia_missing_api_key_raises() -> None:

    with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
        make_settings(
            embedding_provider="nvidia",
            nvidia_api_key="",
            _test_allow_non_ollama=True,
        )


def test_api_key_field_hermetic_default_empty() -> None:
    settings = make_settings()
    assert settings.api_key.get_secret_value() == ""


def test_api_key_overridable() -> None:
    settings = make_settings(api_key="sk-test-key")
    assert settings.api_key.get_secret_value() == "sk-test-key"


def test_api_key_not_in_str() -> None:
    settings = make_settings(api_key="sk-ultra-secret-42")
    rendered = str(settings)
    assert "sk-ultra-secret-42" not in rendered


def test_env_local_overrides_env(tmp_path) -> None:
    env_file = tmp_path / ".env"
    local_file = tmp_path / ".env.local"
    env_file.write_text("COLLECTION_NAME=from_env\n", encoding="utf-8")
    local_file.write_text("COLLECTION_NAME=from_env_local\n", encoding="utf-8")

    previous_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        kwargs: dict = {
            "_env_file": (".env", ".env.local"),
            "skip_provider_check": True,
            "llm_provider": "ollama",
            "llm_model": "llama3.2:3b",
            "embedding_provider": "ollama",
            "embedding_model_name": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
        }
        settings = AppSettings(**kwargs)
    finally:
        os.chdir(previous_cwd)

    assert settings.collection_name == "from_env_local"


def test_no_duplicate_embed_concurrency() -> None:
    settings = make_settings()
    assert settings.model_fields["embed_concurrency"].default == 1


def test_validate_all_passes_on_defaults() -> None:
    settings = make_settings()
    settings.validate_all()  # must not raise


def test_validate_all_detects_conflicts() -> None:
    from pydantic import ValidationError

    def _make_bad(**overrides) -> AppSettings:
        from tests.conftest import make_settings

        base = make_settings()
        kwargs: dict = {
            field: getattr(base, field)
            for field in base.model_fields
            if field not in {"sources", "skip_provider_check"}
        }
        kwargs.update(overrides)
        return AppSettings(**kwargs)

    bad = _make_bad(reranker_top_k=20, retrieval_top_k=5)
    with pytest.raises(ValidationError, match="reranker_top_k"):
        bad.validate_all()

    negative = _make_bad(max_pages_per_source=-1)
    with pytest.raises(ValidationError, match="max_pages_per_source"):
        negative.validate_all()

    no_provider = _make_bad(llm_provider="", embedding_provider="")
    with pytest.raises(ValidationError, match="provider"):
        no_provider.validate_all()
