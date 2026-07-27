import json

from data_engineering_copilot.config.settings import AppSettings, load_documentation_sources


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
    settings = AppSettings()

    assert settings.logging_enabled is True


def test_app_settings_hybrid_search_defaults() -> None:
    settings = AppSettings()
    assert settings.hybrid_search_enabled is True
    assert settings.hybrid_rrf_k == 60
    assert settings.context_compression_enabled is False
    assert settings.max_context_tokens == 4096
    assert settings.query_rewrite_enabled is True
    assert settings.groundedness_enabled is True


def test_app_settings_hybrid_search_overridable() -> None:
    settings = AppSettings(
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
    settings = AppSettings(code_llm_provider="", code_llm_model="")
    assert settings.code_llm_provider == ""
    assert settings.code_llm_model == ""
    assert settings.nvidia_nim_rpm_limit == 40
    assert settings.nvidia_nim_base_url == "https://integrate.api.nvidia.com/v1"


def test_code_llm_overridable() -> None:
    settings = AppSettings(
        code_llm_provider="nvidia",
        code_llm_model="qwen/qwen2.5-coder-32b-instruct",
        nvidia_nim_api_key="nvapi-test",
        nvidia_nim_rpm_limit=80,
        nvidia_nim_base_url="https://custom.nvidia.com/v1",
    )
    assert settings.code_llm_provider == "nvidia"
    assert settings.code_llm_model == "qwen/qwen2.5-coder-32b-instruct"
    assert settings.nvidia_nim_rpm_limit == 80
    assert settings.nvidia_nim_base_url == "https://custom.nvidia.com/v1"


def test_nvidia_embedding_dimension_default() -> None:
    settings = AppSettings()
    assert settings.nvidia_embedding_dimension == 2048
    assert settings.nvidia_embedding_model == "nvidia/nemotron-3-embed-1b"


def test_local_embedding_dimension_default() -> None:
    settings = AppSettings()
    assert settings.local_embedding_dimension == 768


def test_nvidia_nim_rpd_limit_default() -> None:
    settings = AppSettings()
    assert settings.nvidia_nim_rpd_limit == 1000


def test_embedding_provider_nvidia_missing_api_key_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="NVIDIA_NIM_API_KEY is required"):
        AppSettings(
            embedding_provider="nvidia",
            nvidia_nim_api_key="",
        )
