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
