import json

from data_engineering_copilot.config.settings import load_documentation_sources


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
