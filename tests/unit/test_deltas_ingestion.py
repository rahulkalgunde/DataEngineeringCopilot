"""Dry-run validation tests for Delta Lake documentation ingestion.

Verifies the versioned Delta source (docs.delta.io@4.4.0) is correctly configured
in documentation_sources.json and integrates with the crawl state pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from data_engineering_copilot.config.settings import DocumentationSource, load_documentation_sources


def _get_documentation_sources_path() -> Path:
    """Return the path to the documentation_sources.json config file."""
    return Path(__file__).parent.parent.parent / "data_engineering_copilot" / "config" / "documentation_sources.json"


class TestDeltaSourceConfiguration:
    """Tests for the Delta Lake documentation source configuration."""

    def test_delta_source_exists_in_config(self) -> None:
        """The Delta Lake source entry must exist in documentation_sources.json."""
        config_path = _get_documentation_sources_path()
        with config_path.open("r", encoding="utf-8") as file:
            sources = json.load(file)

        delta_sources = [s for s in sources if s.get("name") == "Delta Lake Documentation"]
        assert len(delta_sources) == 1, "Exactly one 'Delta Lake Documentation' source must exist"

    def test_delta_source_has_correct_fields(self) -> None:
        """The Delta source must have all required fields with correct values."""
        config_path = _get_documentation_sources_path()
        with config_path.open("r", encoding="utf-8") as file:
            sources = json.load(file)

        delta_source = next(s for s in sources if s.get("name") == "Delta Lake Documentation")

        # Verify required fields
        assert delta_source["name"] == "Delta Lake Documentation"
        assert isinstance(delta_source["start_urls"], list)
        assert len(delta_source["start_urls"]) >= 1
        assert delta_source["start_urls"][0] == "https://docs.delta.io/4.4.0/"

        assert isinstance(delta_source["allowed_domains"], list)
        assert "docs.delta.io" in delta_source["allowed_domains"]

        assert isinstance(delta_source["url_prefixes"], list)
        assert "https://docs.delta.io/4.4.0/" in delta_source["url_prefixes"]

    def test_delta_source_loads_via_settings_loader(self) -> None:
        """The Delta source must parse correctly through load_documentation_sources()."""
        config_path = _get_documentation_sources_path()
        sources = load_documentation_sources(config_path)

        delta_sources = [s for s in sources if s.name == "Delta Lake Documentation"]
        assert len(delta_sources) == 1

        delta_source = delta_sources[0]
        assert isinstance(delta_source, DocumentationSource)
        assert delta_source.name == "Delta Lake Documentation"
        assert delta_source.start_urls == ("https://docs.delta.io/4.4.0/",)
        assert delta_source.allowed_domains == ("docs.delta.io",)
        assert delta_source.url_prefixes == ("https://docs.delta.io/4.4.0/",)
        assert delta_source.priority == 1  # default

    def test_delta_source_url_prefix_matches_start_url(self) -> None:
        """The url_prefixes must be a prefix of start_urls for correct crawl scoping."""
        config_path = _get_documentation_sources_path()
        with config_path.open("r", encoding="utf-8") as file:
            sources = json.load(file)

        delta_source = next(s for s in sources if s.get("name") == "Delta Lake Documentation")
        start_url = delta_source["start_urls"][0]
        prefix = delta_source["url_prefixes"][0]
        assert start_url.startswith(prefix), "start_url must start with url_prefix"


class TestDeltaSourceCrawlStateIntegration:
    """Mocked integration tests for Delta source with crawl state pipeline."""
