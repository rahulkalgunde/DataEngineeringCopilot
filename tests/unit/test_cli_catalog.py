"""Tests for cli_catalog.py."""

from __future__ import annotations

from data_engineering_copilot.cli_catalog import _build_recommended, _catalog_probe_targets
from data_engineering_copilot.services.provider_catalog import CatalogModel, ProbeEntry


def _make_model(provider: str = "groq", model: str = "test-model") -> CatalogModel:
    return CatalogModel(provider=provider, model=model)


def _make_probe(provider: str = "groq", model: str = "test-model", status: str = "OK") -> ProbeEntry:
    return ProbeEntry(provider=provider, model=model, status=status)


class TestCatalogProbeTargets:
    def test_returns_all_when_no_filter(self) -> None:
        models = [_make_model("groq"), _make_model("nvidia")]
        result = _catalog_probe_targets(models, None)
        assert result == models

    def test_filters_by_provider(self) -> None:
        models = [_make_model("groq"), _make_model("nvidia"), _make_model("openrouter")]
        result = _catalog_probe_targets(models, {"groq", "nvidia"})
        assert len(result) == 2
        assert all(m.provider in ("groq", "nvidia") for m in result)

    def test_filter_is_case_insensitive(self) -> None:
        models = [_make_model("Groq"), _make_model("NVIDIA")]
        result = _catalog_probe_targets(models, {"groq"})
        assert len(result) == 1
        assert result[0].provider == "Groq"

    def test_empty_list(self) -> None:
        assert _catalog_probe_targets([], {"groq"}) == []


class TestBuildRecommended:
    def test_returns_all_purposes(self) -> None:
        probes = [_make_probe()]
        result = _build_recommended(probes)
        assert "global" in result
        assert "answer" in result
        assert "rewrite" in result
        assert "groundedness" in result
        assert "intent" in result
        assert "enrichment" in result
        assert "evaluation" in result
        assert "code" in result

    def test_empty_probes(self) -> None:
        result = _build_recommended([])
        assert all(v == [] for v in result.values())
