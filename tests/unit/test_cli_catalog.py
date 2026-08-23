"""Tests for cli_catalog probe-target filtering and SKIP-path probe entries."""

from __future__ import annotations

import pytest

from data_engineering_copilot.cli_catalog import _catalog_probe_targets, _probe_one
from data_engineering_copilot.services.provider_catalog import CatalogModel

pytestmark = pytest.mark.unit


def _model(provider: str, model: str = "m1") -> CatalogModel:
    return CatalogModel(provider=provider, model=model)


class TestCatalogProbeTargets:
    def test_no_filter_returns_all(self):
        models = [_model("groq"), _model("zai")]
        assert _catalog_probe_targets(models, None) == models

    def test_empty_filter_set_returns_all(self):
        models = [_model("groq")]
        assert _catalog_probe_targets(models, set()) == models

    def test_filters_case_insensitively_on_provider(self):
        models = [_model("groq", "a"), _model("ZAI", "b"), _model("openai", "c")]
        out = _catalog_probe_targets(models, {"groq", "zai"})
        assert [m.model for m in out] == ["a", "b"]


class TestProbeOneSkipPaths:
    async def test_skip_when_probe_backend_unavailable(self, monkeypatch):
        import data_engineering_copilot.cli_catalog as cc

        monkeypatch.setattr(cc, "_probe_llm_target", None)
        entry = await _probe_one(_model("groq"), app_settings=None, prompt="p", timeout=1.0)  # type: ignore[arg-type]
        assert entry.status == "SKIP"
        assert "unavailable" in entry.message
        assert entry.provider == "groq"
        assert entry.kind == "llm"

    async def test_skip_local_hf_without_probing(self):
        # local-hf is embedding-only; must never be LLM-probed even when the
        # probe backend exists (no network call possible here either way).
        entry = await _probe_one(_model("local-hf"), app_settings=None, prompt="p", timeout=1.0)  # type: ignore[arg-type]
        assert entry.status == "SKIP"
        assert "embedding-only" in entry.message


class TestProbeOneDelegates:
    async def test_result_fields_copied_from_probe_result(self, monkeypatch):
        import data_engineering_copilot.cli_catalog as cc
        from data_engineering_copilot.cli_llm_probe import ProbeResult, ProbeTarget

        target = ProbeTarget(kind="llm", provider="groq", model="m1", purpose=None)

        async def fake_probe(t, app_settings, prompt, timeout):
            assert t == target
            return ProbeResult(target=t, status="OK", latency_ms=12.5, http_status=200, message="hi")

        monkeypatch.setattr(cc, "_probe_llm_target", fake_probe)
        monkeypatch.setattr(cc, "ProbeTarget", ProbeTarget)
        entry = await _probe_one(_model("groq"), app_settings=None, prompt="p", timeout=1.0)  # type: ignore[arg-type]
        assert entry.status == "OK"
        assert entry.latency_ms == 12.5
        assert entry.http_status == 200
        assert entry.message == "hi"


@pytest.mark.parametrize(
    ("provider", "status"),
    [("local-hf", "SKIP"), ("groq", "OK")],
)
async def test_parametrized_dispositions(provider, status, monkeypatch):
    if provider == "groq":
        import data_engineering_copilot.cli_catalog as cc
        from data_engineering_copilot.cli_llm_probe import ProbeResult

        async def fake_probe(t, app_settings, prompt, timeout):
            return ProbeResult(target=t, status="OK")

        monkeypatch.setattr(cc, "_probe_llm_target", fake_probe)
    entry = await _probe_one(_model(provider), app_settings=None, prompt="p", timeout=1.0)  # type: ignore[arg-type]
    assert entry.status == status
