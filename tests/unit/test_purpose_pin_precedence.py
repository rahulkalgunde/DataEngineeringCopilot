"""Behavioral pin: {purpose}_llm_provider settings pin beats catalog auto-order.

Regression for the silent mis-route found 2026-08-23: eval-generation's default
judge built `build_llm_fallback_chain(purpose="evaluation")` WITHOUT passing
purpose_provider, so catalog_auto_order demoted the pinned opencodezen provider
and the free-window model never received a single call. The chain config must
derive the pin from settings when the caller omits it.
"""

from __future__ import annotations

import json
from datetime import UTC

import pytest

from data_engineering_copilot.factory import build_llm_fallback_chain
from data_engineering_copilot.infrastructure.llm_client import LLMClient


def _provider_names(chain: object) -> list[str]:
    assert not isinstance(chain, LLMClient), "expected a multi-provider chain, got single LLMClient"
    return [p.name for p in chain._config.providers]  # type: ignore[attr-defined]


def _catalog(tmp_path, order_by_purpose: dict[str, list[str]]):
    cat = tmp_path / "provider_catalog.json"
    from datetime import datetime

    cat.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "probes": [],
                "recommended_fallback_order": order_by_purpose,
            }
        )
    )
    return cat


@pytest.mark.unit
class TestPurposePinBeatsCatalog:
    def test_settings_pin_wins_over_catalog_auto_order(self, tmp_path):
        from tests.conftest import make_settings

        s = make_settings(
            catalog_auto_order=True,
            provider_catalog_path=_catalog(tmp_path, {"evaluation": ["cerebras", "groq"]}),
            evaluation_llm_provider="opencodezen",
            opencodezen_model="x-preview-f-free",
            opencodezen_api_key="oc-test",
            cerebras_api_key="cb-test",
            _test_allow_non_ollama=True,
        )
        chain = build_llm_fallback_chain(purpose="evaluation", app_settings=s)
        names = _provider_names(chain)
        assert names[0] == "opencodezen"
        assert chain._config.providers[0].client.model == "x-preview-f-free"  # type: ignore[attr-defined]

    def test_no_pin_still_uses_catalog_order(self, tmp_path):
        from tests.conftest import make_settings

        s = make_settings(
            catalog_auto_order=True,
            provider_catalog_path=_catalog(tmp_path, {"evaluation": ["cerebras", "groq"]}),
            evaluation_llm_provider="",
            cerebras_api_key="cb-test",
            _test_allow_non_ollama=True,
        )
        chain = build_llm_fallback_chain(purpose="evaluation", app_settings=s)
        assert _provider_names(chain)[0] == "cerebras"


@pytest.mark.parametrize(
    "purpose",
    ["answer", "rewrite", "groundedness", "intent", "enrichment", "evaluation", "code"],
)
def test_every_purpose_pin_wins_over_catalog(purpose, tmp_path):
    """Property (G4): for EVERY purpose, a settings pin must head the chain even
    when catalog_auto_order would order differently — the 2026-08-24 mis-route
    class must be impossible on any path."""
    from tests.conftest import make_settings

    s = make_settings(
        catalog_auto_order=True,
        provider_catalog_path=_catalog(tmp_path, {purpose: ["cerebras"]}),
        **{
            f"{purpose}_llm_provider": "opencodezen",
            "opencodezen_model": "x-preview-f-free",
            "opencodezen_api_key": "oc-test",
            "cerebras_api_key": "cb-test",
            "_test_allow_non_ollama": True,
        },
    )
    chain = build_llm_fallback_chain(purpose=purpose, app_settings=s)
    assert _provider_names(chain)[0] == "opencodezen"
