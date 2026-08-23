"""Unit tests for provider catalog loader/filter/ranker."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest

from data_engineering_copilot.services.provider_catalog import (
    CatalogModel,
    ProbeEntry,
    ProviderCatalog,
    compute_recommended_order,
    filter_rag_suitable,
    get_catalog_fallback_order,
    is_catalog_stale,
    is_rag_suitable,
    load_free_tier_models,
    load_provider_catalog,
    serialize_catalog,
)


def _write_free_tier(tmp: Path, models: list[dict]) -> Path:
    p = tmp / "free_tier_models.json"
    p.write_text(json.dumps({"version": "1.0", "models": models}), encoding="utf-8")
    return p


def test_load_free_tier_models_ok(tmp_path):
    p = _write_free_tier(
        tmp_path,
        [
            {
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "tier": "free_forever",
                "context_window": 131072,
                "supports_structured_output": True,
            },
            {
                "provider": "zai",
                "model": "glm-4.7-flash",
                "tier": "free_forever",
                "context_window": 128000,
                "supports_structured_output": True,
            },
        ],
    )
    models = load_free_tier_models(p)
    assert len(models) == 2
    assert models[0].provider == "groq"


def test_load_free_tier_rejects_non_free(tmp_path):
    p = _write_free_tier(
        tmp_path, [{"provider": "together", "model": "meta-llama/Llama-3.3-70B", "tier": "free_credit"}]
    )
    with pytest.raises(ValueError, match="free_forever"):
        load_free_tier_models(p)


def test_load_free_tier_duplicate_rejected(tmp_path):
    p = _write_free_tier(
        tmp_path,
        [
            {"provider": "groq", "model": "openai/gpt-oss-20b", "tier": "free_forever"},
            {"provider": "groq", "model": "openai/gpt-oss-20b", "tier": "free_forever"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_free_tier_models(p)


def test_is_rag_suitable():
    m = CatalogModel(
        provider="groq",
        model="openai/gpt-oss-20b",
        supports_structured_output=True,
        context_window=131072,
        rag_suitable=True,
    )
    assert is_rag_suitable(m, "answer") is True
    assert is_rag_suitable(m, None) is True
    m2 = CatalogModel(
        provider="siliconflow",
        model="Qwen/Qwen3-8B",
        supports_structured_output=False,
        context_window=8192,
        rag_suitable=True,
    )
    assert is_rag_suitable(m2, "answer") is False
    assert is_rag_suitable(m2, "rewrite") is True
    m3 = CatalogModel(
        provider="groq", model="x", supports_structured_output=True, context_window=4000, rag_suitable=True
    )
    assert is_rag_suitable(m3) is False


def test_filter_rag_suitable_per_purpose():
    models = [
        CatalogModel(provider="groq", model="a", supports_structured_output=True, rag_suitable=True),
        CatalogModel(provider="siliconflow", model="b", supports_structured_output=False, rag_suitable=True),
    ]
    assert len(filter_rag_suitable(models, "answer")) == 1
    assert len(filter_rag_suitable(models, "rewrite")) == 2


def test_compute_recommended_order_fastest_per_provider():
    probes = [
        ProbeEntry(
            provider="groq",
            model="openai/gpt-oss-20b",
            status="OK",
            latency_ms=120,
            rag_suitable=True,
            context_window=131072,
            supports_structured_output=True,
        ),
        ProbeEntry(
            provider="groq",
            model="openai/gpt-oss-20b-2",
            status="OK",
            latency_ms=80,
            rag_suitable=True,
            context_window=131072,
            supports_structured_output=True,
        ),
        ProbeEntry(
            provider="zai",
            model="glm-4.7-flash",
            status="OK",
            latency_ms=200,
            rag_suitable=True,
            context_window=128000,
            supports_structured_output=True,
        ),
        ProbeEntry(
            provider="cerebras",
            model="gemma-4-31b",
            status="FAIL",
            latency_ms=50,
            rag_suitable=True,
            context_window=8192,
            supports_structured_output=True,
        ),
        ProbeEntry(
            provider="ollama",
            model="llama3.2:3b",
            status="OK",
            latency_ms=10,
            rag_suitable=True,
            context_window=8192,
            supports_structured_output=True,
        ),
    ]
    order = compute_recommended_order(probes)
    # groq fastest 80, zai 200; ollama excluded, cerebras FAIL excluded
    assert order == ["groq", "zai"]


def test_compute_recommended_order_purpose_filter():
    probes = [
        ProbeEntry(
            provider="groq",
            model="a",
            status="OK",
            latency_ms=10,
            rag_suitable=True,
            context_window=131072,
            supports_structured_output=True,
        ),
        ProbeEntry(
            provider="siliconflow",
            model="b",
            status="OK",
            latency_ms=5,
            rag_suitable=True,
            context_window=8192,
            supports_structured_output=False,
        ),
    ]
    assert compute_recommended_order(probes, purpose="answer") == ["groq"]
    assert compute_recommended_order(probes, purpose="rewrite") == ["siliconflow", "groq"]


def test_is_catalog_stale():
    from datetime import datetime, timedelta

    cat = ProviderCatalog(generated_at=datetime.now(UTC).isoformat())
    assert is_catalog_stale(cat, stale_days=7) is False
    old = (datetime.now(UTC) - timedelta(days=8)).isoformat()
    cat2 = ProviderCatalog(generated_at=old)
    assert is_catalog_stale(cat2, stale_days=7) is True
    assert is_catalog_stale(None) is True


def test_get_catalog_fallback_order():
    cat = ProviderCatalog(
        generated_at="2026-08-23T00:00:00+00:00",
        recommended_fallback_order={"global": ["groq", "zai"], "answer": ["zai", "groq"]},
    )
    assert get_catalog_fallback_order("answer", cat) == ["zai", "groq"]
    assert get_catalog_fallback_order("rewrite", cat) == ["groq", "zai"]
    assert get_catalog_fallback_order("answer", None) is None


def test_serialize_roundtrip(tmp_path):
    cat = ProviderCatalog(
        generated_at="2026-08-23T00:00:00+00:00",
        probes=[
            ProbeEntry(
                provider="groq",
                model="openai/gpt-oss-20b",
                status="OK",
                latency_ms=100,
                rag_suitable=True,
                context_window=131072,
                supports_structured_output=True,
            )
        ],
        recommended_fallback_order={"global": ["groq"]},
    )
    data = serialize_catalog(cat)
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_provider_catalog(p)
    assert loaded is not None
    assert loaded.probes[0].provider == "groq"
    assert loaded.recommended_fallback_order["global"] == ["groq"]


def test_load_real_free_tier_file():
    from data_engineering_copilot.config.settings import PROJECT_ROOT

    p = PROJECT_ROOT / "data_engineering_copilot" / "config" / "free_tier_models.json"
    models = load_free_tier_models(p)
    assert len(models) >= 10
    assert all(m.tier == "free_forever" for m in models)
    # all providers deduped per model
    keys = [(m.provider, m.model) for m in models]
    assert len(keys) == len(set(keys))


def test_catalog_integration_with_factory(tmp_path):
    """Factory uses catalog order when catalog_auto_order=True and catalog fresh."""
    import json
    from datetime import datetime

    from data_engineering_copilot.factory import get_catalog_fallback_order as factory_order
    from tests.conftest import make_settings

    catalog_path = tmp_path / "catalog.json"
    catalog_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "probes": [
            {
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "status": "OK",
                "latency_ms": 10,
                "rag_suitable": True,
                "context_window": 131072,
                "supports_structured_output": True,
            },
            {
                "provider": "zai",
                "model": "glm-4.7-flash",
                "status": "OK",
                "latency_ms": 20,
                "rag_suitable": True,
                "context_window": 128000,
                "supports_structured_output": True,
            },
        ],
        "recommended_fallback_order": {"global": ["zai", "groq"], "answer": ["groq", "zai"]},
    }
    catalog_path.write_text(json.dumps(catalog_data), encoding="utf-8")
    s = make_settings(
        catalog_auto_order=True,
        provider_catalog_path=catalog_path,
        catalog_stale_days=7,
        _test_allow_non_ollama=True,
        groq_api_key="p",
        zai_api_key="p",
    )
    assert factory_order("answer", s) == ["groq", "zai"]
    assert factory_order("global", s) == ["zai", "groq"]
