from data_engineering_copilot.infrastructure.health_checks import (
    build_health_report,
    build_status_report,
    probe_embedding_chain,
    probe_llm_chain,
)
from tests.conftest import make_settings


def test_probe_embedding_chain_nvidia_primary():
    s = make_settings(
        embedding_provider="nvidia",
        embedding_fallback_order=["nvidia", "openrouter", "huggingface", "local-hf"],
        nvidia_api_key="nv-test",
        openrouter_api_key="or-test",
        huggingface_api_key="hf-test",
        _test_allow_non_ollama=True,
    )
    chain = probe_embedding_chain(s)
    assert [c.provider for c in chain] == ["nvidia", "openrouter", "huggingface", "local-hf"]
    assert chain[0].has_key is True
    assert chain[0].model == s.nvidia_embedding_model
    assert chain[3].is_local is True
    assert chain[3].has_key is True  # local-hf always has_key


def test_probe_embedding_chain_missing_keys():
    s = make_settings(
        embedding_fallback_order=["nvidia", "openrouter", "local-hf"],
        nvidia_api_key="",
        openrouter_api_key="",
    )
    chain = probe_embedding_chain(s)
    assert chain[0].has_key is False
    assert chain[1].has_key is False
    assert chain[2].is_local is True


def test_probe_llm_chain_order_and_keys():
    s = make_settings(
        llm_provider="groq",
        llm_fallback_order=["groq", "cerebras", "ollama"],
        groq_api_key="g-test",
        cerebras_api_key="",
        _test_allow_non_ollama=True,
    )
    chain = probe_llm_chain(s)
    assert [c.provider for c in chain] == ["groq", "cerebras", "ollama"]
    assert chain[0].has_key is True
    assert chain[1].has_key is False
    assert chain[2].is_local is True


def test_build_health_report_local_only():
    s = make_settings()
    report = build_health_report(s)
    # conftest defaults to local-hf + ollama
    assert report.docker is not None
    assert report.qdrant is not None
    assert report.redis is not None
    assert len(report.embedding_chain) >= 1
    assert len(report.llm_chain) >= 1


def test_build_status_report_has_alias_fields():
    s = make_settings()
    report = build_status_report(s)
    assert hasattr(report, "qdrant_alias_target")
    assert hasattr(report, "qdrant_points")
