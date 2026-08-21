"""Unit tests for generation-layer tuning (P1/P2/P3): temperature/seed/penalties
injection via the factory, and capability-guarded payload construction in LLMClient.
"""

from __future__ import annotations

from data_engineering_copilot.factory import _build_purpose_llm_client
from data_engineering_copilot.infrastructure.llm_client import LLMClient
from tests.conftest import make_settings


def _answer_client(**settings_overrides):
    s = make_settings(llm_provider="ollama", llm_model="llama3.2:3b", **settings_overrides)
    client = _build_purpose_llm_client(provider="ollama", model="llama3.2:3b", app_settings=s, purpose="answer")
    assert client is not None
    return client


def test_answer_purpose_uses_generation_temperature():
    client = _answer_client(generation_temperature=0.12)
    assert client._temperature == 0.12


def test_code_purpose_uses_code_generation_temperature():
    s = make_settings(llm_provider="ollama", llm_model="llama3.2:3b", code_generation_temperature=0.21)
    client = _build_purpose_llm_client(provider="ollama", model="llama3.2:3b", app_settings=s, purpose="code")
    assert client is not None
    assert client._temperature == 0.21


def test_non_answer_purpose_keeps_default_temperature():
    s = make_settings(llm_provider="ollama", llm_model="llama3.2:3b", generation_temperature=0.12)
    client = _build_purpose_llm_client(provider="ollama", model="llama3.2:3b", app_settings=s, purpose="rewrite")
    assert client is not None
    assert client._temperature == 0.05


def test_evaluation_purpose_uses_evaluation_temperature():
    s = make_settings(llm_provider="ollama", llm_model="llama3.2:3b", evaluation_temperature=0.0)
    client = _build_purpose_llm_client(provider="ollama", model="llama3.2:3b", app_settings=s, purpose="evaluation")
    assert client is not None
    assert client._temperature == 0.0


def test_seed_reaches_client_when_set():
    client = _answer_client(generation_seed=7)
    assert client._seed == 7


def test_structured_schema_set_only_for_answer_purpose():
    answer_client = _answer_client()
    assert answer_client._structured_schema is not None
    rewrite = make_settings(llm_provider="ollama", llm_model="llama3.2:3b")
    rewrite_client = _build_purpose_llm_client(
        provider="ollama", model="llama3.2:3b", app_settings=rewrite, purpose="rewrite"
    )
    assert rewrite_client is not None
    assert rewrite_client._structured_schema is None


def test_penalties_in_payload_for_supporting_provider():
    c = LLMClient(
        base_url="http://x/v1",
        model="m",
        provider="openrouter",
        frequency_penalty=0.5,
        presence_penalty=0.3,
        top_p=0.9,
    )
    p: dict = {}
    c._apply_generation_params(p)
    assert p["frequency_penalty"] == 0.5
    assert p["presence_penalty"] == 0.3
    assert p["top_p"] == 0.9


def test_penalties_absent_for_unsupported_provider():
    c = LLMClient(
        base_url="http://x/v1",
        model="m",
        provider="anthropic",
        frequency_penalty=0.5,
        presence_penalty=0.3,
        top_p=0.9,
    )
    p: dict = {}
    c._apply_generation_params(p)
    assert "frequency_penalty" not in p
    assert "presence_penalty" not in p
    assert "top_p" not in p


def test_structured_schema_uses_response_format_for_openai_style():
    c = LLMClient(
        base_url="http://x/v1",
        model="m",
        provider="openrouter",
        structured_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    p: dict = {}
    c._apply_generation_params(p)
    assert p["response_format"]["type"] == "json_schema"
    assert p["response_format"]["json_schema"]["strict"] is True


def test_structured_schema_uses_format_for_ollama():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    c = LLMClient(base_url="http://x/v1", model="m", provider="ollama", structured_schema=schema)
    p: dict = {}
    c._apply_generation_params(p)
    assert p["format"] == schema


def test_structured_schema_absent_for_unsupported_provider():
    c = LLMClient(
        base_url="http://x/v1",
        model="m",
        provider="anthropic",
        structured_schema={"type": "object"},
    )
    p: dict = {}
    c._apply_generation_params(p)
    assert "response_format" not in p
    assert "format" not in p


def test_seed_in_payload_only_for_supporting_provider():
    c = LLMClient(base_url="http://x/v1", model="m", provider="openrouter", seed=7)
    p: dict = {}
    c._apply_generation_params(p)
    assert p["seed"] == 7

    c2 = LLMClient(base_url="http://x/v1", model="m", provider="anthropic", seed=7)
    p2: dict = {}
    c2._apply_generation_params(p2)
    assert "seed" not in p2
