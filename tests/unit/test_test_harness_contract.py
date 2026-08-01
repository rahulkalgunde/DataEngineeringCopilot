"""Contract tests pinning the test harness itself.

These lock in the behaviors the hermetic test framework depends on, so a
future edit to ``tests/conftest.py`` can never silently break:

1. ``infra_unavailable`` — the REQUIRE_INFRA gate (skip vs. hard failure).
2. ``_redis_is_reachable`` — the auth-aware probe against compose Redis.
3. ``_needed_infra`` — the marker → service mapping used at collection time.
4. ``_test_allow_non_ollama`` — the explicit opt-in for provider-routing tests.
5. ``make_settings`` — the hermetic Ollama-only defaults.
"""

from __future__ import annotations

import socket

import pytest

from tests import conftest as tc

# ---------------------------------------------------------------------------
# REQUIRE_INFRA gate
# ---------------------------------------------------------------------------


def test_infra_unavailable_skips_when_infra_not_required(monkeypatch):
    monkeypatch.setattr(tc, "_REQUIRE_INFRA", False)
    with pytest.raises(pytest.skip.Exception, match="no redis today"):
        tc.infra_unavailable("no redis today")


def test_infra_unavailable_fails_when_infra_required(monkeypatch):
    monkeypatch.setattr(tc, "_REQUIRE_INFRA", True)
    with pytest.raises(RuntimeError, match="REQUIRE_INFRA=1: required infra unavailable: qdrant down"):
        tc.infra_unavailable("qdrant down")


def test_require_helpers_route_through_infra_unavailable(monkeypatch):
    monkeypatch.setattr(tc, "_REQUIRE_INFRA", False)
    with pytest.raises(pytest.skip.Exception, match="Qdrant not reachable"):
        tc.require_qdrant(url="http://127.0.0.1:1")
    with pytest.raises(pytest.skip.Exception, match="Redis not reachable"):
        tc.require_redis(url="redis://127.0.0.1:1/0")


# ---------------------------------------------------------------------------
# Redis probe (auth-aware)
# ---------------------------------------------------------------------------


class _FakeSocket:
    """Minimal socket stand-in feeding canned responses in order."""

    def __init__(self, responses: list[bytes]):
        self._responses = list(responses)
        self.sent: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, _nbytes: int) -> bytes:
        return self._responses.pop(0) if self._responses else b""

    def close(self) -> None:
        pass


def _patch_connection(monkeypatch, responses: list[bytes]):
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_a, **_k: _FakeSocket(responses),
    )


def test_redis_probe_authenticates_against_requirepass_server(monkeypatch):
    _patch_connection(monkeypatch, [b"-NOAUTH Authentication required.\r\n", b"+OK\r\n", b"+PONG\r\n"])
    assert tc._redis_is_reachable() is True


def test_redis_probe_accepts_passwordless_server(monkeypatch):
    _patch_connection(monkeypatch, [b"+PONG\r\n"])
    assert tc._redis_is_reachable("redis://localhost:6379/0") is True


def test_redis_probe_fails_when_password_missing(monkeypatch):
    _patch_connection(monkeypatch, [b"-NOAUTH Authentication required.\r\n"])
    assert tc._redis_is_reachable("redis://localhost:6379/0") is False


def test_redis_probe_fails_on_connection_error(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: (_ for _ in ()).throw(OSError("refused")))
    assert tc._redis_is_reachable() is False


# ---------------------------------------------------------------------------
# Marker → infra mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("markers", "expected"),
    [
        ({"unit"}, set()),
        ({"qdrant"}, {"Qdrant"}),
        ({"ollama"}, {"Ollama"}),
        ({"langfuse"}, {"Langfuse"}),
        ({"redis"}, {"Redis"}),
        ({"rag"}, {"Qdrant", "Ollama"}),
        ({"ingestion"}, {"Qdrant", "Ollama"}),
        ({"qdrant", "ollama", "langfuse", "redis"}, {"Qdrant", "Ollama", "Langfuse", "Redis"}),
        ({"e2e", "ingestion"}, {"Qdrant", "Ollama"}),
    ],
)
def test_needed_infra_marker_mapping(markers, expected):
    assert tc._needed_infra(markers) == expected


# ---------------------------------------------------------------------------
# Non-Ollama provider opt-in
# ---------------------------------------------------------------------------


def test_non_ollama_provider_raises_without_opt_in():
    with pytest.raises(RuntimeError, match="non-Ollama LLM provider"):
        tc.make_settings(llm_provider="openrouter", openrouter_api_key="sk-test")


def test_non_ollama_provider_allowed_with_opt_in():
    settings = tc.make_settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        _test_allow_non_ollama=True,
    )
    assert settings.llm_provider == "openrouter"
    assert "_test_allow_non_ollama" not in settings.model_fields_set


# ---------------------------------------------------------------------------
# Hermetic make_settings defaults
# ---------------------------------------------------------------------------


def test_make_settings_defaults_to_ollama_and_empty_overrides():
    settings = tc.make_settings()
    assert settings.llm_provider == "ollama"
    assert settings.embedding_provider == "ollama"
    assert settings.llm_model == "llama3.2:3b"
    assert settings.embedding_model_name == "nomic-embed-text"
    for field in ("answer", "rewrite", "groundedness", "intent", "enrichment", "evaluation", "code"):
        assert getattr(settings, f"{field}_llm_provider") == ""
    assert settings.code_llm_model == ""
    assert settings.openrouter_api_key.get_secret_value() == ""
    assert settings.groq_api_key.get_secret_value() == ""


def test_make_settings_env_file_never_leaks_into_settings():
    settings = tc.make_settings()
    assert "_env_file" not in settings.model_fields_set
    assert "_test_allow_non_ollama" not in settings.model_fields_set
