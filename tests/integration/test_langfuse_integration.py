"""Integration tests for Langfuse observability.

Tests health check, client initialization, and trace/span/generation lifecycle
against a real Langfuse instance.

Run with: pytest tests/test_langfuse_integration.py -v -m integration
"""

import pytest

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.observability.langfuse_client import (
    _check_langfuse_health,
    get_langfuse_instance,
)
from tests.conftest import require_langfuse

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.langfuse
class TestLangfuseHealthCheck:
    def test_health_endpoint_returns_ok(self):
        require_langfuse()
        # settings.langfuse_host may point to Docker hostname (langfuse:3000)
        # which is unresolvable from the host. Use localhost:3000 instead.
        result = _check_langfuse_health("http://localhost:3000", timeout=5)
        assert result is True

    def test_unreachable_host_returns_false(self):
        result = _check_langfuse_health("http://127.0.0.1:19999", timeout=1)
        assert result is False


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.langfuse
class TestLangfuseClientInit:
    @pytest.fixture(scope="class")
    def langfuse_client(self):
        require_langfuse()
        client = get_langfuse_instance()
        if client is None:
            pytest.skip("Langfuse client could not be initialized")
        return client

    def test_client_not_none(self, langfuse_client):
        assert langfuse_client is not None

    def test_client_has_start_observation(self, langfuse_client):
        assert hasattr(langfuse_client, "start_observation")
        assert callable(langfuse_client.start_observation)

    def test_client_has_flush(self, langfuse_client):
        assert hasattr(langfuse_client, "flush")
        assert callable(langfuse_client.flush)


# ---------------------------------------------------------------------------
# Trace lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.langfuse
class TestLangfuseTraceLifecycle:
    @pytest.fixture(scope="class")
    def langfuse_client(self):
        require_langfuse()
        client = get_langfuse_instance()
        if client is None:
            pytest.skip("Langfuse client could not be initialized")
        return client

    def test_create_trace_and_flush(self, langfuse_client):
        trace = langfuse_client.start_observation(name="itest-trace", input="test", as_type="trace")
        assert trace is not None
        trace.update(output="done")
        trace.end()
        langfuse_client.flush()

    def test_trace_with_child_span(self, langfuse_client):
        trace = langfuse_client.start_observation(name="itest-trace-span", as_type="trace")
        span = trace.start_observation(name="itest-span", input="span-in", as_type="span")
        span.update(output="span-out")
        span.end()
        trace.update(output="done")
        trace.end()
        langfuse_client.flush()

    def test_trace_with_generation(self, langfuse_client):
        trace = langfuse_client.start_observation(name="itest-trace-gen", as_type="trace")
        gen = trace.start_observation(name="itest-gen", model="test-model", input="Q?", as_type="generation")
        gen.update(output="A")
        gen.end()
        trace.update(output="A")
        trace.end()
        langfuse_client.flush()

    def test_multiple_spans(self, langfuse_client):
        trace = langfuse_client.start_observation(name="itest-multi-span", input="multi", as_type="trace")
        for i in range(3):
            span = trace.start_observation(name=f"itest-span-{i}", input=f"in-{i}", as_type="span")
            span.update(output=f"out-{i}")
            span.end()
        trace.update(output="all done")
        trace.end()
        langfuse_client.flush()

    def test_trace_with_metadata_and_tags(self, langfuse_client):
        trace = langfuse_client.start_observation(
            name="itest-metadata",
            input="meta-test",
            metadata={"env": "test", "version": "1.0"},
            as_type="trace",
        )
        trace.update(output="done", metadata={"completed": True})
        trace.end()
        langfuse_client.flush()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.langfuse
class TestLangfuseConfiguration:
    def test_settings_have_langfuse_keys(self):
        assert settings.langfuse_public_key.get_secret_value().startswith("pk-lf-")
        assert settings.langfuse_secret_key.get_secret_value().startswith("sk-lf-")
        assert settings.langfuse_host.startswith("http")

    def test_langfuse_host_env_override(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "http://custom:9999")
        from data_engineering_copilot.config.settings import AppSettings

        custom = AppSettings()
        assert custom.langfuse_host == "http://custom:9999"

    def test_langfuse_defaults_when_no_env(self, monkeypatch):
        for var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        from data_engineering_copilot.config.settings import AppSettings

        custom = AppSettings(_env_file=None)
        assert custom.langfuse_host == "http://langfuse:3000"
