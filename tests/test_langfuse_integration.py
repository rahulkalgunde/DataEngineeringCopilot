"""
Integration tests for Langfuse observability client.

These tests require a running Langfuse instance (docker compose up langfuse langfuse-postgres clickhouse minio).
They use the real Langfuse v3 API — no mocks.

Mark: pytest -m integration
Skip in CI: set LANGFUSE_INTEGRATION_TESTS=0 or when Langfuse is unreachable.
"""

import os
import json
import urllib.request
import urllib.error
import pytest

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.observability.langfuse_client import (
    get_langfuse_instance,
    _check_langfuse_health,
    start_trace,
    end_trace,
    log_event,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _langfuse_is_reachable(host: str = None, timeout: int = 3) -> bool:
    """Return True if the Langfuse health endpoint responds with OK."""
    if host is None:
        host = settings.langfuse_host
    health_url = f"{host.rstrip('/')}/api/public/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                return data.get("status") == "OK"
    except Exception:
        pass
    return False


def _skip_if_langfuse_unreachable():
    """Pytest helper: skip the current test if Langfuse is not running."""
    if not _langfuse_is_reachable():
        pytest.skip("Langfuse is not reachable — skipping integration test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def langfuse_client():
    """Return a real Langfuse v3 client, or skip the module if unreachable."""
    _skip_if_langfuse_unreachable()
    client = get_langfuse_instance()
    if client is None:
        pytest.skip("Langfuse client could not be initialized")
    return client


# ---------------------------------------------------------------------------
# Health-check tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLangfuseHealthCheck:
    """Verify the Langfuse server is reachable and healthy."""

    def test_health_endpoint_returns_ok(self):
        """GET /api/public/health should return {"status":"OK"}."""
        _skip_if_langfuse_unreachable()
        result = _check_langfuse_health(settings.langfuse_host, timeout=5)
        assert result is True, (
            f"Langfuse health check failed for host={settings.langfuse_host}"
        )

    def test_health_endpoint_unreachable_host(self):
        """A non-existent host should return False (not raise)."""
        result = _check_langfuse_health("http://127.0.0.1:19999", timeout=1)
        assert result is False


# ---------------------------------------------------------------------------
# Client initialization tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLangfuseClientInitialization:
    """Verify the Langfuse client can be created and is functional."""

    def test_get_langfuse_instance_returns_client(self, langfuse_client):
        """Client should not be None when Langfuse is running."""
        assert langfuse_client is not None

    def test_client_has_expected_api(self, langfuse_client):
        """The v3 client should expose start_observation and flush."""
        assert hasattr(langfuse_client, "start_observation")
        assert hasattr(langfuse_client, "flush")
        assert callable(langfuse_client.start_observation)
        assert callable(langfuse_client.flush)


# ---------------------------------------------------------------------------
# Trace / span / generation lifecycle tests (real API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLangfuseTraceLifecycle:
    """End-to-end trace creation using the real Langfuse v3 API."""

    def test_create_trace_and_flush(self, langfuse_client):
        """Create a minimal trace, end it, flush, and verify no exception."""
        trace = langfuse_client.start_observation(
            name="integration-test-trace",
            input="test input",
            as_type="trace",
        )
        assert trace is not None

        trace.update(output="test output")
        trace.end()

        # Flush should not raise
        langfuse_client.flush()

    def test_create_trace_with_span(self, langfuse_client):
        """Create a trace with a child span and flush."""
        trace = langfuse_client.start_observation(
            name="integration-test-trace-span",
            as_type="trace",
        )

        span = trace.start_observation(
            name="integration-test-span",
            input="span input",
            as_type="span",
        )
        span.update(output="span output")
        span.end()

        trace.update(output="trace output")
        trace.end()
        langfuse_client.flush()

    def test_create_trace_with_generation(self, langfuse_client):
        """Create a trace with a generation-type observation (LLM call)."""
        trace = langfuse_client.start_observation(
            name="integration-test-trace-gen",
            as_type="trace",
        )

        gen = trace.start_observation(
            name="integration-test-generation",
            model="test-model",
            input="What is 2+2?",
            as_type="generation",
        )
        gen.update(output="4")
        gen.end()

        trace.update(output="4")
        trace.end()
        langfuse_client.flush()

    def test_multiple_spans_in_trace(self, langfuse_client):
        """Create a trace with multiple sibling spans."""
        trace = langfuse_client.start_observation(
            name="integration-test-multi-span",
            input="multi-span test",
            as_type="trace",
        )

        for i in range(3):
            span = trace.start_observation(
                name=f"integration-test-span-{i}",
                input=f"input-{i}",
                as_type="span",
            )
            span.update(output=f"output-{i}")
            span.end()

        trace.update(output="all spans done")
        trace.end()
        langfuse_client.flush()

    def test_trace_with_metadata_and_tags(self, langfuse_client):
        """Create a trace with metadata and tags."""
        trace = langfuse_client.start_observation(
            name="integration-test-metadata",
            input="test with metadata",
            metadata={"env": "test", "version": "1.0.0"},
            as_type="trace",
        )
        trace.update(
            output="done",
            metadata={"completed": True},
        )
        trace.end()
        langfuse_client.flush()


# ---------------------------------------------------------------------------
# Helper-function tests (real Langfuse)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLangfuseHelperFunctions:
    """Verify the helper functions in langfuse_client.py work with real Langfuse."""

    def test_start_and_end_trace(self):
        """start_trace / end_trace should not raise."""
        _skip_if_langfuse_unreachable()
        trace = start_trace(name="helper-trace-test", input="hello")
        assert trace is not None
        end_trace(trace, output="world")

        # Flush via the module-level instance
        lf = get_langfuse_instance()
        if lf:
            lf.flush()

    def test_log_event(self):
        """log_event should create a span inside a trace."""
        _skip_if_langfuse_unreachable()
        trace = start_trace(name="helper-event-trace")
        assert trace is not None

        log_event(trace, name="helper-event-span", input="event input")
        end_trace(trace, output="done")

        lf = get_langfuse_instance()
        if lf:
            lf.flush()


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLangfuseConfiguration:
    """Verify Langfuse settings are correctly resolved."""

    def test_settings_have_langfuse_keys(self):
        """AppSettings should have non-empty Langfuse keys."""
        assert settings.langfuse_public_key.startswith("pk-lf-")
        assert settings.langfuse_secret_key.startswith("sk-lf-")
        assert settings.langfuse_host.startswith("http")

    def test_langfuse_host_from_env_var(self, monkeypatch):
        """LANGFUSE_HOST env var should override the default."""
        monkeypatch.setenv("LANGFUSE_HOST", "http://custom-langfuse:9999")
        # Re-create settings to pick up the env var
        custom_settings = AppSettings()
        assert custom_settings.langfuse_host == "http://custom-langfuse:9999"

    def test_langfuse_public_key_from_env_var(self, monkeypatch):
        """LANGFUSE_PUBLIC_KEY env var should override the default."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-custom-key")
        custom_settings = AppSettings()
        assert custom_settings.langfuse_public_key == "pk-lf-custom-key"

    def test_langfuse_secret_key_from_env_var(self, monkeypatch):
        """LANGFUSE_SECRET_KEY env var should override the default."""
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-custom-secret")
        custom_settings = AppSettings()
        assert custom_settings.langfuse_secret_key == "sk-lf-custom-secret"

    def test_langfuse_defaults_when_no_env_vars(self, monkeypatch):
        """When no env vars are set, defaults should be used."""
        for var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        custom_settings = AppSettings()
        assert custom_settings.langfuse_host == "http://localhost:3000"
        assert custom_settings.langfuse_public_key.startswith("pk-lf-")
        assert custom_settings.langfuse_secret_key.startswith("sk-lf-")