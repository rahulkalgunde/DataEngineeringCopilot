"""Langfuse contract verification tests.

Validates the Langfuse client adapter surface, host fallback logic,
and configuration contract without requiring a running Langfuse instance.

Run with: pytest tests/integration/test_langfuse_contract.py -v -m integration
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_engineering_copilot.observability.langfuse_client import (
    LangfuseCompat,
    _candidate_langfuse_hosts,
)

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Host candidate fallback logic
# ---------------------------------------------------------------------------


class TestLangfuseHostCandidates:
    def test_docker_hostname_generates_localhost_fallbacks(self):
        candidates = _candidate_langfuse_hosts("http://langfuse:3000")
        assert "http://langfuse:3000" in candidates
        assert "http://localhost:3000" in candidates
        assert "http://127.0.0.1:3000" in candidates

    def test_localhost_generates_127_fallback(self):
        candidates = _candidate_langfuse_hosts("http://localhost:3000")
        assert "http://localhost:3000" in candidates
        assert "http://127.0.0.1:3000" in candidates

    def test_127_localhost_generates_localhost_fallback(self):
        candidates = _candidate_langfuse_hosts("http://127.0.0.1:3000")
        assert "http://127.0.0.1:3000" in candidates
        assert "http://localhost:3000" in candidates

    def test_empty_host_returns_empty(self):
        assert _candidate_langfuse_hosts("") == []
        assert _candidate_langfuse_hosts("  ") == []

    def test_no_duplicates(self):
        candidates = _candidate_langfuse_hosts("http://localhost:3000")
        assert len(candidates) == len(set(candidates))


# ---------------------------------------------------------------------------
# LangfuseCompat adapter contract
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLangfuseCompatContract:
    def test_compat_exposes_required_api(self):
        mock_client = MagicMock()
        compat = LangfuseCompat(mock_client)
        assert callable(getattr(compat, "start_observation", None))
        assert callable(getattr(compat, "flush", None))
        assert callable(getattr(compat, "trace", None))
        assert callable(getattr(compat, "span", None))
        assert callable(getattr(compat, "generation", None))

    def test_trace_delegates_to_start_observation(self):
        mock_client = MagicMock()
        compat = LangfuseCompat(mock_client)
        compat.trace(name="test-trace")
        mock_client.start_observation.assert_called_once()

    def test_flush_delegates_to_inner_client(self):
        mock_client = MagicMock()
        compat = LangfuseCompat(mock_client)
        compat.flush()
        mock_client.flush.assert_called_once()

    def test_auth_check_delegates_to_inner_client(self):
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        compat = LangfuseCompat(mock_client)
        assert compat.auth_check() is True
