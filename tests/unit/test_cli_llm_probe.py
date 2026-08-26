"""Tests for cli_llm_probe.py."""

from __future__ import annotations

import httpx

from data_engineering_copilot.cli_llm_probe import (
    ProbeResult,
    ProbeTarget,
    _format_table,
    _redact_headers,
    _serialize,
    _summarize_error,
    _verdict,
)


def _make_target(kind: str = "llm", provider: str = "groq", model: str = "test") -> ProbeTarget:
    return ProbeTarget(kind=kind, provider=provider, model=model, roles=["global"])


def _make_result(status: str = "OK", **kwargs) -> ProbeResult:
    return ProbeResult(target=_make_target(), status=status, **kwargs)


class TestRedactHeaders:
    def test_redacts_authorization(self) -> None:
        headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
        result = _redact_headers(headers)
        assert result["Authorization"] == "[REDACTED]"
        assert result["Content-Type"] == "application/json"

    def test_handles_dict_input(self) -> None:
        headers = {"authorization": "Bearer token", "Accept": "*/*"}
        result = _redact_headers(headers)
        assert result["authorization"] == "[REDACTED]"
        assert result["Accept"] == "*/*"

    def test_case_insensitive_redaction(self) -> None:
        headers = {"AUTHORIZATION": "secret"}
        result = _redact_headers(headers)
        assert result["AUTHORIZATION"] == "[REDACTED]"


class TestVerdict:
    def test_ok_status(self) -> None:
        result = _make_result("OK")
        assert _verdict(result) == "OK"

    def test_skip_status(self) -> None:
        result = _make_result("SKIP")
        assert _verdict(result) == "SKIP"

    def test_fail_status(self) -> None:
        result = _make_result("FAIL")
        assert _verdict(result) == "FAIL"


class TestSummarizeError:
    def test_rate_limited(self) -> None:
        response = httpx.Response(429, text="rate limited", headers={"Retry-After": "30"})
        request = httpx.Request("GET", "http://example.com")
        exc = httpx.HTTPStatusError("Rate limited", request=request, response=response)
        result = _summarize_error(exc)
        assert "RATE_LIMITED" in result
        assert "Retry-After: 30s" in result

    def test_auth_error(self) -> None:
        response = httpx.Response(401, text="unauthorized")
        request = httpx.Request("GET", "http://example.com")
        exc = httpx.HTTPStatusError("Auth error", request=request, response=response)
        result = _summarize_error(exc)
        assert "AUTH" in result
        assert "401" in result

    def test_generic_http_error(self) -> None:
        response = httpx.Response(500, text="server error")
        request = httpx.Request("GET", "http://example.com")
        exc = httpx.HTTPStatusError("Server error", request=request, response=response)
        result = _summarize_error(exc)
        assert "HTTP 500" in result


class TestFormatTable:
    def test_formats_results(self) -> None:
        results = [
            _make_result("OK", latency_ms=100.0, http_status=200),
            _make_result("FAIL", http_status=500),
        ]
        table = _format_table(results)
        assert "Kind" in table
        assert "Provider" in table
        assert "groq" in table
        assert "OK" in table
        assert "FAIL" in table

    def test_empty_results(self) -> None:
        table = _format_table([])
        assert "Kind" in table


class TestSerialize:
    def test_serializes_results(self) -> None:
        results = [
            _make_result("OK", latency_ms=100.0, http_status=200, prompt_tokens=10),
        ]
        serialized = _serialize(results)
        assert len(serialized) == 1
        assert serialized[0]["status"] == "OK"
        assert serialized[0]["latency_ms"] == 100.0
        assert serialized[0]["http_status"] == 200

    def test_empty_results(self) -> None:
        assert _serialize([]) == []
