"""Tests for cli_llm_probe pure helpers.

No network anywhere: payload building uses attribute doubles, error
summarization wraps synthetic httpx responses. `dec probe-llm` makes paid
calls in production — these tests pin the offline logic that report formats
and payloads depend on.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from data_engineering_copilot.cli_llm_probe import (
    ProbeResult,
    ProbeTarget,
    _build_llm_request_payload,
    _redact_headers,
    _summarize_error,
    _verdict,
)

pytestmark = pytest.mark.unit


class TestRedactHeaders:
    def test_authorization_redacted_case_insensitive(self):
        out = _redact_headers({"Authorization": "Bearer sk-secret", "x-custom": "v"})
        assert out["Authorization"] == "[REDACTED]"
        assert out["x-custom"] == "v"

    def test_no_secrets_in_any_value(self):
        headers = {"authorization": "Bearer sk-secret", "AUTHORIZATION": "Bearer sk2", "api-key": "keep-visible"}
        out = _redact_headers(headers)
        joined = str(out)
        assert "sk-secret" not in joined and "sk2" not in joined
        assert out["api-key"] == "keep-visible"


class TestBuildLlmRequestPayload:
    def _client(self, **overrides):
        base = dict(
            model="llama3",
            _temperature=0.15,
            _max_tokens=512,
            _max_tokens_field="max_tokens",
            _extra_body={},
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_minimal_payload_shape(self):
        payload = _build_llm_request_payload(self._client(), "What is Spark?")  # type: ignore[arg-type]
        assert payload["model"] == "llama3"
        assert payload["temperature"] == 0.15
        assert payload["max_tokens"] == 512
        assert payload["messages"][-1]["role"] == "user"
        assert "What is Spark?" in json_dumps(payload)

    def test_max_tokens_omitted_when_none_or_nonpositive(self):
        for bad in (None, 0, -5):
            payload = _build_llm_request_payload(self._client(_max_tokens=bad), "p")  # type: ignore[arg-type]
            assert "max_tokens" not in payload

    def test_custom_max_tokens_field_name(self):
        payload = _build_llm_request_payload(
            self._client(_max_tokens_field="max_completion_tokens"),  # type: ignore[arg-type]
            "p",
        )
        assert "max_tokens" not in payload
        assert payload["max_completion_tokens"] == 512

    def test_extra_body_merged_last(self):
        payload = _build_llm_request_payload(
            self._client(_extra_body={"seed": 42, "temperature": 9.9}),  # type: ignore[arg-type]
            "p",
        )
        # extra_body wins on key collisions: it is applied last
        assert payload["temperature"] == 9.9
        assert payload["seed"] == 42


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload)


class TestSummarizeError:
    @staticmethod
    def _error(status: int, text: str, headers: dict | None = None) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "https://api.example.com/v1/chat")
        response = httpx.Response(status_code=status, text=text, headers=headers or {}, request=request)
        return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)

    def test_rate_limit_with_retry_after(self):
        msg = _summarize_error(self._error(429, "slow down", {"Retry-After": "7"}))
        assert msg.startswith("RATE_LIMITED (Retry-After: 7s)")
        assert "slow down" in msg

    def test_rate_limit_without_retry_after(self):
        msg = _summarize_error(self._error(429, "slow down"))
        assert "Retry-After" not in msg
        assert msg.startswith("RATE_LIMITED")

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_errors_labeled(self, status):
        msg = _summarize_error(self._error(status, "denied"))
        assert msg.startswith(f"AUTH: HTTP {status}")

    def test_generic_status_passthrough(self):
        msg = _summarize_error(self._error(502, "bad gateway"))
        assert msg.startswith("HTTP 502")

    def test_body_truncated_to_300_chars(self):
        msg = _summarize_error(self._error(500, "x" * 1000))
        assert len(msg) < 400


class TestVerdict:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [("OK", "OK"), ("SKIP", "SKIP"), ("FAIL", "FAIL")],
    )
    def test_mapping(self, status, expected):
        result = ProbeResult(target=ProbeTarget(kind="llm", provider="p", model="m"), status=status)
        assert _verdict(result) == expected

    def test_table_includes_all_results(self):
        rows = [
            ProbeResult(target=ProbeTarget(kind="llm", provider="a", model="m1"), status="OK"),
            ProbeResult(target=ProbeTarget(kind="embedding", provider="b", model="m2"), status="FAIL"),
        ]
        from data_engineering_copilot.cli_llm_probe import _format_table

        table = _format_table(rows)
        assert "a" in table and "FAIL" in table
