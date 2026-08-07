"""Tests for API user/session extraction used for Langfuse session tracking."""

from __future__ import annotations

from types import SimpleNamespace

from data_engineering_copilot.api.routes import _extract_user_session


def _request(headers=None, query_params=None) -> SimpleNamespace:
    return SimpleNamespace(headers=headers or {}, query_params=query_params or {})


def test_headers_take_precedence_over_query_params() -> None:
    req = _request(
        {"X-User-ID": "hdr-user", "X-Session-ID": "hdr-session"},
        {"user_id": "q-user", "session_id": "q-session"},
    )
    assert _extract_user_session(req) == ("hdr-user", "hdr-session")


def test_query_params_used_when_headers_absent() -> None:
    req = _request({}, {"user_id": "q-user", "session_id": "q-session"})
    assert _extract_user_session(req) == ("q-user", "q-session")


def test_none_when_neither_present() -> None:
    assert _extract_user_session(_request()) == (None, None)


def test_mixed_headers_and_query_params() -> None:
    req = _request({"X-Session-ID": "hdr-session"}, {"user_id": "q-user"})
    assert _extract_user_session(req) == ("q-user", "hdr-session")
