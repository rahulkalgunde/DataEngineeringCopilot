"""Tests for Streamlit session/user identifier generation."""

from __future__ import annotations

import uuid

from data_engineering_copilot.ui.streamlit_app import _new_session_identifiers


def test_new_session_identifiers_are_unique_and_derived() -> None:
    session_id, user_id = _new_session_identifiers()
    # session_id is a valid UUID v4
    uuid.UUID(session_id)
    # user_id is the stable anon-<session prefix> form
    assert user_id == f"anon-{session_id[:8]}"
    assert user_id.startswith("anon-")


def test_new_session_identifiers_differ_between_calls() -> None:
    first = _new_session_identifiers()
    second = _new_session_identifiers()
    assert first != second
    assert first[1] != second[1]
