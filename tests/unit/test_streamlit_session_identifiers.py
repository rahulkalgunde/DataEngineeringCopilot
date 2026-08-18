"""Tests for Streamlit session/user identifier generation."""

from __future__ import annotations

import uuid

import respx

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


def test_get_chat_user_id_is_stable() -> None:
    import streamlit as st

    from data_engineering_copilot.ui.streamlit_app import _get_chat_user_id

    st.session_state.pop("chat_user_id", None)
    first = _get_chat_user_id()
    second = _get_chat_user_id()
    assert first == second
    assert first.startswith("anon-")


@respx.mock
def test_stream_chat_once_parses_sse_events() -> None:
    import httpx

    from data_engineering_copilot.ui import streamlit_app

    body = "\n".join(
        [
            'data: {"type": "session_created", "session_id": "s1", "title": "hi"}\n\n',
            'data: {"type": "sources", "sources": [{"source_name": "x", "url": "https://x"}]}\n\n',
            'data: {"type": "token", "content": "Hello "}\n\n',
            'data: {"type": "token", "content": "world"}\n\n',
            'data: {"type": "done", "text": "Hello world", "confidence": 0.9}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    respx.post(f"{streamlit_app.API_BASE_URL}/api/v1/chat").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    events, full_text, session_id = streamlit_app._stream_chat_once("hello", None)

    assert full_text == "Hello world"
    assert session_id == "s1"
    types = [e["type"] for e in events]
    assert types[0] == "session_created"
    assert "token" in types
    assert types[-1] == "done"


def test_extract_streaming_answer_ignores_incomplete_json() -> None:
    """Partial JSON fragments must render nothing, not raw JSON, mid-stream."""
    from data_engineering_copilot.ui.streamlit_app import _extract_streaming_answer

    assert _extract_streaming_answer("") is None
    assert _extract_streaming_answer('{"status": ') is None
    assert _extract_streaming_answer('{"status": "SUCCESS", "answer": "Spark') is None
    assert _extract_streaming_answer('{"status": "SUCCESS", "answer": "Spark", "missing_info": null}') == "Spark"
    assert _extract_streaming_answer("plain text") is None


@respx.mock
def test_chat_turn_renders_sources_and_persists_groundedness(monkeypatch) -> None:
    """Sources SSE event must surface as a Sources expander + persist on the message."""
    import httpx
    from streamlit.testing.v1 import AppTest

    from data_engineering_copilot.ui import streamlit_app

    # Hermetic: force the chat tab to render without an ambient Qdrant.
    monkeypatch.setenv("STREAMLIT_ASSUME_QDRANT_UP", "1")

    respx.post(f"{streamlit_app.API_BASE_URL}/api/v1/chat").mock(
        return_value=httpx.Response(
            200,
            text='data: {"type": "session_created", "session_id": "s1", "title": "t"}\n\n'
            'data: {"type": "sources", "sources": [{"source_name": "Delta Docs", "title": "Delta Lake Guide", "url": "https://docs.delta.io/", "snippet": "ACID.", "chunk_id": "c42"}]}\n\n'
            'data: {"type": "token", "content": "ok"}\n\n'
            'data: {"type": "done", "text": "ok", "confidence": 0.9, "groundedness_score": 0.6, "groundedness_claims": ["Delta is YARN-only"]}\n\n'
            "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )

    at = AppTest.from_file("data_engineering_copilot/ui/streamlit_app.py", default_timeout=30)
    at.session_state["chat_session_id"] = "s1"
    at.run()
    at.chat_input[0].set_value("What is Delta?")
    at.run()

    assistant = [m for m in at.session_state["chat_messages"] if m["role"] == "assistant"][-1]
    assert assistant["sources"] == [
        {
            "source_name": "Delta Docs",
            "title": "Delta Lake Guide",
            "url": "https://docs.delta.io/",
            "snippet": "ACID.",
            "chunk_id": "c42",
        }
    ]
    assert assistant["groundedness_score"] == 0.6
    assert assistant["groundedness_claims"] == ["Delta is YARN-only"]

    # The expander label must be rendered in the DOM.
    labels = [b.label for b in at.expander]
    assert any("Sources (1)" in label for label in labels), f"no Sources expander, got {labels}"


@respx.mock
def test_suggestion_chip_click_submits_as_user_message(monkeypatch) -> None:
    """Clicking a follow-up chip must submit it as the next user turn."""
    import httpx
    from streamlit.testing.v1 import AppTest

    from data_engineering_copilot.ui import streamlit_app

    # Hermetic: force the chat tab to render without an ambient Qdrant.
    monkeypatch.setenv("STREAMLIT_ASSUME_QDRANT_UP", "1")

    respx.post(f"{streamlit_app.API_BASE_URL}/api/v1/chat").mock(
        return_value=httpx.Response(
            200,
            text='data: {"type": "session_created", "session_id": "s1", "title": "t"}\n\n'
            'data: {"type": "token", "content": "ok"}\n\n'
            'data: {"type": "done", "text": "ok", "confidence": 0.9}\n\n'
            'data: {"type": "suggestions", "suggestions": ["Follow-up A?", "Follow-up B?"]}\n\n'
            "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )

    at = AppTest.from_file("data_engineering_copilot/ui/streamlit_app.py", default_timeout=30)
    at.session_state["chat_session_id"] = "s1"
    at.run()

    # Ask a question so an answer is generated; its suggestions must render.
    at.chat_input[0].set_value("What is Spark?")
    at.run()
    chips = [b for b in at.button if b.label in ("Follow-up A?", "Follow-up B?")]
    assert len(chips) == 2, "suggestions must render right after the answer is generated"

    # Clicking a chip submits it as the next user turn.
    clicked = next(b for b in at.button if b.label == "Follow-up A?")
    clicked.click()
    at.run()

    roles = [m["role"] for m in at.session_state["chat_messages"]]
    contents = [m["content"] for m in at.session_state["chat_messages"]]
    assert "Follow-up A?" in contents, f"chip text must be submitted as a user message, got {contents}"
    assert roles == ["user", "assistant", "user", "assistant"], f"unexpected roles: {roles}"
