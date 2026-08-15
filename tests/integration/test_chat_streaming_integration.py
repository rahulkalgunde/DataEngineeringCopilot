"""Integration tests for the POST /api/v1/chat SSE endpoint.

Exercises the real HTTP/SSE contract: ``session_created`` → ``status``/``sources``
→ ``token`` → ``done``, terminal ``[DONE]`` marker, and error events. The
conversation service is swapped for a deterministic double; the HTTP layer
(RBAC resolution, SSE framing, disconnect guard) is real.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from data_engineering_copilot.domain.models import ChatMessage, ChatSession


class _FakeConversation:
    """Deterministic double of ConversationService driving the SSE contract."""

    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.messages: dict[str, list[ChatMessage]] = {}

    def seed_session(self, session_id: str, title: str = "T") -> None:
        from data_engineering_copilot.domain.models import ChatMessage, ChatSession

        self.sessions[session_id] = ChatSession(
            session_id=session_id,
            user_id="anonymous",
            title=title,
            created_at=1.0,
            updated_at=2.0,
        )
        self.messages[session_id] = [
            ChatMessage(message_id="m1", session_id=session_id, role="user", content="hi", timestamp=1.0),
            ChatMessage(message_id="m2", session_id=session_id, role="assistant", content="hello", timestamp=2.0),
        ]

    async def start_or_resume(self, session_id, user_id, *, first_message=None):
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        from data_engineering_copilot.domain.models import ChatSession

        return ChatSession(
            session_id=session_id or "fake-session",
            user_id=user_id,
            title=first_message or "T",
        )

    async def get_history(self, session_id, max_turns):
        return self.messages.get(session_id, [])

    async def list_sessions(self, user_id, limit=50):
        return list(self.sessions.values())

    async def get_session(self, session_id):
        return self.sessions.get(session_id)

    async def delete_session(self, session_id):
        self.sessions.pop(session_id, None)
        self.messages.pop(session_id, None)

    async def chat_stream(
        self, session_id, user_id, message, *, source_filter=None, max_history_turns=10, cache_scope=None
    ):
        yield {"type": "session_created", "session_id": "fake-session", "title": message[:40]}
        yield {"type": "status", "message": "Retrieving documents"}
        yield {
            "type": "sources",
            "sources": [{"source_name": "test", "url": "https://x", "title": "T", "snippet": "s"}],
        }
        yield {"type": "token", "content": "The "}
        yield {"type": "token", "content": "answer."}
        yield {"type": "done", "text": "The answer.", "confidence": 0.85}
        yield {"type": "suggestions", "suggestions": ["Follow-up one?", "Follow-up two?"]}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from data_engineering_copilot.api.app import app

    return TestClient(app)


@pytest.fixture
def fake_conversation():
    return _FakeConversation()


def _parse_sse_events(raw_text: str) -> list[dict]:
    events = []
    for line in raw_text.strip().splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: ") :]))
    return events


def _patch_conversation(monkeypatch, fake_conversation):
    import data_engineering_copilot.services.conversation_service_singleton as css

    async def _get():
        return fake_conversation

    monkeypatch.setattr(css, "get_conversation_service", _get)


@pytest.mark.integration
@pytest.mark.api
def test_chat_sse_contract(monkeypatch, fake_conversation, client):
    _patch_conversation(monkeypatch, fake_conversation)
    resp = client.post("/api/v1/chat", json={"message": "What is Delta Lake?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert resp.headers["cache-control"] == "no-cache"
    events = _parse_sse_events(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "session_created"
    assert "sources" in types
    assert "token" in types
    assert "done" in types
    assert resp.text.rstrip().endswith("data: [DONE]")


@pytest.mark.integration
@pytest.mark.api
def test_chat_sse_emits_suggestions_after_done(monkeypatch, fake_conversation, client):
    _patch_conversation(monkeypatch, fake_conversation)
    resp = client.post("/api/v1/chat", json={"message": "q"})
    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    types = [e["type"] for e in events]
    assert types[-1] == "suggestions"
    assert types.index("suggestions") > types.index("done")
    suggestions = events[-1]["suggestions"]
    assert suggestions == ["Follow-up one?", "Follow-up two?"]


@pytest.mark.integration
@pytest.mark.api
def test_chat_sse_sources_before_first_token(monkeypatch, fake_conversation, client):
    _patch_conversation(monkeypatch, fake_conversation)
    resp = client.post("/api/v1/chat", json={"message": "q"})
    events = _parse_sse_events(resp.text)
    types = [e["type"] for e in events]
    assert types.index("sources") < types.index("token")


@pytest.mark.integration
@pytest.mark.api
def test_chat_emits_error_event_on_failure(monkeypatch, fake_conversation, client):
    async def _boom_chat(session_id, user_id, message, *, source_filter=None, max_history_turns=10, cache_scope=None):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    fake_conversation.chat_stream = _boom_chat  # type: ignore[method-assign]
    _patch_conversation(monkeypatch, fake_conversation)
    resp = client.post("/api/v1/chat", json={"message": "q"})
    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert any(e["type"] == "error" for e in events)
    assert "data: [DONE]" in resp.text


@pytest.mark.integration
@pytest.mark.api
def test_chat_invalid_request_returns_422(client):
    resp = client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.api
def test_list_sessions(monkeypatch, fake_conversation, client):
    fake_conversation.seed_session("s1", "First Chat")
    fake_conversation.seed_session("s2", "Second Chat")
    _patch_conversation(monkeypatch, fake_conversation)
    resp = client.get("/api/v1/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sessions"]) == 2
    titles = {s["title"] for s in body["sessions"]}
    assert titles == {"First Chat", "Second Chat"}


@pytest.mark.integration
@pytest.mark.api
def test_get_session_thread(monkeypatch, fake_conversation, client):
    fake_conversation.seed_session("s1", "First Chat")
    _patch_conversation(monkeypatch, fake_conversation)
    resp = client.get("/api/v1/sessions/s1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s1"
    assert body["title"] == "First Chat"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]


@pytest.mark.integration
@pytest.mark.api
def test_get_session_not_found(monkeypatch, fake_conversation, client):
    _patch_conversation(monkeypatch, fake_conversation)
    resp = client.get("/api/v1/sessions/nope")
    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.api
def test_delete_session(monkeypatch, fake_conversation, client):
    fake_conversation.seed_session("s1")
    _patch_conversation(monkeypatch, fake_conversation)
    resp = client.delete("/api/v1/sessions/s1")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert "s1" not in fake_conversation.sessions
