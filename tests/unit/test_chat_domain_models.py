"""Tests for conversational RAG domain models."""

from __future__ import annotations

from data_engineering_copilot.domain.models import ChatMessage, ChatSession
from data_engineering_copilot.domain.protocols import ChatSessionStoreProtocol


def test_chat_message_defaults():
    msg = ChatMessage(message_id="m1", session_id="s1", role="user", content="hello")
    assert msg.timestamp == 0.0
    assert msg.sources == ()
    assert msg.token_count == 0


def test_chat_message_sources_are_json_safe_dicts():
    msg = ChatMessage(
        message_id="m1",
        session_id="s1",
        role="assistant",
        content="answer",
        sources=({"source_name": "docs", "url": "https://x"},),
    )
    assert msg.sources[0]["source_name"] == "docs"


def test_chat_session_defaults():
    session = ChatSession(session_id="s1")
    assert session.user_id == "anonymous"
    assert session.title == "New Chat"
    assert session.created_at == 0.0
    assert session.metadata == {}


def test_chat_session_with_metadata():
    session = ChatSession(session_id="s1", user_id="u1", title="Title", metadata={"env": "prod"})
    assert session.user_id == "u1"
    assert session.title == "Title"
    assert session.metadata == {"env": "prod"}


def test_chat_session_store_protocol_is_structural():
    """Any class with the required methods satisfies the protocol (no subclassing)."""

    class FakeStore:
        async def create_session(self, session: ChatSession) -> None: ...
        async def get_session(self, session_id: str) -> ChatSession | None: ...
        async def list_sessions(self, user_id: str, limit: int = 50) -> list[ChatSession]: ...
        async def delete_session(self, session_id: str) -> None: ...
        async def get_history(self, session_id: str, max_turns: int) -> list[ChatMessage]: ...
        async def append_message(self, message: ChatMessage) -> None: ...
        async def touch_session(self, session_id: str) -> None: ...

    def _accept(store: ChatSessionStoreProtocol) -> None:
        return None

    _accept(FakeStore())  # type: ignore[arg-type]


def test_chat_models_are_frozen():
    import pytest

    msg = ChatMessage(message_id="m1", session_id="s1", role="user", content="hi")
    with pytest.raises(AttributeError):
        msg.content = "changed"  # type: ignore[misc]
