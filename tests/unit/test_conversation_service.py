"""Tests for the ConversationService orchestration."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from data_engineering_copilot.domain.models import ChatMessage, ChatSession
from data_engineering_copilot.infrastructure.chat_session_store import (
    ChatSessionRedisStore,
    ChatSessionStore,
)
from data_engineering_copilot.services.conversation_rag import ConversationService, _default_title
from tests.doubles.redis import _StubRedis


class _FakePgStore:
    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.messages: dict[str, list[ChatMessage]] = {}
        self.closed = False

    async def _ensure_initialized(self) -> None: ...

    async def create_session(self, session: ChatSession) -> None:
        self.sessions.setdefault(session.session_id, session)

    async def get_session(self, session_id: str) -> ChatSession | None:
        return self.sessions.get(session_id)

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[ChatSession]:
        rows = [s for s in self.sessions.values() if s.user_id == user_id]
        rows.sort(key=lambda s: s.updated_at, reverse=True)
        return rows[:limit]

    async def delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.messages.pop(session_id, None)

    async def get_history(self, session_id: str, max_turns: int) -> list[ChatMessage]:
        rows = sorted(self.messages.get(session_id, []), key=lambda m: m.timestamp)
        return rows[-max_turns:]

    async def append_message(self, message: ChatMessage) -> None:
        self.messages.setdefault(message.session_id, []).append(message)

    async def touch_session(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            self.sessions[session_id] = ChatSession(
                session_id=session.session_id,
                user_id=session.user_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=99.0,
                metadata=session.metadata,
            )

    async def close(self) -> None:
        self.closed = True


class _FakeRagService:
    def __init__(self, events: list[dict]) -> None:
        self._events = events
        self.received_history: list[ChatMessage] | None = None
        self.received_rewriter = None
        self.received_scope_verifier = None
        self.received_reranker = None
        self.received_blocked = []
        self.received_domain = []
        self.received_system_role = None

    async def chat_stream(
        self,
        question: str,
        source_filter=None,
        user_id=None,
        session_id=None,
        conversation_history=None,
        max_history_tokens: int = 2048,
        cache_scope=None,
        chat_query_rewriter=None,
        chat_scope_verifier=None,
        chat_llm_client=None,
        chat_reranker=None,
        chat_blocked_url_substrings=None,
        chat_domain_sources=None,
        chat_system_role=None,
    ) -> AsyncIterator[str]:
        self.received_history = list(conversation_history or [])
        self.received_rewriter = chat_query_rewriter
        self.received_scope_verifier = chat_scope_verifier
        self.received_reranker = chat_reranker
        self.received_blocked = list(chat_blocked_url_substrings or [])
        self.received_domain = list(chat_domain_sources or [])
        self.received_system_role = chat_system_role
        for event in self._events:
            yield f"data: {json.dumps(event)}\n\n"


def _build_service(rag_events: list[dict]) -> tuple[ConversationService, _FakePgStore, _FakeRagService]:
    redis_store = ChatSessionRedisStore(_StubRedis(), ttl_seconds=3600, messages_max_len=100)
    pg_store = _FakePgStore()
    store = ChatSessionStore(redis_store, pg_store)  # type: ignore[arg-type]
    rag = _FakeRagService(rag_events)
    service = ConversationService(rag_service=rag, store=store)  # type: ignore[arg-type]
    return service, pg_store, rag


async def _collect(stream) -> list[dict]:
    return [e async for e in stream]


@pytest.mark.asyncio
async def test_start_or_resume_creates_session_with_title():
    service, pg, _ = _build_service([])
    session = await service.start_or_resume(None, "u1", first_message="How does filter work on arrays?")
    assert session.session_id
    assert session.user_id == "u1"
    assert session.title == "How does filter work on arrays?"
    assert await pg.get_session(session.session_id) is not None


@pytest.mark.asyncio
async def test_start_or_resume_reuses_existing_session():
    service, pg, _ = _build_service([])
    created = await service.start_or_resume(None, "u1", first_message="hello")
    resumed = await service.start_or_resume(created.session_id, "u1")
    assert resumed.session_id == created.session_id


@pytest.mark.asyncio
async def test_default_title_truncates():
    long_q = " ".join(["word"] * 100)
    title = _default_title(long_q, max_chars=20)
    assert len(title) <= 20


@pytest.mark.asyncio
async def test_default_title_fallback():
    assert _default_title("") == "New Chat"
    assert _default_title("   ") == "New Chat"


@pytest.mark.asyncio
async def test_chat_stream_persists_both_turns_and_emits_session_created():
    service, pg, rag = _build_service(
        [
            {"type": "token", "content": "Delta Lake "},
            {"type": "done", "text": "Delta Lake brings ACID.", "confidence": 0.9},
        ]
    )
    events = await _collect(service.chat_stream(None, "u1", "What is Delta Lake?"))

    assert events[0]["type"] == "session_created"
    assert events[0]["session_id"]
    assert "Delta Lake brings ACID." in [e.get("text", "") for e in events]

    # Both user + assistant turns persisted.
    sid = events[0]["session_id"]
    history = await pg.get_history(sid, 10)
    roles = [m.role for m in history]
    assert roles == ["user", "assistant"]
    assert history[0].content == "What is Delta Lake?"
    assert history[1].content == "Delta Lake brings ACID."


@pytest.mark.asyncio
async def test_chat_stream_passes_history_to_rag():
    service, pg, rag = _build_service([{"type": "done", "text": "answer", "confidence": 0.5}])
    sid = "existing-session"
    await service.start_or_resume(sid, "u1", first_message="q1")
    await pg.append_message(ChatMessage(message_id="m1", session_id=sid, role="user", content="q1", timestamp=1.0))
    await pg.append_message(ChatMessage(message_id="m2", session_id=sid, role="assistant", content="a1", timestamp=2.0))
    await _collect(service.chat_stream(sid, "u1", "q2"))
    assert rag.received_history is not None
    assert [m.content for m in rag.received_history] == ["q1", "a1"]


@pytest.mark.asyncio
async def test_chat_stream_persists_assistant_on_mid_stream_error():
    service, pg, rag = _build_service([{"type": "error", "message": "Generation failed"}])
    events = await _collect(service.chat_stream(None, "u1", "question"))
    sid = events[0]["session_id"]
    history = await pg.get_history(sid, 10)
    # Assistant turn persisted with the interruption marker.
    assert history[-1].role == "assistant"
    assert history[-1].content == "(interrupted)"


@pytest.mark.asyncio
async def test_delete_and_list_sessions():
    service, pg, _ = _build_service([])
    s1 = await service.start_or_resume(None, "u1", first_message="one")
    s2 = await service.start_or_resume(None, "u1", first_message="two")
    sessions = await service.list_sessions("u1")
    assert {s.session_id for s in sessions} == {s1.session_id, s2.session_id}
    await service.delete_session(s1.session_id)
    assert await service.get_session(s1.session_id) is None
    assert await service.get_session(s2.session_id) is not None


@pytest.mark.asyncio
async def test_close_closes_pg_store():
    service, pg, _ = _build_service([])
    await service.close()
    assert pg.closed


@pytest.mark.asyncio
async def test_local_components_forwarded_to_rag():
    """Local rewriter/scope-verifier injected into the ConversationService must
    be forwarded to the RAG chat_stream call."""
    redis_store = ChatSessionRedisStore(_StubRedis(), ttl_seconds=3600, messages_max_len=100)
    pg_store = _FakePgStore()
    store = ChatSessionStore(redis_store, pg_store)  # type: ignore[arg-type]
    rag = _FakeRagService([{"type": "done", "text": "answer", "confidence": 0.5}])

    class _Sentinel:
        pass

    rewriter = _Sentinel()
    scope = _Sentinel()
    reranker = _Sentinel()
    service = ConversationService(
        rag_service=rag,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        local_query_rewriter=rewriter,
        local_scope_verifier=scope,
        local_llm_client=_Sentinel(),
        answer_local=True,
        local_reranker=reranker,
        blocked_url_substrings=["system-prompts.md"],
        domain_sources=["Apache Spark 4.0.0"],
        system_role="custom persona",
    )
    await _collect(service.chat_stream(None, "u1", "q"))
    assert rag.received_rewriter is rewriter
    assert rag.received_scope_verifier is scope
    assert rag.received_reranker is reranker
    assert rag.received_blocked == ["system-prompts.md"]
    assert rag.received_domain == ["Apache Spark 4.0.0"]
    assert rag.received_system_role == "custom persona"
