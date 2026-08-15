"""Tests for the ChatSessionStore facade (Redis hot + Postgres cold)."""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import ChatMessage, ChatSession
from data_engineering_copilot.infrastructure.chat_session_store import (
    ChatSessionRedisStore,
    ChatSessionStore,
)
from tests.doubles.redis import _StubRedis


class _FakePgStore:
    """In-memory stand-in for ChatSessionPostgresStore (avoids infra)."""

    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.messages: dict[str, list[ChatMessage]] = {}
        self.fail_writes = False

    async def _ensure_initialized(self) -> None: ...

    async def create_session(self, session: ChatSession) -> None:
        if self.fail_writes:
            raise RuntimeError("pg down")
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
        rows = self.messages.get(session_id, [])
        rows.sort(key=lambda m: m.timestamp)
        return rows[-max_turns:]

    async def append_message(self, message: ChatMessage) -> None:
        if self.fail_writes:
            raise RuntimeError("pg down")
        self.messages.setdefault(message.session_id, []).append(message)

    async def touch_session(self, session_id: str) -> None:
        if self.fail_writes:
            raise RuntimeError("pg down")
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


@pytest.fixture
def facade() -> ChatSessionStore:
    redis_store = ChatSessionRedisStore(_StubRedis(), ttl_seconds=3600, messages_max_len=10)
    return ChatSessionStore(redis_store, _FakePgStore())  # type: ignore[arg-type]


def _session(session_id: str = "s1", user_id: str = "u1") -> ChatSession:
    return ChatSession(session_id=session_id, user_id=user_id, title="T", created_at=1.0, updated_at=1.0)


def _message(session_id: str = "s1", message_id: str = "m1", content: str = "hi") -> ChatMessage:
    return ChatMessage(
        message_id=message_id,
        session_id=session_id,
        role="user",
        content=content,
        timestamp=2.0,
    )


async def test_get_session_from_redis_without_pg(facade):
    await facade._redis.create_session(_session())  # noqa: SLF001
    loaded = await facade.get_session("s1")
    assert loaded is not None
    assert loaded.session_id == "s1"


async def test_get_session_rehydrates_from_pg(facade):
    await facade._pg.create_session(_session())  # noqa: SLF001
    loaded = await facade.get_session("s1")
    assert loaded is not None
    # Rehydrated into Redis cache.
    assert await facade._redis.get_session("s1") is not None  # noqa: SLF001


async def test_get_history_from_redis(facade):
    session = _session()
    await facade.create_session(session)
    await facade.append_message(_message())
    history = await facade.get_history("s1", 10)
    assert [m.content for m in history] == ["hi"]


async def test_get_history_rehydrates_from_pg(facade):
    session = _session()
    await facade._pg.create_session(session)  # noqa: SLF001
    await facade._pg.append_message(_message())  # noqa: SLF001
    history = await facade.get_history("s1", 10)
    assert [m.content for m in history] == ["hi"]
    # Write-through into Redis.
    assert await facade._redis.get_history("s1", 10)  # noqa: SLF001


async def test_append_message_fail_open_on_pg(facade):
    facade._pg.fail_writes = True  # noqa: SLF001
    await facade.append_message(_message())  # must not raise
    assert await facade._redis.get_history("s1", 10)  # noqa: SLF001


async def test_delete_session_removes_both(facade):
    session = _session()
    await facade.create_session(session)
    await facade.append_message(_message())
    await facade.delete_session("s1")
    assert await facade.get_session("s1") is None
    assert await facade.get_history("s1", 10) == []


async def test_list_sessions_delegates_to_pg(facade):
    await facade._pg.create_session(_session("a", "u1"))  # noqa: SLF001
    await facade._pg.create_session(_session("b", "u1"))  # noqa: SLF001
    await facade._pg.create_session(_session("c", "u2"))  # noqa: SLF001
    sessions = await facade.list_sessions("u1")
    assert {s.session_id for s in sessions} == {"a", "b"}


async def test_facade_satisfies_store_protocol(facade):
    from data_engineering_copilot.domain.protocols import ChatSessionStoreProtocol

    def _accept(store: ChatSessionStoreProtocol) -> None:
        return None

    _accept(facade)  # type: ignore[arg-type]
