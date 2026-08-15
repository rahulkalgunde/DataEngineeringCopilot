"""Tests for the Redis hot store of chat sessions."""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import ChatMessage, ChatSession
from data_engineering_copilot.infrastructure.chat_session_store import MESSAGES_MAX_LEN, ChatSessionRedisStore
from tests.doubles.redis import _StubRedis


@pytest.fixture
def redis():
    return _StubRedis()


@pytest.fixture
def store(redis):
    return ChatSessionRedisStore(redis, ttl_seconds=3600, messages_max_len=5)


def _session(session_id: str = "s1", user_id: str = "u1", title: str = "T") -> ChatSession:
    return ChatSession(session_id=session_id, user_id=user_id, title=title, created_at=1.0, updated_at=1.0)


def _message(session_id: str = "s1", message_id: str = "m1", role: str = "user", content: str = "hi") -> ChatMessage:
    return ChatMessage(
        message_id=message_id,
        session_id=session_id,
        role=role,  # type: ignore[arg-type]
        content=content,
        timestamp=2.0,
    )


async def test_create_and_get_session(redis, store):
    session = _session()
    await store.create_session(session)
    loaded = await store.get_session("s1")
    assert loaded is not None
    assert loaded.session_id == "s1"
    assert loaded.user_id == "u1"
    assert loaded.title == "T"
    assert loaded.metadata == {}


async def test_get_session_missing_returns_none(store):
    assert await store.get_session("missing") is None


async def test_delete_session_removes_meta_and_messages(redis, store):
    session = _session()
    await store.create_session(session)
    await store.append_message(_message())
    await store.delete_session("s1")
    assert await store.get_session("s1") is None
    assert await store.get_history("s1", 10) == []


async def test_append_and_get_history_order(redis, store):
    session = _session()
    await store.create_session(session)
    for i in range(3):
        await store.append_message(_message(message_id=f"m{i}", content=f"c{i}"))
    history = await store.get_history("s1", 10)
    assert [m.content for m in history] == ["c0", "c1", "c2"]


async def test_get_history_limited_to_max_turns(redis, store):
    session = _session()
    await store.create_session(session)
    for i in range(5):
        await store.append_message(_message(message_id=f"m{i}", content=f"c{i}"))
    history = await store.get_history("s1", 2)
    assert [m.content for m in history] == ["c3", "c4"]


async def test_messages_bounded_by_ltrim(redis, store):
    session = _session()
    await store.create_session(session)
    for i in range(10):
        await store.append_message(_message(message_id=f"m{i}", content=f"c{i}"))
    history = await store.get_history("s1", 100)
    assert len(history) == 5  # messages_max_len=5


async def test_history_with_sources_roundtrip(redis, store):
    session = _session()
    await store.create_session(session)
    msg = ChatMessage(
        message_id="m1",
        session_id="s1",
        role="assistant",
        content="answer",
        sources=({"source_name": "docs", "url": "https://x"},),
        token_count=12,
        groundedness_score=0.42,
        groundedness_claims=("claim one", "claim two"),
    )
    await store.append_message(msg)
    history = await store.get_history("s1", 10)
    assert history[0].sources == ({"source_name": "docs", "url": "https://x"},)
    assert history[0].token_count == 12
    assert history[0].groundedness_score == 0.42
    assert history[0].groundedness_claims == ("claim one", "claim two")


async def test_history_groundedness_defaults_when_missing(redis, store):
    """Legacy messages without groundedness fields must deserialize to defaults."""
    session = _session()
    await store.create_session(session)
    raw = '{"message_id":"m1","session_id":"s1","role":"assistant","content":"answer","timestamp":0.0,"sources":[],"token_count":0}'
    await store._redis.rpush(store._messages_key("s1"), raw)
    history = await store.get_history("s1", 10)
    assert history[0].groundedness_score == 1.0
    assert history[0].groundedness_claims == ()


async def test_touch_session_updates_updated_at(redis, store):
    session = _session()
    await store.create_session(session)
    await store.touch_session("s1")
    loaded = await store.get_session("s1")
    assert loaded is not None
    assert loaded.updated_at >= 1.0


async def test_touch_session_missing_is_noop(store):
    await store.touch_session("missing")  # must not raise


async def test_list_sessions_returns_empty_for_redis(store):
    assert await store.list_sessions("u1") == []


def test_messages_max_len_default():
    assert MESSAGES_MAX_LEN == 100
