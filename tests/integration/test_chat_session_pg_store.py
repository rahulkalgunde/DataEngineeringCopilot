"""Tests for ChatSessionPostgresStore.

These tests require a running PostgreSQL instance, provided by a session-scoped
Postgres testcontainer. Marked ``serial`` because the tables are shared.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from data_engineering_copilot.domain.models import ChatMessage, ChatSession
from data_engineering_copilot.infrastructure.chat_session_store import ChatSessionPostgresStore

pytestmark = [pytest.mark.integration, pytest.mark.serial]


@pytest_asyncio.fixture
async def pg_store(pg_dsn):
    store = ChatSessionPostgresStore(pg_dsn)
    try:
        await store.initialize()
    except Exception:
        pytest.skip("PostgreSQL unreachable")
    # Isolate tests: drop shared rows left by prior tests in this session.
    assert store._pool is not None  # noqa: SLF001
    async with store._pool.acquire() as conn:  # noqa: SLF001
        await conn.execute("DELETE FROM chat_messages")
        await conn.execute("DELETE FROM chat_sessions")
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(pg_store):
    assert pg_store.is_initialized
    async with pg_store._pool.acquire() as conn:  # noqa: SLF001
        tables = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        table_names = {row["tablename"] for row in tables}
        assert "chat_sessions" in table_names
        assert "chat_messages" in table_names


@pytest.mark.asyncio
async def test_create_and_get_session(pg_store):
    session = ChatSession(session_id="s1", user_id="u1", title="T", created_at=1.0, updated_at=1.0)
    await pg_store.create_session(session)
    loaded = await pg_store.get_session("s1")
    assert loaded is not None
    assert loaded.session_id == "s1"
    assert loaded.user_id == "u1"
    assert loaded.title == "T"


@pytest.mark.asyncio
async def test_get_session_missing(pg_store):
    assert await pg_store.get_session("missing") is None


@pytest.mark.asyncio
async def test_append_and_get_history(pg_store):
    session = ChatSession(session_id="s2", user_id="u1", created_at=1.0, updated_at=1.0)
    await pg_store.create_session(session)
    for i in range(3):
        await pg_store.append_message(
            ChatMessage(
                message_id=f"m{i}",
                session_id="s2",
                role="user",
                content=f"c{i}",
                timestamp=float(i),
            )
        )
    history = await pg_store.get_history("s2", 10)
    assert [m.content for m in history] == ["c0", "c1", "c2"]


@pytest.mark.asyncio
async def test_history_limited_and_ordered(pg_store):
    session = ChatSession(session_id="s3", user_id="u1", created_at=1.0, updated_at=1.0)
    await pg_store.create_session(session)
    for i in range(5):
        await pg_store.append_message(
            ChatMessage(
                message_id=f"m{i}",
                session_id="s3",
                role="user",
                content=f"c{i}",
                timestamp=float(i),
            )
        )
    history = await pg_store.get_history("s3", 2)
    assert [m.content for m in history] == ["c3", "c4"]


@pytest.mark.asyncio
async def test_sources_and_token_count_roundtrip(pg_store):
    session = ChatSession(session_id="s4", user_id="u1", created_at=1.0, updated_at=1.0)
    await pg_store.create_session(session)
    await pg_store.append_message(
        ChatMessage(
            message_id="m1",
            session_id="s4",
            role="assistant",
            content="answer",
            timestamp=2.0,
            sources=({"source_name": "docs", "url": "https://x"},),
            token_count=7,
        )
    )
    history = await pg_store.get_history("s4", 10)
    assert history[0].sources == ({"source_name": "docs", "url": "https://x"},)
    assert history[0].token_count == 7


@pytest.mark.asyncio
async def test_list_sessions_ordered_by_updated_at(pg_store):
    await pg_store.create_session(ChatSession(session_id="u1a", user_id="u1", updated_at=1.0, created_at=1.0))
    await pg_store.create_session(ChatSession(session_id="u1b", user_id="u1", updated_at=3.0, created_at=1.0))
    await pg_store.create_session(ChatSession(session_id="other", user_id="u2", updated_at=2.0, created_at=1.0))
    sessions = await pg_store.list_sessions("u1")
    assert [s.session_id for s in sessions] == ["u1b", "u1a"]


@pytest.mark.asyncio
async def test_delete_session_cascades_messages(pg_store):
    session = ChatSession(session_id="s5", user_id="u1", created_at=1.0, updated_at=1.0)
    await pg_store.create_session(session)
    await pg_store.append_message(
        ChatMessage(message_id="m1", session_id="s5", role="user", content="hi", timestamp=1.0)
    )
    await pg_store.delete_session("s5")
    assert await pg_store.get_session("s5") is None
    assert await pg_store.get_history("s5", 10) == []


@pytest.mark.asyncio
async def test_touch_session_updates_updated_at(pg_store):
    session = ChatSession(session_id="s6", user_id="u1", created_at=1.0, updated_at=1.0)
    await pg_store.create_session(session)
    await pg_store.touch_session("s6")
    loaded = await pg_store.get_session("s6")
    assert loaded is not None
    assert loaded.updated_at >= 1.0
