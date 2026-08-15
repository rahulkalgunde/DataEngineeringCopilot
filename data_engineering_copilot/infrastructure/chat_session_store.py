"""Redis + Postgres stores for multi-turn chat sessions.

Sessions and recent messages are cached in Redis (bounded, with TTL) so the
conversational pipeline never touches Postgres on the hot path. Postgres is the
durable source of truth; the ``ChatSessionStore`` facade rehydrates from it on
a Redis miss.
"""

from __future__ import annotations

import json
import logging
import time

import asyncpg

from data_engineering_copilot.domain.models import ChatMessage, ChatSession

log = logging.getLogger(__name__)

META_KEY_TEMPLATE = "chat:session:{session_id}:meta"
MESSAGES_KEY_TEMPLATE = "chat:session:{session_id}:messages"

# Maximum number of messages retained per session in the Redis hot cache.
MESSAGES_MAX_LEN = 100

CHAT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources JSONB,
    token_count INTEGER NOT NULL DEFAULT 0,
    groundedness_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    groundedness_claims JSONB,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, created_at);
"""


class ChatSessionRedisStore:
    """Bounded, TTL'd Redis cache for chat sessions and recent messages.

    Uses a shared async Redis client (see ``factory.get_shared_redis_client``)
    and never closes it. Message history is stored as a Redis list; the most
    recent ``MESSAGES_MAX_LEN`` messages are kept via ``RPUSH`` + ``LTRIM``.
    """

    def __init__(self, redis, ttl_seconds: int = 259200, messages_max_len: int = MESSAGES_MAX_LEN) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds
        self._messages_max_len = messages_max_len

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return META_KEY_TEMPLATE.format(session_id=session_id)

    @staticmethod
    def _messages_key(session_id: str) -> str:
        return MESSAGES_KEY_TEMPLATE.format(session_id=session_id)

    @staticmethod
    def _serialize_message(message: ChatMessage) -> str:
        return json.dumps(
            {
                "message_id": message.message_id,
                "session_id": message.session_id,
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp,
                "sources": list(message.sources),
                "token_count": message.token_count,
                "groundedness_score": message.groundedness_score,
                "groundedness_claims": list(message.groundedness_claims),
            }
        )

    @classmethod
    def _deserialize_message(cls, raw: str) -> ChatMessage:
        data = json.loads(raw)
        return ChatMessage(
            message_id=data["message_id"],
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", 0.0),
            sources=tuple(data.get("sources", ())),
            token_count=data.get("token_count", 0),
            groundedness_score=data.get("groundedness_score", 1.0),
            groundedness_claims=tuple(data.get("groundedness_claims", ())),
        )

    @staticmethod
    def _serialize_session(session: ChatSession) -> str:
        return json.dumps(
            {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "metadata": session.metadata,
            }
        )

    @classmethod
    def _deserialize_session(cls, raw: str) -> ChatSession:
        data = json.loads(raw)
        return ChatSession(
            session_id=data["session_id"],
            user_id=data.get("user_id", "anonymous"),
            title=data.get("title", "New Chat"),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            metadata=data.get("metadata", {}),
        )

    async def create_session(self, session: ChatSession) -> None:
        await self._redis.set(self._meta_key(session.session_id), self._serialize_session(session))
        await self._redis.expire(self._meta_key(session.session_id), self._ttl_seconds)

    async def get_session(self, session_id: str) -> ChatSession | None:
        raw = await self._redis.get(self._meta_key(session_id))
        if raw is None:
            return None
        try:
            return self._deserialize_session(raw)
        except (json.JSONDecodeError, KeyError):
            log.warning("chat.redis.invalid_session_meta session_id=%s", session_id)
            return None

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[ChatSession]:
        # Redis is a cache, not the source of truth for session enumeration —
        # the facade lists from Postgres and this store is only for hot-path
        # session/message reads and writes.
        return []

    async def delete_session(self, session_id: str) -> None:
        await self._redis.delete(self._meta_key(session_id), self._messages_key(session_id))

    async def get_history(self, session_id: str, max_turns: int) -> list[ChatMessage]:
        if max_turns <= 0:
            return []
        raw_messages = await self._redis.lrange(self._messages_key(session_id), -max_turns, -1)
        messages: list[ChatMessage] = []
        for raw in raw_messages:
            try:
                messages.append(self._deserialize_message(raw))
            except (json.JSONDecodeError, KeyError):
                log.warning("chat.redis.invalid_message session_id=%s", session_id)
        return messages

    async def append_message(self, message: ChatMessage) -> None:
        key = self._messages_key(message.session_id)
        await self._redis.rpush(key, self._serialize_message(message))
        await self._redis.ltrim(key, -self._messages_max_len, -1)
        await self._redis.expire(key, self._ttl_seconds)

    async def touch_session(self, session_id: str) -> None:
        raw = await self._redis.get(self._meta_key(session_id))
        if raw is None:
            return
        try:
            session = self._deserialize_session(raw)
            session = ChatSession(
                session_id=session.session_id,
                user_id=session.user_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=time.time(),
                metadata=session.metadata,
            )
            await self.create_session(session)
        except (json.JSONDecodeError, KeyError):
            log.warning("chat.redis.touch_invalid_session session_id=%s", session_id)


class ChatSessionPostgresStore:
    """Durable Postgres store for chat sessions and messages.

    Mirrors the ``crawl_db.py`` lifecycle: an ``asyncpg`` connection pool is
    created lazily by :meth:`initialize` (which also ensures the schema exists)
    and closed by :meth:`close`. DSN comes from ``settings.chat_db_url`` (the
    caller resolves the fallback to ``crawl_db_url``).
    """

    def __init__(self, dsn: str, pool_min_size: int = 2, pool_max_size: int = 10) -> None:
        self._dsn = dsn
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: asyncpg.Pool | None = None

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            from data_engineering_copilot.domain.exceptions import CrawlError

            raise CrawlError("ChatSessionPostgresStore not initialized. Call initialize() before use.")
        return self._pool

    @property
    def is_initialized(self) -> bool:
        return self._pool is not None

    async def _ensure_initialized(self) -> None:
        """Create the pool lazily on first use (sync factory builds the store)."""
        if self._pool is None:
            await self.initialize()

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(CHAT_SCHEMA_SQL)
        log.info("chat.pg.initialized dsn=%s", self._dsn)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict:
        return {
            "message_id": message.message_id,
            "session_id": message.session_id,
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
            "sources": list(message.sources),
            "token_count": message.token_count,
            "groundedness_score": message.groundedness_score,
            "groundedness_claims": list(message.groundedness_claims),
        }

    @classmethod
    def _deserialize_message(cls, data: dict) -> ChatMessage:
        sources = data.get("sources", ())
        if isinstance(sources, str):
            try:
                sources = json.loads(sources)
            except json.JSONDecodeError:
                sources = ()
        claims = data.get("groundedness_claims", ())
        if isinstance(claims, str):
            try:
                claims = json.loads(claims)
            except json.JSONDecodeError:
                claims = ()
        return ChatMessage(
            message_id=data["message_id"],
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", 0.0),
            sources=tuple(sources),
            token_count=data.get("token_count", 0),
            groundedness_score=data.get("groundedness_score", 1.0),
            groundedness_claims=tuple(claims),
        )

    @staticmethod
    def _serialize_session(session: ChatSession) -> dict:
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "metadata": session.metadata,
        }

    @classmethod
    def _deserialize_session(cls, data: dict) -> ChatSession:
        metadata = data.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        return ChatSession(
            session_id=data["session_id"],
            user_id=data.get("user_id", "anonymous"),
            title=data.get("title", "New Chat"),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            metadata=metadata,
        )

    async def create_session(self, session: ChatSession) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO chat_sessions
                   (session_id, user_id, title, created_at, updated_at, metadata)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (session_id) DO NOTHING""",
                session.session_id,
                session.user_id,
                session.title,
                session.created_at,
                session.updated_at,
                json.dumps(session.metadata),
            )

    async def get_session(self, session_id: str) -> ChatSession | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM chat_sessions WHERE session_id = $1", session_id)
            if row is None:
                return None
            return self._deserialize_session(dict(row))

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[ChatSession]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM chat_sessions WHERE user_id = $1 ORDER BY updated_at DESC LIMIT $2",
                user_id,
                limit,
            )
            return [self._deserialize_session(dict(row)) for row in rows]

    async def delete_session(self, session_id: str) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM chat_sessions WHERE session_id = $1", session_id)

    async def get_history(self, session_id: str, max_turns: int) -> list[ChatMessage]:
        if max_turns <= 0:
            return []
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM chat_messages WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
                session_id,
                max_turns,
            )
        messages = [self._deserialize_message(dict(row)) for row in reversed(rows)]
        messages.sort(key=lambda m: m.timestamp)
        return messages

    async def append_message(self, message: ChatMessage) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO chat_messages
                   (message_id, session_id, role, content, sources, token_count,
                    groundedness_score, groundedness_claims, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                   ON CONFLICT (message_id) DO NOTHING""",
                message.message_id,
                message.session_id,
                message.role,
                message.content,
                json.dumps(list(message.sources)),
                message.token_count,
                message.groundedness_score,
                json.dumps(list(message.groundedness_claims)),
                message.timestamp,
            )

    async def touch_session(self, session_id: str) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE chat_sessions SET updated_at = $1 WHERE session_id = $2",
                time.time(),
                session_id,
            )


class ChatSessionStore:
    """Facade combining the Redis hot cache with the durable Postgres store.

    Read path: Redis first (hot path), falling back to Postgres on a miss with
    write-through back into Redis. Write path: Redis + best-effort Postgres
    (fail-open — a PG outage never blocks a chat turn; errors are logged).
    """

    def __init__(self, redis_store: ChatSessionRedisStore, pg_store: ChatSessionPostgresStore) -> None:
        self._redis = redis_store
        self._pg = pg_store

    async def _ensure_pg(self) -> None:
        await self._pg._ensure_initialized()  # noqa: SLF001

    async def create_session(self, session: ChatSession) -> None:
        await self._redis.create_session(session)
        await self._ensure_pg()
        await self._pg.create_session(session)

    async def get_session(self, session_id: str) -> ChatSession | None:
        cached = await self._redis.get_session(session_id)
        if cached is not None:
            return cached
        await self._ensure_pg()
        session = await self._pg.get_session(session_id)
        if session is not None:
            await self._redis.create_session(session)
        return session

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[ChatSession]:
        await self._ensure_pg()
        return await self._pg.list_sessions(user_id, limit=limit)

    async def delete_session(self, session_id: str) -> None:
        await self._redis.delete_session(session_id)
        await self._ensure_pg()
        await self._pg.delete_session(session_id)

    async def get_history(self, session_id: str, max_turns: int) -> list[ChatMessage]:
        cached = await self._redis.get_history(session_id, max_turns)
        if cached:
            return cached
        await self._ensure_pg()
        messages = await self._pg.get_history(session_id, max_turns)
        for message in messages:
            await self._redis.append_message(message)
        return messages

    async def append_message(self, message: ChatMessage) -> None:
        await self._redis.append_message(message)
        try:
            await self._ensure_pg()
            await self._pg.append_message(message)
        except Exception:
            log.warning("chat.facade.pg_append_failed message_id=%s", message.message_id, exc_info=True)

    async def touch_session(self, session_id: str) -> None:
        await self._redis.touch_session(session_id)
        try:
            await self._ensure_pg()
            await self._pg.touch_session(session_id)
        except Exception:
            log.warning("chat.facade.pg_touch_failed session_id=%s", session_id, exc_info=True)
