"""Conversational RAG orchestration service.

Composes the ``AsyncRagService`` streaming pipeline with the durable chat
session store. Responsibilities:
- create/resume chat sessions (rule-based title from the first message);
- persist user + assistant turns (assistant persisted even on client
  disconnect so history stays coherent);
- expose session list/get/delete for the chat UI and API.

The actual retrieval/streaming lives in ``AsyncRagService.chat_stream``;
this service owns session lifecycle and persistence only.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence

from data_engineering_copilot.domain.models import ChatMessage, ChatSession
from data_engineering_copilot.infrastructure.chat_session_store import ChatSessionStore
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.prompt_builder import CHAT_SYSTEM_ROLE, PromptBuilder

logger = logging.getLogger(__name__)

_MAX_TITLE_CHARS = 60


def _parse_sse_payload(line: str) -> dict | None:
    """Parse a ``data: {...}`` SSE line into a dict, or None if not a payload."""
    if not line.startswith("data: "):
        return None
    try:
        return json.loads(line[len("data: ") :])
    except json.JSONDecodeError:
        return None


def _default_title(question: str, max_chars: int = _MAX_TITLE_CHARS) -> str:
    """Rule-based session title from the first user message."""
    title = " ".join(question.split())
    if len(title) > max_chars:
        title = title[: max_chars - 1].rstrip() + "…"
    return title or "New Chat"


def _message_id() -> str:
    return uuid.uuid4().hex


class ConversationService:
    """Orchestrates multi-turn chat sessions over the RAG streaming pipeline."""

    def __init__(
        self,
        rag_service: AsyncRagService,
        store: ChatSessionStore,
        title_max_chars: int = _MAX_TITLE_CHARS,
        local_query_rewriter=None,
        local_scope_verifier=None,
        local_llm_client=None,
        answer_local: bool = False,
        local_reranker=None,
        blocked_url_substrings: Sequence[str] | None = None,
        domain_sources: Sequence[str] | None = None,
        system_role: str | None = None,
    ) -> None:
        self._rag_service = rag_service
        self._store = store
        self._title_max_chars = title_max_chars
        self._prompt_builder = PromptBuilder()
        self._local_query_rewriter = local_query_rewriter
        self._local_scope_verifier = local_scope_verifier
        self._local_llm_client = local_llm_client
        self._answer_local = answer_local
        self._local_reranker = local_reranker
        self._blocked_url_substrings = tuple(blocked_url_substrings or ())
        self._domain_sources = tuple(domain_sources or ())
        self._system_role = system_role or CHAT_SYSTEM_ROLE

    async def start_or_resume(
        self,
        session_id: str | None,
        user_id: str,
        *,
        first_message: str | None = None,
    ) -> ChatSession:
        """Return the session, creating it (with a title) if missing."""
        if session_id:
            existing = await self._store.get_session(session_id)
            if existing is not None:
                return existing
        created = ChatSession(
            session_id=session_id or uuid.uuid4().hex,
            user_id=user_id or "anonymous",
            title=_default_title(first_message or "", self._title_max_chars),
            created_at=time.time(),
            updated_at=time.time(),
        )
        await self._store.create_session(created)
        return created

    async def get_history(self, session_id: str, max_turns: int) -> list[ChatMessage]:
        return await self._store.get_history(session_id, max_turns)

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[ChatSession]:
        return await self._store.list_sessions(user_id, limit=limit)

    async def get_session(self, session_id: str) -> ChatSession | None:
        return await self._store.get_session(session_id)

    async def delete_session(self, session_id: str) -> None:
        await self._store.delete_session(session_id)

    async def chat_stream(
        self,
        session_id: str | None,
        user_id: str,
        message: str,
        *,
        source_filter: list[str] | None = None,
        max_history_turns: int = 10,
        max_history_tokens: int = 2048,
        cache_scope=None,
    ) -> AsyncIterator[dict]:
        """Stream a conversational turn, persisting user + assistant messages.

        Async generator yielding ``{"type": ...}`` SSE payloads (the same event
        contract as ``AsyncRagService.chat_stream``) plus a ``session_created``
        event with the resolved ``session_id``/``title``. The assistant message
        is persisted after the ``done`` event so it survives client disconnects.
        """
        session = await self.start_or_resume(session_id, user_id, first_message=message)

        history = await self.get_history(session.session_id, max_history_turns)

        user_message = ChatMessage(
            message_id=_message_id(),
            session_id=session.session_id,
            role="user",
            content=message,
            timestamp=time.time(),
        )
        await self._store.append_message(user_message)
        await self._store.touch_session(session.session_id)

        yield {
            "type": "session_created",
            "session_id": session.session_id,
            "title": session.title,
        }

        assistant_message = ChatMessage(
            message_id=_message_id(),
            session_id=session.session_id,
            role="assistant",
            content="",
            timestamp=0.0,
        )

        try:
            async for sse_line in self._rag_service.chat_stream(
                message,
                source_filter=source_filter,
                user_id=user_id,
                session_id=session.session_id,
                conversation_history=history,
                max_history_tokens=max_history_tokens,
                cache_scope=cache_scope,
                chat_query_rewriter=self._local_query_rewriter,
                chat_scope_verifier=self._local_scope_verifier,
                chat_llm_client=self._local_llm_client if self._answer_local else None,
                chat_reranker=self._local_reranker,
                chat_blocked_url_substrings=self._blocked_url_substrings,
                chat_domain_sources=self._domain_sources,
                chat_system_role=self._system_role,
            ):
                payload = _parse_sse_payload(sse_line)
                if payload is None:
                    continue
                if payload.get("type") == "done":
                    assistant_text = payload.get("text", "")
                    assistant_message = ChatMessage(
                        message_id=assistant_message.message_id,
                        session_id=assistant_message.session_id,
                        role="assistant",
                        content=assistant_text,
                        timestamp=time.time(),
                        token_count=_count_tokens(assistant_text),
                    )
                yield payload
        finally:
            # Persist the assistant turn even on disconnect / mid-stream error
            # so history stays coherent.
            if not assistant_message.content:
                assistant_message = ChatMessage(
                    message_id=assistant_message.message_id,
                    session_id=assistant_message.session_id,
                    role="assistant",
                    content="(interrupted)",
                    timestamp=time.time(),
                )
            try:
                await self._store.append_message(assistant_message)
                await self._store.touch_session(session.session_id)
            except Exception:
                logger.warning("chat.conversation.persist_failed session_id=%s", session.session_id, exc_info=True)

    async def close(self) -> None:
        """Best-effort teardown of the backing PG store (pool close)."""
        store = self._store
        if hasattr(store, "_pg"):
            pg = store._pg
            if hasattr(pg, "close"):
                await pg.close()


def _count_tokens(text: str) -> int:
    from data_engineering_copilot.infrastructure.token_budget import count_tokens

    return count_tokens(text)
