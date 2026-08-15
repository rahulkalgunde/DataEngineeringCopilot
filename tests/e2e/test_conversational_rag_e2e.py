"""E2E conversational RAG: two-turn continuity with real Ollama + Qdrant + Redis + PG.

Exercises the full conversational stack: durable session store (Redis hot +
Postgres cold), history-aware prompt injection across turns, and streaming
generation via ``ConversationService.chat_stream``. The first turn seeds the
session; the second turn resumes it and must see the prior exchange in its
conversation history.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import RagConfig, RawDocument
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.services.chunker import DocumentChunker
from tests.conftest import require_qdrant_and_ollama

SAMPLE_HTML = """<html><head><title>Delta Lake Guide</title></head><body>
<main>
<h1>Delta Lake Guide</h1>
<p>Delta Lake is an open-source storage framework that brings ACID transactions
to Apache Spark and big data workloads. It provides scalable metadata handling
and unifies streaming and batch data processing.</p>
<p>Delta Lake time travel allows querying previous versions of the table using
the version history and timestamp-based travel.</p>
</main></body></html>"""


async def _collect_events(stream) -> list[dict]:
    return [e async for e in stream]


@pytest.mark.e2e
class TestConversationalRagE2E:
    """Journey: Ingest → turn 1 → turn 2 resumes with history."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_infra(self, e2e_settings):
        require_qdrant_and_ollama(e2e_settings.qdrant_url)

    async def test_two_turn_continuity(self, e2e_settings, e2e_embedder, e2e_llm, e2e_redis, e2e_pg_dsn):
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
        from data_engineering_copilot.infrastructure.chat_session_store import (
            ChatSessionPostgresStore,
            ChatSessionRedisStore,
            ChatSessionStore,
        )
        from data_engineering_copilot.services.async_rag import AsyncRagService
        from data_engineering_copilot.services.conversation_rag import ConversationService

        store = None
        pg_store = None
        try:
            store = AsyncQdrantVectorStore(
                url=e2e_settings.qdrant_url,
                collection_name=e2e_settings.collection_name,
                embedding_dimension=768,
            )
            await store.initialize()

            raw = RawDocument(
                source_name="Delta Lake Documentation",
                url="https://delta.io/guide",
                html=SAMPLE_HTML,
            )
            parsed = MarkdownParser().parse(raw)
            assert parsed is not None
            chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
            chunks = await chunker.chunk(parsed)
            assert len(chunks) >= 1
            texts = [c.text for c in chunks]
            embeddings = await e2e_embedder.embed_texts(texts)
            await store.upsert_chunks(chunks, embeddings)

            rag = AsyncRagService(
                config=RagConfig(),
                vector_store=store,
                llm_client=e2e_llm,
                embedder=e2e_embedder,
            )

            redis_store = ChatSessionRedisStore(e2e_redis, ttl_seconds=3600, messages_max_len=100)
            pg_store = ChatSessionPostgresStore(e2e_pg_dsn)
            await pg_store.initialize()
            facade = ChatSessionStore(redis_store, pg_store)
            conversation = ConversationService(rag_service=rag, store=facade)

            # --- Turn 1: create session ---
            turn1_events = await _collect_events(conversation.chat_stream(None, "e2e-user", "What is Delta Lake?"))
            assert turn1_events[0]["type"] == "session_created"
            session_id = turn1_events[0]["session_id"]
            assert session_id
            assert any(e["type"] == "done" for e in turn1_events)
            done1 = next(e for e in turn1_events if e["type"] == "done")
            assert len(done1.get("text", "")) > 0

            # --- Turn 2: resume with a pronoun-referencing follow-up ---
            turn2_events = await _collect_events(
                conversation.chat_stream(session_id, "e2e-user", "Does it support time travel?")
            )
            assert any(e["type"] == "done" for e in turn2_events)
            done2 = next(e for e in turn2_events if e["type"] == "done")
            assert len(done2.get("text", "")) > 0

            # --- Continuity: history must contain both turn-1 messages ---
            history = await facade.get_history(session_id, 10)
            assert len(history) == 4  # user + assistant for each of 2 turns
            contents = [m.content for m in history]
            assert "What is Delta Lake?" in contents
            assert "Does it support time travel?" in contents
            roles = [m.role for m in history]
            assert roles == ["user", "assistant", "user", "assistant"]

            # Sessions listable for the user.
            sessions = await facade.list_sessions("e2e-user")
            assert any(s.session_id == session_id for s in sessions)
        finally:
            if store is not None:
                await store.close()
            if pg_store is not None:
                await pg_store.close()
