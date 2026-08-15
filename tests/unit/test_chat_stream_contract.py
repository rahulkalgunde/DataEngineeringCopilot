"""Contract tests for ``AsyncRagService.chat_stream`` (conversational RAG).

Executes the *real* ``chat_stream`` body against deterministic doubles so the
isolated multi-turn pipeline is covered: contextual rewriting, multi-query
retrieval, history-injected prompt, streaming generation, and the no-cache
contract.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from data_engineering_copilot.domain.models import ChatMessage, DocumentChunk, RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService
from tests.doubles.embedder import StubEmbedder
from tests.doubles.llm import STUB_ANSWER, StubLLM
from tests.doubles.vector_store import InMemoryVectorStore

_TOPICS = [
    (
        "Apache Spark",
        "Apache Spark is a unified analytics engine for large-scale data processing. "
        "It provides high-level APIs in Scala, Java, Python, and R.",
    ),
    (
        "Delta Lake",
        "Delta Lake is an open-source storage framework that brings ACID transactions "
        "to Apache Spark and big data workloads.",
    ),
    (
        "Apache Airflow",
        "Apache Airflow is a platform to programmatically author, schedule and monitor workflows defined as code.",
    ),
]


def _build_chunks() -> list[DocumentChunk]:
    chunks = []
    for i, (title, text) in enumerate(_TOPICS):
        chunks.append(
            DocumentChunk(
                chunk_id=f"chat:doc{i:03d}:chunk00",
                source_name="RAG Test Docs",
                title=title,
                url=f"https://example.com/docs/{title.lower().replace(' ', '-')}.html",
                text=text,
            )
        )
    return chunks


class _RecordingLLM:
    """LLM double capturing every prompt for prompt-content assertions."""

    def __init__(self, answer: str = STUB_ANSWER) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.answer

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        text = self.answer
        for token in text.split(" "):
            yield f"{token} "

    @property
    def last_usage(self) -> Any:
        return None

    async def close(self) -> None:
        pass


def _chat_message(role: str, content: str) -> ChatMessage:
    return ChatMessage(message_id="m", session_id="s", role=role, content=content)  # type: ignore[arg-type]


def _collect_events(stream):
    import json

    events = []
    for line in stream:
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[len("data: ") :]))
    return events


async def _build_service(rewriter_llm, generation_llm) -> AsyncRagService:
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    service = AsyncRagService(
        config=RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.05,
            max_context_chars=2000,
            chat_suggestions_enabled=False,
        ),
        vector_store=store,
        llm_client=generation_llm,
        embedder=embedder,
        query_rewriter=QueryRewriter(llm_client=rewriter_llm, enabled=True, hyde_enabled=False),
    )
    return service


@pytest.mark.asyncio
async def test_chat_stream_emits_status_sources_token_done():
    service = await _build_service(_RecordingLLM(), _RecordingLLM())
    events = _collect_events([e async for e in service.chat_stream("What is Apache Spark?")])
    types = [e["type"] for e in events]

    assert types[0] == "status"
    assert "sources" in types, f"Expected a sources event, got {types}"
    assert "token" in types, f"Expected token events, got {types}"
    assert "done" in types, f"Expected a done event, got {types}"

    done = next(e for e in events if e["type"] == "done")
    assert isinstance(done["text"], str) and len(done["text"]) > 0
    assert isinstance(done["confidence"], float)

    sources = next(e for e in events if e["type"] == "sources")
    assert isinstance(sources["sources"], list) and len(sources["sources"]) >= 1
    assert "source_name" in sources["sources"][0]
    assert "url" in sources["sources"][0]

    # sources must precede the first token
    first_token_idx = types.index("token")
    sources_idx = types.index("sources")
    assert sources_idx < first_token_idx
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_turn1_no_history_in_prompt():
    gen = _RecordingLLM()
    service = await _build_service(_RecordingLLM(), gen)
    _collect_events([e async for e in service.chat_stream("What is Apache Spark?")])
    # The generation prompt must NOT contain the history block on turn 1.
    assert gen.prompts, "generation LLM must have been called"
    generation_prompt = gen.prompts[-1]
    assert "## CONVERSATION HISTORY" not in generation_prompt
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_with_history_injects_history_and_rewrites():
    gen = _RecordingLLM()
    rewriter = _RecordingLLM()
    service = await _build_service(rewriter, gen)
    history = [
        _chat_message("user", "How does filter work on arrays?"),
        _chat_message("assistant", "Use pyspark.sql.functions.filter on ArrayType columns."),
    ]
    _collect_events([e async for e in service.chat_stream("What about its syntax?", conversation_history=history)])
    assert rewriter.prompts, "contextual rewriter must be invoked with history"
    rewrite_prompt = rewriter.prompts[0]
    assert "## CONVERSATION HISTORY" in rewrite_prompt
    assert "User: How does filter work on arrays?" in rewrite_prompt

    assert gen.prompts
    generation_prompt = gen.prompts[-1]
    assert "## CONVERSATION HISTORY" in generation_prompt
    assert "Assistant: Use pyspark.sql.functions.filter" in generation_prompt
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_survives_rewriter_failure():
    class _BoomRewriter:
        async def async_rewrite(self, query: str, conversation_history=None):
            raise RuntimeError("rewriter exploded")

    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    service = AsyncRagService(
        config=RagConfig(retrieval_top_k=5, confidence_threshold=0.05, max_context_chars=2000),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        query_rewriter=_BoomRewriter(),  # type: ignore[arg-type]
    )
    events = _collect_events([e async for e in service.chat_stream("What is Apache Airflow?")])
    types = [e["type"] for e in events]
    assert "done" in types, f"Stream must complete despite rewriter failure, got {types}"
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_turn1_reads_and_writes_cache():
    """Turn-1 (no history) may read/write the shared cache; a hit short-circuits."""

    class _HitCache:
        def __init__(self) -> None:
            self.reads = 0
            self.writes = 0

        async def aget(self, *args, **kwargs):
            self.reads += 1
            from data_engineering_copilot.domain.models import CachedAnswer, DocumentChunk

            return CachedAnswer(
                text="Cached answer text",
                sources=(
                    DocumentChunk(
                        chunk_id="cached:1",
                        source_name="Cached Docs",
                        title="T",
                        url="https://cached",
                        text="context",
                    ),
                ),
                confidence=0.9,
            )

        async def aset_exact(self, *args, **kwargs) -> None:
            self.writes += 1

        async def aset_semantic(self, *args, **kwargs) -> None:
            self.writes += 1

    cache = _HitCache()
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    service = AsyncRagService(
        config=RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.05,
            max_context_chars=2000,
            cache_enabled=True,
            chat_suggestions_mode="rule",
        ),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        cache=cache,  # type: ignore[arg-type]
        query_rewriter=QueryRewriter(llm_client=StubLLM(), enabled=True, hyde_enabled=False),
    )
    events = _collect_events([e async for e in service.chat_stream("What is Delta Lake?")])
    assert cache.reads >= 1, "turn-1 must read the shared cache"
    done = next(e for e in events if e["type"] == "done")
    assert done["text"] == "Cached answer text"
    assert cache.writes == 0, "cache hit must not write"
    # Cache hits must also surface follow-up suggestions (consistent UI).
    sugg_events = [e for e in events if e["type"] == "suggestions"]
    assert sugg_events, "turn-1 cache hit must emit a suggestions event"
    assert sugg_events[0]["suggestions"], "suggestions must be non-empty"
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_cache_hit_returns_cached_suggestions_without_regeneration():
    """A cache hit with cached suggestions returns them directly — no LLM call."""

    class _HitCacheWithSuggestions:
        async def aget(self, *args, **kwargs):
            from data_engineering_copilot.domain.models import CachedAnswer, DocumentChunk

            return CachedAnswer(
                text="Cached answer",
                sources=(
                    DocumentChunk(
                        chunk_id="cached:1",
                        source_name="Cached Docs",
                        title="Spark SQL",
                        url="https://cached",
                        text="Spark SQL is a module.",
                    ),
                ),
                confidence=0.9,
                suggestions=("Cached follow-up one?", "Cached follow-up two?"),
            )

    cache = _HitCacheWithSuggestions()
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    class _CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            return "Should not be called"

        def generate_stream(self, prompt: str):
            raise NotImplementedError

        @property
        def last_usage(self):
            return None

    counting = _CountingLLM()
    service = AsyncRagService(
        config=RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.05,
            max_context_chars=2000,
            cache_enabled=True,
            chat_suggestions_mode="hybrid",
        ),
        vector_store=store,
        llm_client=counting,  # type: ignore[arg-type]
        embedder=embedder,
        cache=cache,  # type: ignore[arg-type]
        query_rewriter=QueryRewriter(llm_client=StubLLM(), enabled=True, hyde_enabled=False),
    )
    events = _collect_events([e async for e in service.chat_stream("What is Delta Lake?")])
    sugg_events = [e for e in events if e["type"] == "suggestions"]
    assert sugg_events, "cache hit with cached suggestions must emit them"
    assert sugg_events[0]["suggestions"] == ["Cached follow-up one?", "Cached follow-up two?"]
    assert counting.calls == 0, "cached suggestions must be returned without an LLM call"
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_turn1_miss_writes_cache():
    """Turn-1 cache miss runs the pipeline and writes the answer back."""

    class _MissCache:
        def __init__(self) -> None:
            self.reads = 0
            self.writes = 0
            self.written: list = []

        async def aget(self, *args, **kwargs) -> None:
            self.reads += 1
            return None

        async def aset_exact(self, *args, **kwargs) -> None:
            self.writes += 1
            self.written.append(args[1] if len(args) > 1 else kwargs.get("answer"))

        async def aset_semantic(self, *args, **kwargs) -> None:
            self.writes += 1

    cache = _MissCache()
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    service = AsyncRagService(
        config=RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.05,
            max_context_chars=2000,
            cache_enabled=True,
            chat_suggestions_mode="rule",
        ),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        cache=cache,  # type: ignore[arg-type]
        query_rewriter=QueryRewriter(llm_client=StubLLM(), enabled=True, hyde_enabled=False),
    )
    events = _collect_events([e async for e in service.chat_stream("What is Delta Lake?")])
    assert any(e["type"] == "done" for e in events)
    assert cache.reads >= 1
    assert cache.writes >= 1, "turn-1 miss must write the answer back"
    # The generated suggestions must be cached alongside the answer.
    assert any(getattr(w, "suggestions", ()) for w in cache.written), "miss must cache suggestions with the answer"
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_follow_up_never_touches_cache():
    """Follow-up turns (history present) must never read or write the cache."""

    class _RecordingCache:
        def __init__(self) -> None:
            self.reads = 0
            self.writes = 0

        async def aget(self, *args, **kwargs) -> None:
            self.reads += 1
            return None

        async def aset_exact(self, *args, **kwargs) -> None:
            self.writes += 1

        async def aset_semantic(self, *args, **kwargs) -> None:
            self.writes += 1

    cache = _RecordingCache()
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    from data_engineering_copilot.domain.models import ChatMessage
    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    service = AsyncRagService(
        config=RagConfig(retrieval_top_k=5, confidence_threshold=0.05, max_context_chars=2000, cache_enabled=True),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        cache=cache,  # type: ignore[arg-type]
        query_rewriter=QueryRewriter(llm_client=StubLLM(), enabled=True, hyde_enabled=False),
    )
    history = [ChatMessage(message_id="m", session_id="s", role="user", content="What is Delta Lake?")]
    events = _collect_events(
        [e async for e in service.chat_stream("Does it support time travel?", conversation_history=history)]
    )
    assert any(e["type"] == "done" for e in events)
    assert cache.reads == 0, "follow-up must never read the shared cache"
    assert cache.writes == 0, "follow-up must never write the shared cache"
    await service.close()


class _RecallCache:
    """Fake cache exposing atop_k for the smart-cache recall tier."""

    def __init__(self, pairs, enable_top_k: bool = True) -> None:
        self._pairs = pairs
        self.enable_top_k = enable_top_k
        self.atop_k_calls = 0

    async def atop_k(self, query_embedding, scope=None, k=3, min_similarity=0.70):
        self.atop_k_calls += 1
        if not self.enable_top_k:
            return []
        return list(self._pairs)


def _cached_answer(text: str, url: str = "https://cached/1") -> Any:
    from data_engineering_copilot.domain.models import CachedAnswer

    return CachedAnswer(
        text=text,
        sources=(DocumentChunk(chunk_id="c1", source_name="Cached", title="T", url=url, text=text),),
        confidence=0.8,
        cached_at=time.time(),
    )


async def _build_recall_service(cache, *, scope_verifier=None, recall_enabled=True, answer_llm=None):
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    return AsyncRagService(
        config=RagConfig(
            retrieval_top_k=5,
            confidence_threshold=0.05,
            max_context_chars=2000,
            cache_enabled=True,
            chat_cache_recall_enabled=recall_enabled,
            chat_cache_top_k=3,
            chat_cache_recall_threshold=0.70,
            chat_cache_max_age_seconds=86400,
        ),
        vector_store=store,
        llm_client=answer_llm or StubLLM(),
        embedder=embedder,
        cache=cache,  # type: ignore[arg-type]
        scope_verifier=scope_verifier,  # type: ignore[arg-type]
        query_rewriter=QueryRewriter(llm_client=StubLLM(), enabled=True, hyde_enabled=False),
    )


class _CoveringScope:
    async def verify(self, question: str, context: str) -> bool:
        return True


class _RejectingScope:
    async def verify(self, question: str, context: str) -> bool:
        return False


def _history():
    return [ChatMessage(message_id="m", session_id="s", role="user", content="What is Delta Lake?")]


@pytest.mark.asyncio
async def test_smart_cache_verified_pass_returns_cached_grounding():
    cache = _RecallCache([(0.9, _cached_answer("Delta Lake is a storage layer."))])
    from tests.doubles.llm import StaticLLM

    smart_llm = StaticLLM(answer="SMART_CACHE_ANSWER: Delta Lake is a storage layer.")
    service = await _build_recall_service(cache, scope_verifier=_CoveringScope(), answer_llm=smart_llm)
    events = _collect_events(
        [e async for e in service.chat_stream("Tell me about Delta", conversation_history=_history())]
    )
    assert cache.atop_k_calls == 1
    assert any(e["type"] == "sources" for e in events)
    done = next(e for e in events if e["type"] == "done")
    assert done["text"] == "SMART_CACHE_ANSWER: Delta Lake is a storage layer."
    await service.close()


@pytest.mark.asyncio
async def test_smart_cache_scope_reject_falls_through_to_pipeline():
    cache = _RecallCache([(0.9, _cached_answer("Delta Lake is a storage layer."))])
    service = await _build_recall_service(cache, scope_verifier=_RejectingScope())
    events = _collect_events(
        [e async for e in service.chat_stream("What is Delta Lake?", conversation_history=_history())]
    )
    # Falls through: pipeline ran, done contains a real generated answer, not cached text.
    done = next(e for e in events if e["type"] == "done")
    assert "storage layer" not in done["text"]
    await service.close()


@pytest.mark.asyncio
async def test_smart_cache_disabled_never_calls_top_k():
    cache = _RecallCache([(0.9, _cached_answer("Delta Lake is a storage layer."))])
    service = await _build_recall_service(cache, scope_verifier=_CoveringScope(), recall_enabled=False)
    _collect_events([e async for e in service.chat_stream("What is Delta Lake?", conversation_history=_history())])
    assert cache.atop_k_calls == 0
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_done_emits_clean_answer_not_raw_json():
    """P0-B: the done event must carry the parsed answer, not the raw JSON envelope."""
    from tests.doubles.llm import StaticLLM

    raw_json = '{"status": "SUCCESS", "answer": "Spark SQL enables structured data processing.", "missing_info": null}'
    service = await _build_blocklist_service(
        [
            DocumentChunk(
                chunk_id="c1",
                source_name="Apache Spark 4.0.0",
                title="Spark",
                url="https://spark.apache.org/docs/latest/sql.html",
                text="Spark SQL is a Spark module.",
            )
        ]
    )
    # Replace the LLM with one returning the JSON envelope.
    service.llm_client = StaticLLM(answer=raw_json)  # type: ignore[assignment]
    events = _collect_events([e async for e in service.chat_stream("What is Spark SQL?")])
    done = next(e for e in events if e["type"] == "done")
    assert done["text"] == "Spark SQL enables structured data processing."
    assert '{"status"' not in done["text"]
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_answers_identity_without_retrieval():
    """P0-A: identity questions are answered by the fixed persona, not docs."""
    from data_engineering_copilot.services.async_rag import _IDENTITY_ANSWER

    service = await _build_blocklist_service(
        [
            DocumentChunk(
                chunk_id="c1",
                source_name="Claude Platform Docs",
                title="System prompts",
                url="https://platform.claude.com/docs/en/release-notes/system-prompts.md",
                text="The assistant is Claude, created by Anthropic.",
            )
        ]
    )
    events = _collect_events([e async for e in service.chat_stream("who are you?")])
    done = next(e for e in events if e["type"] == "done")
    assert done["text"] == _IDENTITY_ANSWER
    # No retrieval happened (no sources event).
    assert all(e["type"] != "sources" for e in events)
    await service.close()


def test_clean_chat_text_strips_refusal_json():
    """P0-B: refusal text with a leading JSON object should render as prose only."""
    from data_engineering_copilot.services.async_rag import _clean_chat_text

    raw = (
        '{"status": "INSUFFICIENT_CONTEXT", "answer": "", "missing_info": "x"}\n\n'
        "I cannot answer this question because the provided documentation does not cover its topic."
    )
    cleaned = _clean_chat_text(raw)
    assert '{"status"' not in cleaned
    assert "I cannot answer this question" in cleaned


def test_clean_chat_text_strips_sources_line():
    """The LLM-appended trailing 'Sources: [...]' line must not reach the user."""
    from data_engineering_copilot.services.async_rag import _clean_chat_text

    raw = "Spark SQL supports structured queries.\n\nSources: [Apache Spark 4.0.0 [Related Statements]]"
    cleaned = _clean_chat_text(raw)
    assert "Sources:" not in cleaned
    assert "Spark SQL supports structured queries." in cleaned


def test_clean_chat_text_keeps_sources_line_when_mid_answer():
    """A 'Sources:' mention inside the answer (not trailing) is preserved."""
    from data_engineering_copilot.services.async_rag import _clean_chat_text

    raw = "See the Sources section. Spark SQL is used.\n\nSources: [some doc]"
    cleaned = _clean_chat_text(raw)
    # Only a TRAILING 'Sources: [...]' line is stripped.
    assert "See the Sources section." in cleaned


@pytest.mark.asyncio
async def test_smart_cache_no_pairs_falls_through():
    cache = _RecallCache([], enable_top_k=False)
    service = await _build_recall_service(cache, scope_verifier=_CoveringScope())
    events = _collect_events(
        [e async for e in service.chat_stream("What is Delta Lake?", conversation_history=_history())]
    )
    done = next(e for e in events if e["type"] == "done")
    assert done["text"], "must fall through to the full pipeline when no pairs"
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_refuses_on_domain_mismatch():
    """P3: a data-engineering query must not be answered from foreign (Claude) docs."""
    claude = DocumentChunk(
        chunk_id="claude_x",
        source_name="Claude Platform Docs",
        title="Prompt caching",
        url="https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md",
        text="Claude prompt caching reduces cost and latency for the MCP connector and Anthropic API.",
    )
    service = await _build_blocklist_service([claude])
    events = _collect_events([e async for e in service.chat_stream("How do I fix Spark OOM errors?")])
    done = next(e for e in events if e["type"] == "done")
    assert "cannot answer" in done["text"].lower() or "does not cover" in done["text"].lower()
    assert "prompt caching" not in done["text"].lower()
    await service.close()


async def _build_blocklist_service(store_chunks, blocked=None, domain=None, config_overrides=None):
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    vectors = await embedder.embed_texts([c.text for c in store_chunks])
    await store.upsert_chunks(store_chunks, vectors)
    from data_engineering_copilot.services.query_rewriting import QueryRewriter

    cfg = RagConfig(
        retrieval_top_k=10,
        confidence_threshold=0.05,
        max_context_chars=2000,
        cache_enabled=False,
    )
    if config_overrides:
        cfg = RagConfig(**{**cfg.__dict__, **config_overrides})

    return AsyncRagService(
        config=cfg,
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        query_rewriter=QueryRewriter(llm_client=StubLLM(), enabled=True, hyde_enabled=False),
    )


@pytest.mark.asyncio
async def test_chat_stream_filters_blocked_url_chunks():
    """P0-A: chunks whose URL matches a blocked substring must never reach context."""
    good = DocumentChunk(
        chunk_id="good",
        source_name="Apache Spark 4.0.0",
        title="Spark",
        url="https://spark.apache.org/docs/latest/sql.html",
        text="Spark SQL is a Spark module for structured data processing.",
    )
    bad = DocumentChunk(
        chunk_id="bad",
        source_name="Claude Platform Docs",
        title="System prompts",
        url="https://platform.claude.com/docs/en/release-notes/system-prompts.md",
        text="The assistant is Claude, created by Anthropic.",
    )
    service = await _build_blocklist_service([good, bad])
    events = _collect_events(
        [e async for e in service.chat_stream("What is Spark SQL?", chat_blocked_url_substrings=["system-prompts.md"])]
    )
    sources = next(e for e in events if e["type"] == "sources")
    urls = [s["url"] for s in sources["sources"]]
    assert all("system-prompts.md" not in u for u in urls), f"blocked chunk leaked: {urls}"
    done = next(e for e in events if e["type"] == "done")
    assert done["text"], "pipeline should still complete with allowed chunks"
    await service.close()


@pytest.mark.asyncio
async def test_chat_stream_domain_source_filter():
    """P1: only whitelisted sources are retrieved."""
    spark = DocumentChunk(
        chunk_id="spark1",
        source_name="Apache Spark 4.0.0",
        title="Spark",
        url="https://spark.apache.org/docs/latest/sql.html",
        text="Spark SQL is a Spark module for structured data processing.",
    )
    claude = DocumentChunk(
        chunk_id="claude1",
        source_name="Claude Platform Docs",
        title="Claude",
        url="https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md",
        text="Claude prompt caching reduces cost and latency.",
    )
    service = await _build_blocklist_service([spark, claude])
    events = _collect_events(
        [e async for e in service.chat_stream("what is data processing", chat_domain_sources=["Apache Spark 4.0.0"])]
    )
    sources = next(e for e in events if e["type"] == "sources")
    names = {s["source_name"] for s in sources["sources"]}
    assert names == {"Apache Spark 4.0.0"}, f"non-whitelisted source leaked: {names}"
    await service.close()


@pytest.mark.asyncio
async def test_generate_suggestions_llm_mode():
    """LLM mode returns parsed suggestions grounded in the retrieved context."""
    from data_engineering_copilot.domain.models import RetrievedChunk

    service = await _build_blocklist_service(
        [
            DocumentChunk(
                chunk_id="c1",
                source_name="Apache Spark 4.0.0",
                title="Spark SQL",
                url="https://spark.apache.org/docs/latest/sql.html",
                text="Spark SQL is a Spark module for structured data processing.",
            )
        ],
        config_overrides={"chat_suggestions_mode": "llm", "chat_suggestions_count": 3},
    )

    class _RecordingLLM:
        def __init__(self) -> None:
            self.prompt = ""

        async def generate(self, prompt: str, **kwargs: object) -> str:
            self.prompt = prompt
            return "What is Spark SQL?\nHow do joins work?\nWhat is a DataFrame?"

    recording = _RecordingLLM()
    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="a",
                source_name="Apache Spark 4.0.0",
                title="Spark SQL",
                url="u",
                text="Spark SQL is a Spark module for structured data processing.",
            ),
            distance=0.1,
            confidence=0.9,
        )
    ]
    suggestions = await service._generate_suggestions(
        "What is Spark?",
        "Spark SQL is a module.",
        chunks,
        recording,  # type: ignore[arg-type]
        intent="how_to",
        history=[_chat_message("user", "What is Spark?"), _chat_message("assistant", "Spark SQL is a module.")],
    )
    assert len(suggestions) == 3
    assert suggestions[0] == "What is Spark SQL?"
    # The prompt must include retrieved context (repo-answerable) AND the intent
    # and conversation trajectory so suggestions follow logical next steps.
    assert "## RETRIEVED CONTEXT" in recording.prompt
    assert "Spark SQL is a Spark module" in recording.prompt
    assert "## INTENT" in recording.prompt and "how_to" in recording.prompt
    assert "## CONVERSATION HISTORY" in recording.prompt
    assert "What is Spark?" in recording.prompt
    await service.close()


@pytest.mark.asyncio
async def test_generate_suggestions_hybrid_falls_back_to_rule_on_llm_failure():
    """Hybrid mode: LLM failure falls back to deterministic rule suggestions."""
    service = await _build_blocklist_service(
        [
            DocumentChunk(
                chunk_id="c1",
                source_name="Apache Spark 4.0.0",
                title="Spark SQL",
                url="https://spark.apache.org/docs/latest/sql.html",
                text="Spark SQL is a module.",
            ),
            DocumentChunk(
                chunk_id="c2",
                source_name="Apache Spark 4.0.0",
                title="DataFrames",
                url="https://spark.apache.org/docs/latest/sql.html",
                text="DataFrames are structured.",
            ),
        ],
        config_overrides={"chat_suggestions_mode": "hybrid"},
    )

    class _BoomLLM:
        async def generate(self, prompt: str, **kwargs: object) -> str:
            raise RuntimeError("boom")

    from data_engineering_copilot.domain.models import RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(chunk_id="a", source_name="s", title="Spark SQL", url="u", text="t"),
            distance=0.1,
            confidence=0.9,
        ),
        RetrievedChunk(
            chunk=DocumentChunk(chunk_id="b", source_name="s", title="DataFrames", url="u2", text="t2"),
            distance=0.2,
            confidence=0.8,
        ),
    ]
    suggestions = await service._generate_suggestions("What is Spark?", "ans", chunks, _BoomLLM())  # type: ignore[arg-type]
    assert suggestions, "rule fallback must produce suggestions"
    assert "Spark SQL" in suggestions[0]
    await service.close()


def test_rule_suggestions_from_titles():
    from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
    from data_engineering_copilot.services.async_rag import AsyncRagService

    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(chunk_id="a", source_name="s", title="Spark SQL", url="u", text="t"),
            distance=0.1,
            confidence=0.9,
        ),
        RetrievedChunk(
            chunk=DocumentChunk(chunk_id="b", source_name="s", title="DataFrames", url="u2", text="t2"),
            distance=0.2,
            confidence=0.8,
        ),
    ]
    suggestions = AsyncRagService._rule_suggestions("What is Spark?", chunks)
    assert suggestions[0] == "What does the documentation say about Spark SQL?"
    assert len(suggestions) == 3


def test_rule_suggestions_intent_aware():
    """Rule fallback must shape suggestions by intent (e.g. code intent → examples)."""
    from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
    from data_engineering_copilot.services.async_rag import AsyncRagService

    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(chunk_id="a", source_name="s", title="Spark SQL", url="u", text="t"),
            distance=0.1,
            confidence=0.9,
        ),
        RetrievedChunk(
            chunk=DocumentChunk(chunk_id="b", source_name="s", title="DataFrames", url="u2", text="t2"),
            distance=0.2,
            confidence=0.8,
        ),
    ]
    code_suggestions = AsyncRagService._rule_suggestions("How do I join?", chunks, intent="code_example")
    assert "code example" in code_suggestions[0].lower()
    comparative = AsyncRagService._rule_suggestions("Compare A and B", chunks, intent="comparative")
    assert "differences between" in comparative[0].lower()


@pytest.mark.asyncio
async def test_chat_stream_emits_suggestions_event():
    """The SSE stream includes a suggestions event after done."""

    service = await _build_blocklist_service(
        [
            DocumentChunk(
                chunk_id="c1",
                source_name="Apache Spark 4.0.0",
                title="Spark SQL",
                url="https://spark.apache.org/docs/latest/sql.html",
                text="Spark SQL is a Spark module for structured data processing.",
            )
        ],
        config_overrides={"chat_suggestions_mode": "rule"},
    )
    events = _collect_events([e async for e in service.chat_stream("What is Spark SQL?")])
    types = [e["type"] for e in events]
    assert types[-1] == "suggestions"
    suggestions = events[-1]["suggestions"]
    assert isinstance(suggestions, list) and suggestions
    await service.close()
