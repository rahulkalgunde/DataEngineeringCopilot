"""Hermetic RAG pipeline-logic tests (no infra, no network).

Ports of the former real-Qdrant/Ollama pipeline tests, running the full
retrieval → context → answer flow against deterministic doubles
(``InMemoryVectorStore`` + ``StubEmbedder`` + ``StubLLM``) so the suite runs
offline and stable.  Real-infra coverage lives in the single Ollama smoke test
in ``tests/integration/test_rag_integration.py``.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import Answer, DocumentChunk, RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService, _rerank_pool_size
from tests.doubles.embedder import StubEmbedder
from tests.doubles.llm import StubLLM
from tests.doubles.vector_store import InMemoryVectorStore

_TOPICS = [
    (
        "Apache Spark",
        "Apache Spark is a unified analytics engine for large-scale data processing. "
        "It provides high-level APIs in Scala, Java, Python, and R.",
    ),
    (
        "Spark SQL",
        "Spark SQL is a Spark module for structured data processing. It provides a "
        "programming abstraction called DataFrames and SQL.",
    ),
    (
        "Spark Streaming",
        "Spark Streaming enables scalable, high-throughput, fault-tolerant stream processing of live data streams.",
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
    (
        "Airflow DAGs",
        "A DAG (Directed Acyclic Graph) in Airflow is a collection of tasks organized "
        "with dependencies and scheduling logic.",
    ),
    (
        "Databricks",
        "Databricks is a unified analytics platform built on top of Apache Spark that "
        "provides collaborative notebooks and data pipelines.",
    ),
    (
        "Data Lakehouse",
        "The data lakehouse architecture combines the best features of data lakes and "
        "data warehouses into a single platform.",
    ),
    (
        "Structured Streaming",
        "Structured Streaming is a scalable stream processing engine built on the Spark SQL engine.",
    ),
    (
        "PySpark",
        "PySpark is the Python API for Apache Spark. It allows you to write Spark applications using Python.",
    ),
]


def _build_chunks() -> list[DocumentChunk]:
    chunks = []
    for i, (title, text) in enumerate(_TOPICS):
        chunks.append(
            DocumentChunk(
                chunk_id=f"rag:doc{i:03d}:chunk00",
                source_name="RAG Test Docs",
                title=title,
                url=f"https://example.com/docs/{title.lower().replace(' ', '-')}.html",
                text=text,
            )
        )
    return chunks


@pytest.fixture
async def _rag():
    """Function-scoped hermetic RAG service (doubles only)."""
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)

    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    service = AsyncRagService(
        config=RagConfig(
            retrieval_top_k=10,
            confidence_threshold=0.10,
            max_context_chars=2000,
        ),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
    )
    yield service
    await embedder.close()
    await store.close()


@pytest.mark.asyncio
async def test_answer_returns_answer_object(_rag):
    answer = await _rag.answer("What is Apache Spark?")
    assert isinstance(answer, Answer)
    assert isinstance(answer.text, str)
    assert isinstance(answer.sources, tuple)
    assert isinstance(answer.confidence, float)


@pytest.mark.asyncio
async def test_answer_has_substantial_text(_rag):
    answer = await _rag.answer("What is Apache Spark?")
    assert len(answer.text) > 20, f"Answer too short: {answer.text!r}"


@pytest.mark.asyncio
async def test_answer_cites_sources(_rag):
    answer = await _rag.answer("What is Delta Lake?")
    assert len(answer.sources) > 0, "Should cite at least one source"


@pytest.mark.asyncio
async def test_answer_confidence_nonnegative(_rag):
    answer = await _rag.answer("What is Apache Airflow?")
    assert answer.confidence >= 0.0


@pytest.mark.asyncio
async def test_multiple_sequential_questions(_rag):
    for q in ["What is Apache Spark?", "What is Delta Lake?"]:
        answer = await _rag.answer(q)
        assert len(answer.text) > 10, f"Bad answer for: {q}"


@pytest.mark.asyncio
async def test_unrelated_question_acknowledges_gap(_rag):
    """An unrelated question should produce an answer that acknowledges the
    docs don't cover it (via the gap-acknowledging stub response)."""
    answer = await _rag.answer("What is the capital of France?")
    text_lower = answer.text.lower()
    acknowledges_gap = any(
        phrase in text_lower
        for phrase in [
            "cannot answer",
            "does not provide",
            "does not address",
            "not covered",
            "outside",
            "does not contain",
        ]
    )
    assert acknowledges_gap or answer.confidence < 0.5, (
        f"Expected the answer to acknowledge the gap. confidence={answer.confidence:.4f}, text={answer.text[:200]!r}"
    )


@pytest.mark.asyncio
async def test_answer_captures_retrieval_provenance(_rag):
    """Opt-in provenance capture exposes per-variant/fused/final retrieval order."""
    prov: list[dict] = []
    answer = await _rag.answer("What is Delta Lake?", provenance=prov)
    assert len(prov) == 1
    record = prov[0]
    assert record["question"] == "What is Delta Lake?"
    assert record["cache_hit"] is False
    # Single retrieval variant (no rewriter in the fixture): the original query.
    assert record["query_variants"][0]["variant"] == "original"
    assert record["query_variants"][0]["retrieved"]
    assert record["fused"]
    assert record["candidate_pool_size"] == len(record["fused"])
    # Final context must match the returned sources by URL.
    assert len(record["final_context"]) == len(answer.sources)
    assert {c["url"] for c in record["final_context"]} == {c.url for c in answer.sources}
    assert record["rerank"]["enabled"] is False
    assert record["stage_times"].get("retrieval", 0) >= 0


@pytest.mark.asyncio
async def test_answer_provenance_rerank_disabled_noop(_rag):
    """Without rerank the pool equals the fused candidate set; truncation to
    ``reranker_top_k`` still applies (that is the final context size)."""
    prov: list[dict] = []
    await _rag.answer("What is Spark Streaming?", provenance=prov)
    record = prov[0]
    assert record["rerank"]["enabled"] is False
    assert record["rerank"]["pool_size"] == record["candidate_pool_size"]
    assert len(record["final_context"]) == record["rerank"]["final_top_k"]


@pytest.mark.asyncio
async def test_answer_provenance_labels_multi_query_variants(_rag):
    """Decomposed, expanded, and HyDE variants are labelled distinctly."""
    from data_engineering_copilot.services.query_rewriting import RewrittenQuery

    class _Rewriter:
        async def async_rewrite(self, question: str) -> RewrittenQuery:
            return RewrittenQuery(
                original_query=question,
                intent="multi_step",
                decomposed_steps=("delta lake acid transactions", "delta lake time travel"),
                hyde_query="Delta Lake supports ACID transactions on top of Apache Spark",
            )

        async def expand_queries(self, question: str, max_variations: int = 2) -> list[str]:
            return ["Delta Lake storage framework"]

    # Rebuild a service with the rewriter wired in.
    from data_engineering_copilot.services.async_rag import AsyncRagService
    from tests.doubles.embedder import StubEmbedder
    from tests.doubles.llm import StubLLM
    from tests.doubles.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)
    service = AsyncRagService(
        config=RagConfig(retrieval_top_k=10, confidence_threshold=0.10, max_context_chars=2000),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        query_rewriter=_Rewriter(),  # type: ignore[arg-type]
    )
    try:
        prov: list[dict] = []
        await service.answer("What is Delta Lake?", provenance=prov)
    finally:
        await embedder.close()
        await store.close()

    variants = {v["variant"] for v in prov[0]["query_variants"]}
    assert "original" in variants
    assert "decomposed" in variants
    assert "expanded" in variants
    assert "hyde" in variants


def test_rerank_pool_size_is_wider_than_retrieval_top_k() -> None:
    """The rerank pool must exceed the dense cutoff so near-miss URLs are rescued.

    Defaults: retrieval_top_k=30, reranker_top_k=20 → 240 (was 150). A document
    fused at rank 175 (observed for Q6) must stay inside the pool.
    """
    assert _rerank_pool_size(30, 20) == 240
    assert _rerank_pool_size(20, 20) == 160
    # Reranker multiplier still dominates for large top-k.
    assert _rerank_pool_size(5, 50) == 250
    assert _rerank_pool_size(30, 20) > 30


@pytest.mark.asyncio
async def test_provenance_reports_budget_dropped_segments() -> None:
    """A tight context budget drops lower-ranked segments with a clear reason.

    The ``dropped`` provenance list names each excluded segment with reason
    ``dropped_due_total_context_budget`` while ``final_context`` reflects only
    the segments actually placed into the prompt.
    """
    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)

    big = [
        DocumentChunk(
            chunk_id=f"big:{i}",
            source_name="RAG Test Docs",
            title=f"Big {i}",
            url=f"https://example.com/big-{i}.html",
            text=f"Segment {i}: " + "x" * 120,
            segment_index=i,
            segment_total=2,
            parent_content_hash="parent-hash",
        )
        for i in range(4)
    ]
    vectors = await embedder.embed_texts([c.text for c in big])
    await store.upsert_chunks(big, vectors)

    # Budget small enough that only the first ~1-2 segments fit.
    service = AsyncRagService(
        config=RagConfig(retrieval_top_k=10, confidence_threshold=0.0, max_context_chars=200),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
    )
    try:
        prov: list[dict] = []
        await service.answer("What is x?", provenance=prov)
    finally:
        await embedder.close()
        await store.close()

    assert len(prov) == 1
    record = prov[0]
    assert "dropped" in record
    assert record["dropped"], "expected at least one budget-dropped segment"
    for dropped in record["dropped"]:
        assert dropped["reason"] == "dropped_due_total_context_budget"
        assert dropped["chunk_id"].startswith("big:")
        assert "segment_index" in dropped
        assert "parent_content_hash" in dropped
        assert "url" in dropped
        assert "rank" in dropped
    # The final context must not claim dropped segments as included.
    final_ids = {c["chunk_id"] for c in record["final_context"]}
    dropped_ids = {d["chunk_id"] for d in record["dropped"]}
    assert not (final_ids & dropped_ids)


@pytest.mark.asyncio
async def test_rag_service_passes_rerank_pool_as_fused_limit() -> None:
    """Task 11: the RAG service asks the store for the rerank pool (not just
    top_k), so Qdrant's fused pool and the reranker's candidate pool agree."""
    from data_engineering_copilot.services.async_rag import AsyncRagService, _rerank_pool_size
    from tests.doubles.embedder import StubEmbedder
    from tests.doubles.llm import StubLLM
    from tests.doubles.vector_store import InMemoryVectorStore

    class _RecordingStore(InMemoryVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.fused_limits: list[int] = []

        async def query(self, *args, **kwargs):
            if "fused_limit" in kwargs:
                self.fused_limits.append(kwargs["fused_limit"])
            return await super().query(*args, **kwargs)

    store = _RecordingStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    config = RagConfig(retrieval_top_k=5, confidence_threshold=0.0, reranker_top_k=20)
    service = AsyncRagService(
        config=config,
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
    )
    try:
        await service.answer("What is Apache Spark?")
    finally:
        await embedder.close()
        await store.close()

    expected = _rerank_pool_size(config.retrieval_top_k, config.reranker_top_k)
    assert store.fused_limits, "expected the RAG service to pass fused_limit"
    assert all(limit == expected for limit in store.fused_limits)
    assert expected >= max(config.retrieval_top_k * 8, config.reranker_top_k * 5)


# ------------------------------------------------------------------
# Task 12: cache-poisoning prevention + diagnostic cache bypass
# ------------------------------------------------------------------


class _RecordingCache:
    """QueryCache stand-in that records read/write attempts."""

    def __init__(self) -> None:
        self.get_calls: list[tuple[str, object]] = []
        self.set_calls: list[tuple[str, object]] = []
        self._store: dict[str, object] = {}
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        return 0.0

    async def aget(self, query, query_embedding=None, scope=None):
        self.get_calls.append((query, scope))
        return self._store.get(query)

    async def aset_exact(self, query, answer, scope=None):
        self.set_calls.append((query, scope))
        self._store[query] = answer

    async def aset_semantic(self, query, query_embedding, answer, scope=None):
        self.set_calls.append((query, scope))

    @staticmethod
    def is_cacheable(answer) -> bool:
        return bool(answer.sources and answer.confidence >= 0.5)


@pytest.mark.asyncio
async def test_rag_service_does_not_read_cache_when_bypassed() -> None:
    from data_engineering_copilot.services.async_rag import AsyncRagService
    from tests.doubles.embedder import StubEmbedder
    from tests.doubles.llm import StubLLM
    from tests.doubles.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    cache = _RecordingCache()
    service = AsyncRagService(
        config=RagConfig(retrieval_top_k=5, confidence_threshold=0.0, cache_enabled=True),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        cache=cache,  # type: ignore[arg-type]
    )
    try:
        # Pre-populate the cache; a bypassed call must not read it.
        cache._store["What is Apache Spark?"] = "stale"
        await service.answer("What is Apache Spark?", bypass_cache=True)
    finally:
        await embedder.close()
        await store.close()

    assert cache.get_calls == [], "bypass_cache=True must not read the cache"


@pytest.mark.asyncio
async def test_rag_service_does_not_write_cache_when_bypassed() -> None:
    from data_engineering_copilot.services.async_rag import AsyncRagService
    from tests.doubles.embedder import StubEmbedder
    from tests.doubles.llm import StubLLM
    from tests.doubles.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    cache = _RecordingCache()
    service = AsyncRagService(
        config=RagConfig(retrieval_top_k=5, confidence_threshold=0.0, cache_enabled=True),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        cache=cache,  # type: ignore[arg-type]
    )
    try:
        await service.answer("What is Apache Spark?", bypass_cache=True)
    finally:
        await embedder.close()
        await store.close()

    assert cache.set_calls == [], "bypass_cache=True must not write the cache"


@pytest.mark.asyncio
async def test_rag_service_cache_enabled_false_skips_read_and_write() -> None:
    from data_engineering_copilot.services.async_rag import AsyncRagService
    from tests.doubles.embedder import StubEmbedder
    from tests.doubles.llm import StubLLM
    from tests.doubles.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    await store.initialize()
    embedder = StubEmbedder(dimension=768)
    chunks = _build_chunks()
    vectors = await embedder.embed_texts([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)

    cache = _RecordingCache()
    service = AsyncRagService(
        config=RagConfig(retrieval_top_k=5, confidence_threshold=0.0, cache_enabled=False),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=embedder,
        cache=cache,  # type: ignore[arg-type]
    )
    try:
        await service.answer("What is Apache Spark?")
    finally:
        await embedder.close()
        await store.close()

    assert cache.get_calls == []
    assert cache.set_calls == []
