"""Contract tests pinning tests/doubles to their production protocols.

Two layers:
1. Structural conformance — every protocol method/property must exist on the
   double and accept the protocol's parameter names, so the doubles can never
   silently drift from the real contracts (catches renames/removals).
2. Behavioral pinning — core observable behavior of each double, so pipeline
   tests that migrate to them keep stable, documented semantics.
"""

from __future__ import annotations

import inspect

import pytest

from data_engineering_copilot.domain.protocols import (
    ChunkerProtocol,
    EmbedderProtocol,
    LLMClientProtocol,
    VectorStoreProtocol,
)
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.semantic_chunker import SemanticChunker
from tests.doubles.embedder import StubEmbedder
from tests.doubles.frontier import InMemoryFrontierDB
from tests.doubles.llm import STUB_ANSWER, STUB_GAP_ANSWER, StaticLLM, StubLLM
from tests.doubles.redis import _StubRedis
from tests.doubles.vector_store import InMemoryVectorStore

_DOUBLE_PROTOCOL_PAIRS = [
    (StubLLM, LLMClientProtocol),
    (StaticLLM, LLMClientProtocol),
    (StubEmbedder, EmbedderProtocol),
    (InMemoryVectorStore, VectorStoreProtocol),
]

# Real production implementations pinned to their protocol so a refactor that
# renames/removes/re-shapes a protocol member is caught at test time.
_REAL_PROTOCOL_PAIRS = [
    (DocumentChunker, ChunkerProtocol),
    (SemanticChunker, ChunkerProtocol),
    (HeaderAwareChunker, ChunkerProtocol),
]


@pytest.mark.parametrize(
    "double_cls,protocol",
    _DOUBLE_PROTOCOL_PAIRS + _REAL_PROTOCOL_PAIRS,
)
def test_double_structurally_satisfies_protocol(double_cls, protocol):
    for name in dir(protocol):
        if name.startswith("_"):
            continue
        proto_attr = getattr(protocol, name)
        if not callable(proto_attr) and not isinstance(proto_attr, property):
            continue
        double_attr = getattr(double_cls, name, None)
        assert double_attr is not None, f"{double_cls.__name__} is missing protocol member {protocol.__name__}.{name}"
        if callable(proto_attr):
            proto_params = set(inspect.signature(proto_attr).parameters) - {"self"}
            double_params = set(inspect.signature(double_attr).parameters) - {"self"}
            missing = proto_params - double_params
            assert not missing, f"{double_cls.__name__}.{name} lacks protocol params {sorted(missing)}"


def test_extract_sentences_sentinel_contract():
    """Pin the sentence-extraction contract that was violated by commit d7e595d.

    ``extract_sentences`` returns ``None`` to mean "sentence pre-extraction not
    supported by this chunker" (callers must fall through to plain ``chunk()``),
    and a (possibly empty) list to mean "sentences were extracted".  A chunker
    that returns a sentinel the caller misreads as "no content" silently skips
    every page, so this contract is pinned explicitly.
    """
    sample_text = (
        "Apache Spark is a unified analytics engine for large-scale data processing. "
        "It provides high-level APIs in Scala, Java, Python, and R."
    )

    # DocumentChunker / HeaderAwareChunker: "not supported" -> None sentinel.
    for chunker in [
        DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100),
        HeaderAwareChunker(chunk_size_words=75, overlap_words=15),
    ]:
        assert hasattr(chunker, "extract_sentences")
        assert chunker.extract_sentences(sample_text) is None

    # SemanticChunker: real text yields an actual sentence list (never None).
    sentences = SemanticChunker.extract_sentences(sample_text)
    assert isinstance(sentences, list)
    assert len(sentences) >= 1


@pytest.mark.asyncio
async def test_stub_llm_returns_answer_and_gap_and_counts_calls():
    llm = StubLLM()
    answer = await llm.generate("What is Apache Spark?")
    assert answer == STUB_ANSWER
    assert llm.call_count == 1

    gap = await llm.generate("What is the capital of France?")
    assert gap == STUB_GAP_ANSWER
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_static_llm_counts_calls():
    llm = StaticLLM(answer="fixed reply")
    assert await llm.generate("q1") == "fixed reply"
    assert await llm.generate("q2") == "fixed reply"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_stub_embedder_is_deterministic_and_shapes_vectors():
    embedder = StubEmbedder(dimension=64)
    v1 = await embedder.embed_query("apache spark engine")
    v2 = await embedder.embed_query("apache spark engine")
    v3 = (await embedder.embed_texts(["delta lake"]))[0]
    assert v1 == v2, "embedding must be deterministic"
    assert len(v1) == 64
    assert len(v3) == 64
    assert v1 != v3
    await embedder.close()


@pytest.mark.asyncio
async def test_in_memory_vector_store_retrieval_and_filters():
    store = InMemoryVectorStore()
    await store.initialize()

    chunk_a = _chunk("a", "Apache Spark analytics engine")
    chunk_b = _chunk("b", "Delta Lake ACID transactions", chunk_type="table")
    embedder = StubEmbedder(dimension=64)
    vectors = await embedder.embed_texts([chunk_a.text, chunk_b.text])
    await store.upsert_chunks([chunk_a, chunk_b], vectors)

    assert await store.count() == 2
    assert await store.count_urls("RAG Test Docs") == 2

    results = await store.query(await embedder.embed_query("apache spark"), top_k=2)
    assert results[0].chunk.chunk_id == "a"
    assert 0.0 <= results[0].confidence <= 1.0
    assert results[0].distance == pytest.approx(1.0 - results[0].confidence)

    text_only = await store.query(
        await embedder.embed_query("apache spark"),
        top_k=2,
        source_filter=["RAG Test Docs"],
        chunk_type_filter="text",
    )
    assert [r.chunk.chunk_id for r in text_only] == ["a"]

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_frontier_state_machine_and_rediscovery():
    db = InMemoryFrontierDB()
    await db.initialize()

    url_hash = await db.discover("https://example.com/docs", "Docs", None, depth=0)
    assert url_hash is not None

    pending = await db.get_pending("Docs")
    assert len(pending) == 1
    assert pending[0].state == "DISCOVERED"

    claimed = await db.claim(url_hash)
    assert claimed is not None
    assert claimed.state == "FETCHING"
    assert claimed.attempts == 1

    await db.mark_failed(url_hash, "timeout")
    failed = await db.get_record(url_hash)
    assert failed is not None and failed.state == "FAILED"

    await db.mark_processed(url_hash)
    processed = await db.get_record(url_hash)
    assert processed is not None and processed.state == "PROCESSED"

    await db.close()


@pytest.mark.asyncio
async def test_stub_redis_hash_and_pipeline_semantics():
    redis = _StubRedis()
    await redis.hset("crawl:url_registry:src", "https://a", "hash1")
    assert await redis.hget("crawl:url_registry:src", "https://a") == b"hash1"

    pipe = redis.pipeline(transaction=False)
    pipe.hget("crawl:url_registry:src", "https://a")
    pipe.hget("crawl:url_registry:src", "missing")
    results = await pipe.execute()
    assert results == [b"hash1", None]

    assert await redis.delete("crawl:url_registry:src") == 1
    assert await redis.hget("crawl:url_registry:src", "https://a") is None


def _chunk(chunk_id: str, text: str, chunk_type: str = "text"):
    from data_engineering_copilot.domain.models import DocumentChunk

    return DocumentChunk(
        chunk_id=chunk_id,
        source_name="RAG Test Docs",
        title="topic",
        url=f"https://example.com/{chunk_id}.html",
        text=text,
        chunk_type=chunk_type,
    )
