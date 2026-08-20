from __future__ import annotations

from typing import Any

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RagConfig, RetrievedChunk
from data_engineering_copilot.services.async_rag import AsyncRagService
from data_engineering_copilot.services.relevance_grader import RelevanceGrader
from tests.doubles.llm import StubLLM


class _FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * 8

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    async def close(self) -> None:
        pass


class _SpyStore:
    def __init__(self) -> None:
        self.queries = 0

    async def query(self, *args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        self.queries += 1
        return []

    async def upsert_chunks(self, chunks: Any, vectors: Any) -> None:
        pass

    async def upsert_frozen_chunks(self, chunks: Any, vectors: Any) -> None:
        pass

    def fit_bm25_corpus(self, texts: list[str]) -> None:
        pass

    async def validate_index_generation(self, expected_points: int | None = None) -> dict[str, Any]:
        return {"passed": True}

    async def count(self) -> int:
        return 0

    async def count_urls(self, source_name: str) -> int:
        return 0

    async def scroll_chunks_by_parent_hash(self, parent_hash: str, source_name: str = "") -> list[DocumentChunk]:
        return []

    async def get_content_hash_for_url(self, url: str, source_name: str = "") -> str:
        return ""

    async def delete_by_url(self, url: str, source_name: str = "") -> None:
        pass

    def fit_bm25(self, texts: list[str]) -> None:
        pass

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _make_service(grader_answer: str, store: _SpyStore) -> AsyncRagService:
    return AsyncRagService(
        config=RagConfig(),
        vector_store=store,
        llm_client=StubLLM(),
        embedder=_FakeEmbedder(),
        relevance_grader=RelevanceGrader(llm_client=StubLLM(answer=grader_answer)),
    )


def _fake_chunk(text: str = "doc") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="c1",
            source_name="test",
            title="t",
            url="http://example.com",
            text=text,
        ),
        distance=0.1,
        confidence=0.5,
    )


@pytest.mark.asyncio
async def test_grade_chunks_valid_score() -> None:
    grader = RelevanceGrader(llm_client=StubLLM(answer='{"relevance_score": 0.85}'))
    score = await grader.grade_chunks("q", [_fake_chunk()])
    assert score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_grade_chunks_low_score() -> None:
    grader = RelevanceGrader(llm_client=StubLLM(answer='{"relevance_score": 0.2}'))
    score = await grader.grade_chunks("q", [_fake_chunk()])
    assert score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_grade_chunks_empty_returns_zero() -> None:
    grader = RelevanceGrader(llm_client=StubLLM(answer="ignored"))
    assert await grader.grade_chunks("q", []) == 0.0


@pytest.mark.asyncio
async def test_grade_chunks_parse_error_returns_one() -> None:
    grader = RelevanceGrader(llm_client=StubLLM(answer="not json"))
    score = await grader.grade_chunks("q", [_fake_chunk()])
    assert 0.0 <= score <= 1.0
    assert score == 1.0


@pytest.mark.asyncio
async def test_guard_skips_expansion_when_relevant() -> None:
    store = _SpyStore()
    svc = _make_service('{"relevance_score": 0.9}', store)
    chunks = [_fake_chunk()]
    out = await svc._relevance_guarded_chunks("q", chunks)
    assert out is chunks
    assert store.queries == 0


@pytest.mark.asyncio
async def test_guard_expands_when_low_relevance() -> None:
    store = _SpyStore()
    svc = _make_service('{"relevance_score": 0.1}', store)
    chunks = [_fake_chunk()]
    await svc._relevance_guarded_chunks("q", chunks)
    assert store.queries == 1
