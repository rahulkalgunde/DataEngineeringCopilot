"""Tests for isolated reranker evaluation harness."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.evaluation.rerank_eval import (
    RerankEvalRow,
    RerankEvalServiceAdapter,
    RerankEvalServiceProtocol,
    _urls_to_relevance,
    load_candidate_pool,
    load_rerank_eval_dataset,
    run_rerank_eval,
    save_candidate_pool,
)

pytestmark = pytest.mark.unit


class TestLoadDataset:
    def test_load_sample(self, tmp_path: pathlib.Path):
        data = tmp_path / "test.jsonl"
        data.write_text(json.dumps({"query": "test", "source_urls": ["http://a"], "relevance_labels": [1]}) + "\n")
        rows = load_rerank_eval_dataset(data)
        assert len(rows) == 1
        assert rows[0].query == "test"

    def test_empty_file(self, tmp_path: pathlib.Path):
        data = tmp_path / "empty.jsonl"
        data.write_text("")
        rows = load_rerank_eval_dataset(data)
        assert rows == []


class TestUrlsToRelevance:
    def test_basic_mapping(self):
        result = _urls_to_relevance(["http://a", "http://b"], [1, 0], ["http://b", "http://a"])
        assert result == [0, 1]

    def test_unknown_url(self):
        result = _urls_to_relevance(["http://a"], [1], ["http://unknown"])
        assert result == [0]


class TestCandidatePool:
    def test_save_load_roundtrip(self, tmp_path: pathlib.Path):
        pool = {"query1": ["chunk1", "chunk2"]}
        path = tmp_path / "pool.json"
        save_candidate_pool(path, pool)
        loaded = load_candidate_pool(path)
        assert loaded == pool

    def test_save_load_roundtrip_with_document_chunks(self, tmp_path: pathlib.Path):
        chunk = DocumentChunk(
            chunk_id="chunk-123",
            source_name="test_source",
            title="Test Title",
            url="https://example.com/test",
            text="Test text content",
            content_hash="abc123",
            section_header="## Test Section",
            chunk_type="text",
            word_count=10,
            heading_path=("Section 1", "Subsection 2"),
        )
        retrieved = RetrievedChunk(chunk=chunk, distance=0.85, confidence=0.92)

        serialized = [
            {"chunk": asdict(retrieved.chunk), "distance": retrieved.distance, "confidence": retrieved.confidence}
        ]
        pool = {"test query": serialized}
        path = tmp_path / "pool.json"
        save_candidate_pool(path, pool)

        loaded = load_candidate_pool(path)
        assert "test query" in loaded

        loaded_data = loaded["test query"][0]
        assert loaded_data["distance"] == 0.85
        assert loaded_data["confidence"] == 0.92
        assert loaded_data["chunk"]["chunk_id"] == "chunk-123"
        assert loaded_data["chunk"]["heading_path"] == ["Section 1", "Subsection 2"]

        roundtripped = DocumentChunk(**loaded_data["chunk"])
        assert tuple(roundtripped.heading_path) == ("Section 1", "Subsection 2")


def _chunk(url: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            source_name="spark",
            title="T",
            url=url,
            text="some text",
            content_hash="h",
            heading_path=("A",),
        ),
        distance=0.1,
        confidence=0.9,
    )


class _FakeRerankService:
    """Doubles RerankEvalServiceProtocol: retrieve returns canned order, rerank reverses."""

    def __init__(self, retrieved: list[RetrievedChunk]):
        self.retrieved = retrieved
        self.retrieve_calls: list[str] = []
        self.rerank_calls = 0

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.retrieve_calls.append(query)
        return self.retrieved

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.rerank_calls += 1
        return list(reversed(chunks))[:top_k]

    def as_protocol(self) -> RerankEvalServiceProtocol:
        return self  # type: ignore[return-value]


class TestRunRerankEvalFreshPools:
    async def test_computes_metrics_and_saves_pool(self, tmp_path: pathlib.Path):
        service = _FakeRerankService([_chunk("http://a", "a"), _chunk("http://b", "b"), _chunk("http://c", "c")])
        pool_path = tmp_path / "pool.json"
        rows = [RerankEvalRow(query="what is spark?", source_urls=["http://a"], relevance_labels=[1])]

        report = await run_rerank_eval(rows, service.as_protocol(), k=2, candidate_pool_path=pool_path)

        # rerank reverses [a,b,c] -> [c,b,a]; top-2 post = [c(0), b(0)], pre = [a(1), b(0)]
        result = report.results[0]
        assert result.pre_rerank_relevance == [1, 0]
        assert result.post_rerank_relevance == [0, 0]
        assert result.metrics["mrr_gain"] == pytest.approx(-1.0)
        assert report.aggregate["ndcg_gain"] == pytest.approx(result.metrics["ndcg_gain"])

        saved = load_candidate_pool(pool_path)
        assert set(saved.keys()) == {"what is spark?"}
        assert saved["what is spark?"][0]["chunk"]["chunk_id"] == "a"

    async def test_no_pool_path_skips_persistence(self, tmp_path: pathlib.Path):
        service = _FakeRerankService([_chunk("http://a", "a")])
        rows = [RerankEvalRow(query="q", source_urls=[], relevance_labels=[])]
        report = await run_rerank_eval(rows, service.as_protocol(), k=1, candidate_pool_path=None)
        assert len(report.results) == 1
        assert not (tmp_path / "pool.json").exists()


class TestRunRerankEvalFrozenPools:
    def _write_pool(self, path: pathlib.Path) -> None:
        entries = [
            {"chunk": asdict(_chunk("http://x", "x").chunk), "distance": 0.2, "confidence": 0.8},
            {"chunk": asdict(_chunk("http://y", "y").chunk), "distance": 0.3, "confidence": 0.7},
        ]
        save_candidate_pool(path, {"frozen-q": entries})

    async def test_frozen_pool_bypasses_retrieval(self, tmp_path: pathlib.Path):
        pool_path = tmp_path / "pool.json"
        self._write_pool(pool_path)
        service = _FakeRerankService([_chunk("http://z", "z")])
        rows = [RerankEvalRow(query="frozen-q", source_urls=["http://y"], relevance_labels=[1])]

        await run_rerank_eval(rows, service.as_protocol(), k=2, candidate_pool_path=pool_path)

        assert service.retrieve_calls == []  # frozen: retrieval never runs
        assert service.rerank_calls == 1
        # heading_path serialized as list must be restored to tuple for DocumentChunk(**data)
        result = await run_rerank_eval(rows, service.as_protocol(), k=2, candidate_pool_path=pool_path)
        assert result.results[0].pre_rerank_relevance.count(1) == 1

    async def test_mixed_frozen_and_fresh_queries_merge_pool(self, tmp_path: pathlib.Path):
        pool_path = tmp_path / "pool.json"
        self._write_pool(pool_path)
        service = _FakeRerankService([_chunk("http://z", "z")])
        rows = [
            RerankEvalRow(query="frozen-q", source_urls=[], relevance_labels=[]),
            RerankEvalRow(query="fresh-q", source_urls=[], relevance_labels=[]),
        ]
        await run_rerank_eval(rows, service.as_protocol(), k=1, candidate_pool_path=pool_path)

        assert sorted(service.retrieve_calls) == ["fresh-q"]
        merged = load_candidate_pool(pool_path)
        assert set(merged.keys()) == {"frozen-q", "fresh-q"}


class TestReportSummary:
    def test_summary_lists_aggregates(self):
        from data_engineering_copilot.evaluation.rerank_eval import RerankEvalReport, RerankEvalResult

        report = RerankEvalReport(
            results=[RerankEvalResult(query="q", pre_rerank_relevance=[0], post_rerank_relevance=[1], metrics={})],
            aggregate={"ndcg_gain": 0.5},
        )
        text = report.summary()
        assert "Queries evaluated: 1" in text
        assert "ndcg_gain: 0.5000" in text


class TestRerankEvalServiceAdapter:
    def _rag(self, reranker=None):
        return SimpleNamespace(
            embedder=SimpleNamespace(embed_query=AsyncMock(return_value=[0.1, 0.2])),
            vector_store=SimpleNamespace(query=AsyncMock(return_value=[_chunk("http://v")])),
            _ensure_reranker_ready=AsyncMock(),
            reranker=reranker,
        )

    async def test_retrieve_delegates_to_embedder_and_store(self):
        rag = self._rag()
        adapter = RerankEvalServiceAdapter(rag)  # type: ignore[arg-type]
        out = await adapter.retrieve("q", top_k=3)
        rag.embedder.embed_query.assert_awaited_once_with("q")
        rag.vector_store.query.assert_awaited_once_with([0.1, 0.2], top_k=3, query_text="q")
        assert out[0].chunk.url == "http://v"

    async def test_rerank_passthrough_when_no_reranker(self):
        rag = self._rag(reranker=None)
        adapter = RerankEvalServiceAdapter(rag)  # type: ignore[arg-type]
        chunks = [_chunk("http://a", "a"), _chunk("http://b", "b")]
        out = await adapter.rerank("q", chunks, top_k=1)
        assert [c.chunk.chunk_id for c in out] == ["a"]

    async def test_rerank_passthrough_when_unavailable(self):
        reranker = SimpleNamespace(is_available=lambda: False, rerank=AsyncMock())
        rag = self._rag(reranker=reranker)
        adapter = RerankEvalServiceAdapter(rag)  # type: ignore[arg-type]
        chunks = [_chunk("http://a", "a")]
        out = await adapter.rerank("q", chunks, top_k=1)
        reranker.rerank.assert_not_awaited()
        assert out == chunks

    async def test_rerank_delegates_when_available(self):
        reranked = [_chunk("http://best", "best")]
        reranker = SimpleNamespace(is_available=lambda: True, rerank=AsyncMock(return_value=reranked))
        rag = self._rag(reranker=reranker)
        adapter = RerankEvalServiceAdapter(rag)  # type: ignore[arg-type]
        out = await adapter.rerank("q", [_chunk("http://a")], top_k=1)
        reranker.rerank.assert_awaited_once()
        assert out is reranked
