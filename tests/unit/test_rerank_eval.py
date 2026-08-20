"""Tests for isolated reranker evaluation harness."""

from __future__ import annotations

import json
import pathlib

import pytest

from data_engineering_copilot.evaluation.rerank_eval import (
    _urls_to_relevance,
    load_candidate_pool,
    load_rerank_eval_dataset,
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
        from dataclasses import asdict

        from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk

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
