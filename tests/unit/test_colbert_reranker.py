"""Tests for ColBERT late-interaction reranker."""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.colbert_reranker import (
    ColBERTReranker,
    _char_ngram_overlap,
    _tokenize,
)

pytestmark = pytest.mark.unit


def _make_chunk(text: str, url: str = "https://example.com") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="test",
            source_name="test",
            title="Test",
            url=url,
            text=text,
        ),
        distance=0.0,
        confidence=0.0,
    )


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation(self):
        assert _tokenize("foo, bar!") == ["foo", "bar"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert _tokenize("v2 model") == ["v2", "model"]


class TestCharNgramOverlap:
    def test_identical_tokens(self):
        assert _char_ngram_overlap(["hello"], ["hello"]) == 1.0

    def test_no_overlap(self):
        assert _char_ngram_overlap(["abc"], ["xyz"]) == 0.0

    def test_partial_overlap(self):
        score = _char_ngram_overlap(["hello"], ["helo"])
        assert 0.0 < score < 1.0

    def test_empty_query(self):
        assert _char_ngram_overlap([], ["hello"]) == 0.0

    def test_empty_doc(self):
        assert _char_ngram_overlap(["hello"], []) == 0.0

    def test_multiple_query_tokens(self):
        score = _char_ngram_overlap(["hello", "world"], ["hello", "world"])
        assert score == 1.0


class TestMaxSimLightweight:
    def test_best_doc_match(self):
        score = _char_ngram_overlap(["spark", "sql"], ["spark", "sql", "query"])
        assert score > 0.8

    def test_worse_doc_match(self):
        good = _char_ngram_overlap(["spark", "sql"], ["spark", "sql"])
        bad = _char_ngram_overlap(["spark", "sql"], ["unrelated", "text"])
        assert good > bad


class TestColBERTReranker:
    @pytest.mark.asyncio
    async def test_rerank_basic(self):
        chunks = [
            _make_chunk("unrelated content"),
            _make_chunk("Spark SQL query optimization"),
            _make_chunk("another unrelated topic"),
        ]
        reranker = ColBERTReranker()
        result = await reranker.rerank("Spark SQL", chunks, top_k=2)
        assert len(result) == 2
        assert result[0].chunk.text == "Spark SQL query optimization"

    @pytest.mark.asyncio
    async def test_empty(self):
        reranker = ColBERTReranker()
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_fewer_than_top_k(self):
        chunks = [_make_chunk("hello")]
        reranker = ColBERTReranker()
        result = await reranker.rerank("hello", chunks, top_k=5)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_scores_normalized(self):
        chunks = [_make_chunk("a"), _make_chunk("b")]
        reranker = ColBERTReranker()
        result = await reranker.rerank("a", chunks, top_k=2)
        scores = [r.confidence for r in result]
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_is_available(self):
        reranker = ColBERTReranker()
        assert reranker.is_available()
