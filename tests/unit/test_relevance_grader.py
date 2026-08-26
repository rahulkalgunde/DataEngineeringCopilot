"""Tests for services/relevance_grader.py."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.relevance_grader import RelevanceGrader


def _make_chunk(text: str = "content") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="c1",
            source_name="src",
            title="T",
            url="http://x",
            text=text,
            file_path="f.md",
        ),
        distance=1.0,
        confidence=1.0,
    )


class TestRelevanceGrader:
    @pytest.mark.asyncio
    async def test_grades_relevance(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = '{"relevance_score": 0.85}'
        grader = RelevanceGrader(llm)
        score = await grader.grade_chunks("query", [_make_chunk()])
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_zero(self) -> None:
        llm = AsyncMock()
        grader = RelevanceGrader(llm)
        score = await grader.grade_chunks("query", [])
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_handles_markdown_fence(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = '```json\n{"relevance_score": 0.7}\n```'
        grader = RelevanceGrader(llm)
        score = await grader.grade_chunks("query", [_make_chunk()])
        assert score == 0.7

    @pytest.mark.asyncio
    async def test_failure_returns_one(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "not json"
        grader = RelevanceGrader(llm)
        score = await grader.grade_chunks("query", [_make_chunk()])
        assert score == 1.0
