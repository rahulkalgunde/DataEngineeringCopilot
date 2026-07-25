"""Tests for ContextualChunkEnricher (contextual_chunk_enricher.py:43)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.services.contextual_chunk_enricher import (
    ContextualChunkEnricher,
    LLMContextSummarizer,
)


def _chunk(cid="c1", text="test text", source="s", title="t", url="http://x") -> DocumentChunk:
    return DocumentChunk(chunk_id=cid, source_name=source, title=title, url=url, text=text)


def _doc(source="s", title="t", text="doc text") -> ParsedDocument:
    return ParsedDocument(source_name=source, title=title, url="http://x", text=text)


class TestLLMContextSummarizer:
    """LLMContextSummarizer (line 22)."""

    async def test_summarize_success(self):
        llm = AsyncMock()
        llm.generate.return_value = "This is a short summary."
        summarizer = LLMContextSummarizer(llm_client=llm, max_summary_words=50)
        result = await summarizer.summarize(_doc())
        assert result == "This is a short summary."

    async def test_summarize_truncates_long_result(self):
        llm = AsyncMock()
        llm.generate.return_value = "line1\nline2\nline3"
        summarizer = LLMContextSummarizer(llm_client=llm)
        result = await summarizer.summarize(_doc())
        assert result == "line1"
        assert "\n" not in result

    async def test_summarize_error_returns_empty(self):
        llm = AsyncMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        summarizer = LLMContextSummarizer(llm_client=llm)
        result = await summarizer.summarize(_doc())
        assert result == ""

    async def test_summarize_prompt_includes_title_and_text(self):
        llm = AsyncMock()
        llm.generate.return_value = "summary"
        summarizer = LLMContextSummarizer(llm_client=llm)
        await summarizer.summarize(_doc(title="Test Title", text="Content here"))
        prompt = llm.generate.call_args[0][0]
        assert "Test Title" in prompt
        assert "Content here" in prompt

    async def test_summarize_text_truncated_to_2000_chars(self):
        llm = AsyncMock()
        llm.generate.return_value = "summary"
        summarizer = LLMContextSummarizer(llm_client=llm)
        long_text = "A" * 5000
        await summarizer.summarize(_doc(text=long_text))
        prompt = llm.generate.call_args[0][0]
        assert "A" * 2000 in prompt
        assert "A" * 2001 not in prompt


class TestContextualChunkEnricher:
    """ContextualChunkEnricher (line 43)."""

    @pytest.fixture
    def summarizer(self):
        m = AsyncMock()
        m.summarize.return_value = "Document summary here."
        return m

    async def test_disabled_returns_original(self):
        enricher = ContextualChunkEnricher(enabled=False)
        chunks = [_chunk("c1", "text")]
        result = await enricher.enrich(_doc(), chunks)
        assert result == chunks

    async def test_no_summarizer_returns_original(self):
        enricher = ContextualChunkEnricher(enabled=True, summarizer=None)
        chunks = [_chunk("c1", "text")]
        result = await enricher.enrich(_doc(), chunks)
        assert result == chunks

    async def test_empty_summary_returns_original(self, summarizer):
        summarizer.summarize.return_value = ""
        enricher = ContextualChunkEnricher(enabled=True, summarizer=summarizer)
        chunks = [_chunk("c1", "text")]
        result = await enricher.enrich(_doc(), chunks)
        assert result == chunks

    async def test_enrich_single_chunk(self, summarizer):
        enricher = ContextualChunkEnricher(enabled=True, summarizer=summarizer)
        chunks = [_chunk("c1", "original text")]
        result = await enricher.enrich(_doc(), chunks)
        assert len(result) == 1
        assert "[Context: Document summary here.]" in result[0].text
        assert "original text" in result[0].text

    async def test_enrich_multiple_chunks(self, summarizer):
        enricher = ContextualChunkEnricher(enabled=True, summarizer=summarizer)
        chunks = [_chunk("c1", "text1"), _chunk("c2", "text2")]
        result = await enricher.enrich(_doc(), chunks)
        assert len(result) == 2
        assert all("[Context: Document summary here.]" in c.text for c in result)

    async def test_enrich_preserves_chunk_fields(self, summarizer):
        enricher = ContextualChunkEnricher(enabled=True, summarizer=summarizer)
        chunks = [_chunk("c1", "text", source="src", title="ttl")]
        result = await enricher.enrich(_doc(), chunks)
        assert result[0].chunk_id == "c1"
        assert result[0].source_name == "src"
        assert result[0].title == "ttl"

    async def test_enrich_chunks_batch_level(self, summarizer):
        enricher = ContextualChunkEnricher(enabled=True, summarizer=summarizer, batch_size=10)
        chunks = [_chunk("c1", "text", source="s1"), _chunk("c2", "text2", source="s1")]
        result = await enricher.enrich_chunks(chunks)
        assert len(result) == 2
        assert "[Context:" in result[0].text

    async def test_enrich_chunks_disabled(self):
        enricher = ContextualChunkEnricher(enabled=False)
        chunks = [_chunk("c1", "text")]
        result = await enricher.enrich_chunks(chunks)
        assert result == chunks

    async def test_enrich_chunks_batch_error_handling(self, summarizer):
        summarizer.summarize.side_effect = [RuntimeError("fail"), "summary"]
        enricher = ContextualChunkEnricher(enabled=True, summarizer=summarizer, batch_size=10)
        chunks = [_chunk("c1", "text", source="s1"), _chunk("c2", "text2", source="s2")]
        result = await enricher.enrich_chunks(chunks)
        assert len(result) == 1
        assert "[Context: summary]" in result[0].text
