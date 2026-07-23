"""Tests for GroundednessVerifier — claim extraction, NLI scoring, annotation."""

from __future__ import annotations

from unittest.mock import AsyncMock

from data_engineering_copilot.domain.models import Answer, DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.groundedness import (
    GroundednessResult,
    GroundednessVerifier,
)


def _make_chunks(*texts: str) -> list[DocumentChunk]:
    return [
        DocumentChunk(chunk_id=f"c{i}", source_name="s", title="t", url="http://x", text=t) for i, t in enumerate(texts)
    ]


def _make_retrieved(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=DocumentChunk(chunk_id=f"c{i}", source_name="s", title="t", url="http://x", text=t),
            distance=0.1,
            confidence=0.9,
        )
        for i, t in enumerate(texts)
    ]


def _make_answer(text: str = "Spark SQL is a module.") -> Answer:
    return Answer(
        text=text,
        sources=(
            DocumentChunk(chunk_id="c0", source_name="s", title="t", url="http://x", text=text),
        ),
        confidence=0.9,
    )


class TestClaimExtraction:
    def test_returns_at_least_one_claim(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        claims = gv.extract_claims("Spark SQL supports DataFrame operations.")
        assert len(claims) >= 1

    def test_multiple_sentences(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        claims = gv.extract_claims("Spark SQL is a module. Delta Lake provides ACID. Both integrate well.")
        assert len(claims) >= 2

    def test_disabled_returns_empty(self):
        gv = GroundednessVerifier(llm_client=None, enabled=False)
        claims = gv.extract_claims("Any text here.")
        assert claims == []


class TestVerify:
    def test_disabled_returns_passthrough(self):
        gv = GroundednessVerifier(llm_client=None, enabled=False)
        result = gv.verify("Spark SQL is great.", _make_chunks("Spark SQL module"))
        assert result.overall_score == 1.0
        assert result.annotations == ()

    def test_enabled_with_text_match(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        chunks = _make_chunks("Spark SQL is a powerful module for structured data.")
        result = gv.verify("Spark SQL is a powerful module.", chunks)
        assert 0.0 <= result.overall_score <= 1.0
        assert len(result.annotations) >= 1

    def test_no_match_returns_low_score(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        chunks = _make_chunks("Kafka Streams processes real-time data.")
        result = gv.verify("Spark SQL is great.", chunks)
        assert result.overall_score < 0.5

    def test_empty_answer_returns_zero(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        result = gv.verify("", _make_chunks("some context"))
        assert result.overall_score == 0.0

    def test_annotate_only_does_not_block(self):
        """Groundedness is annotate-only — never blocks the answer."""
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        result = gv.verify("Complete hallucination about unicorns.", _make_chunks("real context"))
        # Even low score should still return a result, not raise
        assert isinstance(result, GroundednessResult)
        assert result.overall_score >= 0.0


class TestAnnotation:
    def test_annotation_has_required_fields(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        chunks = _make_chunks("Spark SQL provides DataFrame API.")
        result = gv.verify("Spark SQL provides DataFrame API.", chunks)
        for ann in result.annotations:
            assert hasattr(ann, "claim")
            assert hasattr(ann, "supported")
            assert hasattr(ann, "supporting_chunk_ids")
            assert hasattr(ann, "score")


class TestAsyncVerify:
    async def test_disabled_returns_true_empty(self):
        gv = GroundednessVerifier(llm_client=None, enabled=False)
        supported, unsupported = await gv.async_verify(
            _make_answer(), _make_retrieved("Some context.")
        )
        assert supported is True
        assert unsupported == []

    async def test_no_llm_client_falls_back_to_text_overlap(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        supported, unsupported = await gv.async_verify(
            _make_answer("Spark SQL module."),
            _make_retrieved("Spark SQL is a module for data."),
        )
        assert isinstance(supported, bool)
        assert isinstance(unsupported, list)

    async def test_llm_success_returns_tuple(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = (
            '[{"claim": "Spark SQL is a module.", "supported": true}]'
        )
        gv = GroundednessVerifier(llm_client=mock_llm, enabled=True)
        supported, unsupported = await gv.async_verify(
            _make_answer("Spark SQL is a module."),
            _make_retrieved("Spark SQL is a module for data."),
        )
        assert supported is True
        assert unsupported == []

    async def test_llm_unsupported_claim(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = (
            '[{"claim": "Unicorns exist.", "supported": false}]'
        )
        gv = GroundednessVerifier(llm_client=mock_llm, enabled=True)
        supported, unsupported = await gv.async_verify(
            _make_answer("Unicorns exist."),
            _make_retrieved("Spark SQL is a module."),
        )
        assert supported is False
        assert len(unsupported) == 1

    async def test_llm_error_falls_back_to_text_overlap(self):
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        gv = GroundednessVerifier(llm_client=mock_llm, enabled=True)
        supported, unsupported = await gv.async_verify(
            _make_answer("Spark SQL module."),
            _make_retrieved("Spark SQL is a module for data."),
        )
        assert isinstance(supported, bool)
        assert isinstance(unsupported, list)

    async def test_uses_top_5_chunks_only(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = (
            '[{"claim": "Test claim.", "supported": true}]'
        )
        gv = GroundednessVerifier(llm_client=mock_llm, enabled=True)
        chunks = _make_retrieved(*[f"Chunk {i} text." for i in range(10)])
        answer = _make_answer("Test claim.")
        supported, unsupported = await gv.async_verify(answer, chunks)
        assert supported is True
        assert len(mock_llm.generate.call_args[0]) == 1
        prompt_arg = mock_llm.generate.call_args[0][0]
        assert "Chunk 0" in prompt_arg
        assert "Chunk 5" not in prompt_arg
