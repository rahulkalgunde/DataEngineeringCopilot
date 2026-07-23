"""Tests for GroundednessVerifier — claim extraction, NLI scoring, annotation."""
from __future__ import annotations

import pytest
from data_engineering_copilot.services.groundedness import (
    ClaimAnnotation,
    GroundednessResult,
    GroundednessVerifier,
)


def _make_chunks(*texts: str) -> list:
    from data_engineering_copilot.domain.models import DocumentChunk

    return [
        DocumentChunk(chunk_id=f"c{i}", source_name="s", title="t", url="http://x", text=t)
        for i, t in enumerate(texts)
    ]


class TestClaimExtraction:
    def test_returns_at_least_one_claim(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        claims = gv.extract_claims("Spark SQL supports DataFrame operations.")
        assert len(claims) >= 1

    def test_multiple_sentences(self):
        gv = GroundednessVerifier(llm_client=None, enabled=True)
        claims = gv.extract_claims(
            "Spark SQL is a module. Delta Lake provides ACID. Both integrate well."
        )
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
