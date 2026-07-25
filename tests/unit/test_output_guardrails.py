"""Tests for output_guardrails.OutputGuardrails (output_guardrails.py:23-72)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from data_engineering_copilot.services.output_guardrails import (
    GuardrailedAnswer,
    OutputGuardrails,
)


class TestGuardrailedAnswerModel:
    """GuardrailedAnswer Pydantic model (line 15)."""

    def test_valid_answer_with_citations(self):
        ans = GuardrailedAnswer(answer="This is a valid answer text.", citations=[{"url": "http://x"}], confidence=0.8)
        assert ans.answer == "This is a valid answer text."
        assert len(ans.citations) == 1
        assert ans.confidence == 0.8

    def test_defaults(self):
        ans = GuardrailedAnswer(answer="ok")
        assert ans.citations == []
        assert ans.confidence == 0.0

    def test_rejects_empty_answer(self):
        with pytest.raises(ValidationError):
            GuardrailedAnswer(answer="")

    def test_rejects_too_long_answer(self):
        with pytest.raises(ValidationError):
            GuardrailedAnswer(answer="x" * 8193)

    def test_rejects_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            GuardrailedAnswer(answer="ok", confidence=1.5)
        with pytest.raises(ValidationError):
            GuardrailedAnswer(answer="ok", confidence=-0.1)


class TestCleanJson:
    """_clean_json static method (line 65)."""

    def test_plain_json_unchanged(self):
        raw = '{"answer": "hello", "citations": [], "confidence": 0.9}'
        assert OutputGuardrails._clean_json(raw) == raw

    def test_strips_markdown_json_fence(self):
        raw = '```json\n{"answer": "hello"}\n```'
        assert OutputGuardrails._clean_json(raw) == '{"answer": "hello"}'

    def test_strips_plain_fence(self):
        raw = '```\n{"answer": "hello"}\n```'
        assert OutputGuardrails._clean_json(raw) == '{"answer": "hello"}'

    def test_strips_whitespace_around_json(self):
        raw = '  {"answer": "hello"}  '
        assert OutputGuardrails._clean_json(raw) == '{"answer": "hello"}'

    def test_multiline_json_with_fence(self):
        raw = '```json\n{"answer": "multi\nline"}\n```'
        cleaned = OutputGuardrails._clean_json(raw)
        assert cleaned == '{"answer": "multi\nline"}'


class TestVerify:
    """OutputGuardrails.verify class method (line 42)."""

    def test_valid_json_passes(self):
        raw = json.dumps({"answer": "The API returns status 200 on success.", "citations": [], "confidence": 0.9})
        result = OutputGuardrails.verify(raw, source_count=2)
        assert result is not None
        assert result.answer == "The API returns status 200 on success."
        assert result.confidence == 0.9

    def test_non_string_returns_none(self):
        assert OutputGuardrails.verify(123, source_count=1) is None
        assert OutputGuardrails.verify(None, source_count=1) is None
        assert OutputGuardrails.verify([], source_count=1) is None

    def test_invalid_json_returns_none(self):
        assert OutputGuardrails.verify("not json at all", source_count=1) is None

    def test_missing_answer_field_returns_none(self):
        raw = json.dumps({"citations": [], "confidence": 0.5})
        assert OutputGuardrails.verify(raw, source_count=1) is None

    def test_wrong_answer_type_returns_none(self):
        raw = json.dumps({"answer": 123, "citations": [], "confidence": 0.5})
        assert OutputGuardrails.verify(raw, source_count=1) is None

    def test_answer_too_long_returns_none(self):
        raw = json.dumps({"answer": "x" * 8193, "citations": [], "confidence": 0.5})
        assert OutputGuardrails.verify(raw, source_count=1) is None

    def test_short_answer_with_sources_returns_none(self):
        raw = json.dumps({"answer": "Too short.", "citations": [], "confidence": 0.5})
        assert OutputGuardrails.verify(raw, source_count=5) is None

    def test_short_answer_without_sources_passes(self):
        raw = json.dumps({"answer": "Short.", "citations": [], "confidence": 0.5})
        result = OutputGuardrails.verify(raw, source_count=0)
        assert result is not None
        assert result.answer == "Short."

    def test_answer_exactly_20_chars_with_sources_passes(self):
        raw = json.dumps({"answer": "a" * 20, "citations": [], "confidence": 0.5})
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is not None


class TestBoilerplateRejection:
    """Boilerplate patterns (lines 33-39) reject canned answers when sources exist."""

    @pytest.mark.parametrize(
        "boilerplate",
        [
            "I cannot answer this question.",
            "This is outside my knowledge.",
            "I don't have enough information.",
            "I am not able to help with this.",
            "This is beyond my knowledge base.",
        ],
    )
    def test_rejects_boilerplate_with_sources(self, boilerplate: str):
        raw = json.dumps({"answer": boilerplate, "citations": [], "confidence": 0.5})
        assert OutputGuardrails.verify(raw, source_count=3) is None

    @pytest.mark.parametrize(
        "boilerplate",
        [
            "I cannot answer this question.",
            "This is outside my knowledge.",
            "I don't have enough information.",
            "I am not able to help with this.",
            "This is beyond my knowledge base.",
        ],
    )
    def test_allows_boilerplate_without_sources(self, boilerplate: str):
        raw = json.dumps({"answer": boilerplate, "citations": [], "confidence": 0.0})
        result = OutputGuardrails.verify(raw, source_count=0)
        assert result is not None
        assert result.answer == boilerplate

    def test_case_insensitive_boilerplate(self):
        raw = json.dumps({"answer": "I CANNOT ANSWER this.", "citations": [], "confidence": 0.0})
        assert OutputGuardrails.verify(raw, source_count=1) is None

    def test_boilerplate_in_longer_text(self):
        raw = json.dumps(
            {"answer": "Based on the docs, I cannot answer this question fully.", "citations": [], "confidence": 0.5}
        )
        assert OutputGuardrails.verify(raw, source_count=1) is None


class TestVerifyWithFencedJson:
    """Edge case: valid JSON wrapped in markdown fences."""

    def test_verify_with_json_fence(self):
        raw = '```json\n{"answer": "This answer explains the API endpoint thoroughly.", "confidence": 0.8}\n```'
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is not None
        assert result.answer == "This answer explains the API endpoint thoroughly."

    def test_verify_with_plain_fence(self):
        raw = '```\n{"answer": "This answer explains the API endpoint thoroughly.", "confidence": 0.8}\n```'
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is not None
        assert result.answer == "This answer explains the API endpoint thoroughly."
