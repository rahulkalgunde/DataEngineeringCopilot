"""Integration tests for OutputGuardrails against real-world LLM output patterns.

Validates JSON parsing, boilerplate detection, short-answer rejection,
markdown fence stripping, and edge cases found in production logs.

Run with: pytest tests/integration/test_guardrails_integration.py -v -m integration
"""

from __future__ import annotations

import json

import pytest

from data_engineering_copilot.services.output_guardrails import GuardrailedAnswer, OutputGuardrails


@pytest.mark.integration
class TestValidAnswers:
    def test_valid_json_returns_guardrailed_answer(self):
        raw = json.dumps({"answer": "Apache Spark is a distributed data processing engine.", "citations": [{"url": "https://example.com"}], "confidence": 0.9})
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is not None
        assert isinstance(result, GuardrailedAnswer)
        assert result.confidence == 0.9

    def test_valid_json_with_markdown_fences(self):
        raw = '```json\n{"answer": "Spark uses RDDs for distributed data processing.", "citations": [{"url": "https://example.com"}]}\n```'
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is not None
        assert "RDDs" in result.answer

    def test_valid_json_with_code_fence_no_json_tag(self):
        raw = '```\n{"answer": "Airflow manages workflow orchestration via DAGs.", "citations": []}\n```'
        result = OutputGuardrails.verify(raw, source_count=0)
        assert result is not None
        assert "DAGs" in result.answer

    def test_no_citations_when_no_sources(self):
        raw = json.dumps({"answer": "The capital of France is Paris.", "citations": []})
        result = OutputGuardrails.verify(raw, source_count=0)
        assert result is not None


@pytest.mark.integration
class TestBoilerplateRejection:
    def test_rejects_i_cannot_answer(self):
        raw = json.dumps({"answer": "I cannot answer this question as I don't have the information.", "citations": []})
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is None

    def test_rejects_outside_my_knowledge(self):
        raw = json.dumps({"answer": "This topic is outside my knowledge base.", "citations": [{"url": "https://example.com"}]})
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is None

    def test_rejects_i_dont_have_enough(self):
        raw = json.dumps({"answer": "I don't have enough context to provide a meaningful answer.", "citations": []})
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is None

    def test_rejects_i_am_not_able_to(self):
        raw = json.dumps({"answer": "I am not able to determine the answer from the provided sources.", "citations": []})
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is None

    def test_rejects_beyond_my_knowledge(self):
        raw = json.dumps({"answer": "This question is beyond my knowledge.", "citations": []})
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is None

    def test_boilerplate_not_rejected_when_no_sources(self):
        raw = json.dumps({"answer": "I cannot answer this question.", "citations": []})
        result = OutputGuardrails.verify(raw, source_count=0)
        assert result is not None


@pytest.mark.integration
class TestShortAnswerRejection:
    def test_rejects_empty_answer(self):
        raw = json.dumps({"answer": "", "citations": []})
        result = OutputGuardrails.verify(raw, source_count=0)
        assert result is None

    def test_rejects_short_answer_with_sources(self):
        raw = json.dumps({"answer": "Yes.", "citations": [{"url": "https://example.com"}]})
        result = OutputGuardrails.verify(raw, source_count=1)
        assert result is None


@pytest.mark.integration
class TestInvalidInput:
    def test_returns_none_for_non_string(self):
        assert OutputGuardrails.verify(123, source_count=1) is None
        assert OutputGuardrails.verify(None, source_count=1) is None
        assert OutputGuardrails.verify({"answer": "test"}, source_count=1) is None

    def test_returns_none_for_invalid_json(self):
        assert OutputGuardrails.verify("{invalid json", source_count=1) is None

    def test_returns_none_for_wrong_schema(self):
        raw = json.dumps({"wrong_key": "value"})
        assert OutputGuardrails.verify(raw, source_count=1) is None

    def test_returns_none_for_answer_too_long(self):
        raw = json.dumps({"answer": "A" * 9000, "citations": []})
        assert OutputGuardrails.verify(raw, source_count=0) is None
