"""Tests for structured output parsing (JSON citations)."""

from __future__ import annotations

import json

from data_engineering_copilot.services.structured_output import StructuredAnswer, parse_rag_response


class TestParseRagResponse:
    def test_valid_json_response(self):
        raw = json.dumps(
            {
                "answer": "Spark SQL is a module.",
                "citations": [
                    {"source": "spark-docs", "snippet": "Spark SQL provides DataFrame API."},
                ],
            }
        )
        result = parse_rag_response(raw)
        assert isinstance(result, StructuredAnswer)
        assert result.answer == "Spark SQL is a module."
        assert len(result.citations) == 1
        assert result.citations[0]["source"] == "spark-docs"

    def test_json_with_markdown_fencing(self):
        raw = '```json\n{"answer": "Test answer.", "citations": []}\n```'
        result = parse_rag_response(raw)
        assert result.answer == "Test answer."
        assert result.citations == []

    def test_invalid_json_returns_raw_text(self):
        raw = "Spark SQL is a module for data processing."
        result = parse_rag_response(raw)
        assert result.answer == raw
        assert result.citations == []

    def test_empty_string(self):
        result = parse_rag_response("")
        assert result.answer == ""
        assert result.citations == []

    def test_json_with_extra_fields(self):
        raw = json.dumps(
            {
                "answer": "Answer text.",
                "citations": [],
                "confidence": 0.95,
            }
        )
        result = parse_rag_response(raw)
        assert result.answer == "Answer text."

    def test_json_missing_answer_key(self):
        raw = json.dumps({"citations": []})
        result = parse_rag_response(raw)
        assert result.answer == raw
        assert result.citations == []
