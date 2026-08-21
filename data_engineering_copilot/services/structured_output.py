"""Structured output parser for RAG responses with citation support."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class StructuredAnswer:
    answer: str
    citations: list[dict[str, str]] = field(default_factory=list)


class StructuredRAGAnswer(BaseModel):
    """Schema-enforced answer shape for the generation layer.

    Designed to be ``strict``-compatible: every field is required and
    ``additionalProperties`` is false, which OpenAI-style structured outputs
    and Ollama constrained decoding require.
    """

    answer: str = Field(description="The final user-facing answer text.")
    citations: list[str] = Field(default_factory=list, description="Source identifiers cited in the answer.")
    missing_info: bool = Field(default=False, description="True if the context does not fully answer the question.")


# Strict-compatible JSON schema derived from ``StructuredRAGAnswer``. Kept
# explicit (rather than ``model_json_schema()``) so we guarantee the
# required/additionalProperties shape that strict structured outputs need.
STRUCTURED_RAG_ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "missing_info": {"type": "boolean"},
    },
    "required": ["answer", "citations", "missing_info"],
    "additionalProperties": False,
}


def _strip_fenced(text: str) -> str:
    """Remove a leading/trailing ```json fenced code block if present."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def parse_rag_response(raw: str) -> StructuredAnswer:
    """Parse LLM response into structured answer with citations.

    Attempts JSON parsing first. Falls back to raw text if parsing fails.
    Handles ```json fenced code blocks.

    Returns an empty ``answer`` when JSON is parseable but lacks a
    recognized answer field, enabling JSON retry in the RAG service.
    """
    if not raw or not raw.strip():
        return StructuredAnswer(answer="", citations=[])

    text = _strip_fenced(raw.strip())

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            answer = data.get("answer") or data.get("response") or data.get("text") or data.get("content")
            if answer is not None:
                citations = data.get("citations", [])
                if not isinstance(citations, list):
                    citations = []
                return StructuredAnswer(
                    answer=str(answer),
                    citations=citations,
                )
            return StructuredAnswer(answer="", citations=[])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return StructuredAnswer(answer=raw, citations=[])


def parse_structured_rag_response(raw: str) -> StructuredAnswer:
    """Prefer schema-enforced parsing, fall back to the permissive parser.

    When the model emitted a schema-valid JSON object we extract the fields
    via ``StructuredRAGAnswer``; otherwise we degrade to ``parse_rag_response``
    (which handles fenced blocks and raw text). Guarantees a ``StructuredAnswer``
    for either path so downstream callers stay type-stable.
    """
    if not raw or not raw.strip():
        return StructuredAnswer(answer="", citations=[])

    text = _strip_fenced(raw.strip())

    try:
        validated = StructuredRAGAnswer.model_validate_json(text)
        return StructuredAnswer(answer=validated.answer, citations=[{"source": c} for c in validated.citations])
    except Exception:
        return parse_rag_response(raw)


def verify_citations(
    citations: list[dict[str, str]],
    source_names: list[str],
) -> list[dict[str, str]]:
    """Keep only citations whose source matches a retrieved source name."""
    valid_sources = set(source_names)
    return [c for c in citations if c.get("source", "") in valid_sources]
