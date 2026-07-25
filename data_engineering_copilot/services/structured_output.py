"""Structured output parser for RAG responses with citation support."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class StructuredAnswer:
    answer: str
    citations: list[dict[str, str]] = field(default_factory=list)


def parse_rag_response(raw: str) -> StructuredAnswer:
    """Parse LLM response into structured answer with citations.

    Attempts JSON parsing first. Falls back to raw text if parsing fails.
    Handles ```json fenced code blocks.
    """
    if not raw or not raw.strip():
        return StructuredAnswer(answer="", citations=[])

    text = raw.strip()

    # Strip markdown JSON fencing
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

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
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return StructuredAnswer(answer=raw, citations=[])


def verify_citations(
    citations: list[dict[str, str]],
    source_names: list[str],
) -> list[dict[str, str]]:
    """Keep only citations whose source matches a retrieved source name."""
    valid_sources = set(source_names)
    return [c for c in citations if c.get("source", "") in valid_sources]
