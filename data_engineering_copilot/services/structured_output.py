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
        if isinstance(data, dict) and "answer" in data:
            citations = data.get("citations", [])
            if not isinstance(citations, list):
                citations = []
            return StructuredAnswer(
                answer=str(data["answer"]),
                citations=citations,
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return StructuredAnswer(answer=raw, citations=[])
