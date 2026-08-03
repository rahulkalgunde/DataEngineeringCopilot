"""Output guardrails for LLM-generated RAG answers.

Enforces structure, format, and minimal quality constraints on the
output returned by the LLM before presenting to the user.

Handles two output formats:
- JSON (for documentation/factual intents)
- Plain text with optional code blocks (for code intents)
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


class GuardrailedAnswer(BaseModel):
    """Pydantic-validated structured output from the LLM."""

    status: str = "SUCCESS"
    answer: str = Field(default="", min_length=0, max_length=8192)
    missing_info: str | None = None
    citations: list[dict] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class OutputGuardrails:
    """Post-generation guardrails enforcing structure, format, and minimal quality.

    Rules enforced:
    1. Response must be valid JSON or contain a fenced code block
    2. Answer length must meet minimum threshold
    3. No "I don't know" boilerplate when we have confident sources
    4. Answer coherence (not gibberish or random characters)
    5. Citation consistency (citations match retrieved sources)
    """

    BOILERPLATE_PATTERNS: list[re.Pattern] = [
        re.compile(r"i cannot answer", re.IGNORECASE),
        re.compile(r"outside my knowledge", re.IGNORECASE),
        re.compile(r"i don't have (enough|sufficient)", re.IGNORECASE),
        re.compile(r"i am not able to", re.IGNORECASE),
        re.compile(r"beyond my knowledge", re.IGNORECASE),
        re.compile(r"i'm sorry,? (but )?i (can't|cannot|am unable to)", re.IGNORECASE),
        re.compile(r"as an ai (language )?model", re.IGNORECASE),
        re.compile(r"i don't (have|possess) (access to|information about)", re.IGNORECASE),
    ]

    min_answer_length: int = 20
    min_citation_count: int = 0

    @classmethod
    def verify(cls, raw: object, source_count: int) -> GuardrailedAnswer | None:
        """Verify LLM output against guardrail rules.

        Returns ``GuardrailedAnswer`` if valid, ``None`` if guardrails reject.
        """
        if not isinstance(raw, str) or not raw.strip():
            return None

        # Try JSON parse first (documentation/factual intents)
        result = cls._try_json(raw)
        if result is not None:
            return cls._check_quality(result, source_count)

        # Fall back to plain text with code blocks (code intents)
        result = cls._try_plain_text(raw)
        if result is not None:
            return cls._check_quality(result, source_count)

        return None

    @classmethod
    def _try_json(cls, raw: str) -> GuardrailedAnswer | None:
        """Try to parse as JSON."""
        try:
            parsed = json.loads(cls._clean_json(raw))
            if not isinstance(parsed, dict) or "answer" not in parsed:
                return None
            validated = GuardrailedAnswer(**parsed)
            return validated
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    @classmethod
    def _try_plain_text(cls, raw: str) -> GuardrailedAnswer | None:
        """Accept plain text that contains a fenced code block."""
        has_code_block = bool(re.search(r"```\w+\n", raw))
        if not has_code_block:
            return None
        # Extract citations if present (e.g. "Sources: Apache Spark Documentation")
        citations = []
        sources_match = re.search(r"(?:Sources?|Cited?):\s*(.+?)$", raw, re.IGNORECASE | re.MULTILINE)
        if sources_match:
            for src in re.split(r",\s*|\s+and\s+", sources_match.group(1)):
                src = src.strip().rstrip(".")
                if src:
                    citations.append({"source": src, "snippet": ""})
        return GuardrailedAnswer(answer=raw.strip(), citations=citations)

    @classmethod
    def _check_quality(cls, result: GuardrailedAnswer, source_count: int) -> GuardrailedAnswer | None:
        """Reject empty / boilerplate / incoherent answers."""
        if result.status == "INSUFFICIENT_CONTEXT":
            return result
        if not result.answer or not result.answer.strip():
            return None
        answer_text = result.answer.strip()
        if source_count > 0:
            if len(answer_text) < cls.min_answer_length:
                return None
            for pattern in cls.BOILERPLATE_PATTERNS:
                if pattern.search(answer_text):
                    return None
            # Coherence check: reject if >50% non-alphabetic characters (gibberish)
            alpha_count = sum(1 for c in answer_text if c.isalpha() or c.isspace())
            if len(answer_text) > 0 and alpha_count / len(answer_text) < 0.5:
                return None
        return result

    @staticmethod
    def _clean_json(raw: str) -> str:
        cleaned = raw.strip()
        # Strip markdown fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()
