"""Output guardrails for LLM-generated RAG answers.

Enforces structure, format, and minimal quality constraints on the JSON
output returned by the LLM before presenting to the user.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


class GuardrailedAnswer(BaseModel):
    """Pydantic-validated structured output from the LLM."""

    answer: str = Field(..., min_length=1, max_length=8192)
    citations: list[dict] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class OutputGuardrails:
    """Post-generation guardrails enforcing structure, format, and minimal quality.

    Rules enforced:
    1. JSON must parse and conform to GuardrailedAnswer schema
    2. Answer length must be > 20 chars
    3. At least one citation must be present (if sources available)
    4. No "I don't know" boilerplate when we have confident sources
    """

    BOILERPLATE_PATTERNS: list[re.Pattern] = [
        re.compile(r"i cannot answer", re.IGNORECASE),
        re.compile(r"outside my knowledge", re.IGNORECASE),
        re.compile(r"i don't have (enough|sufficient)", re.IGNORECASE),
        re.compile(r"i am not able to", re.IGNORECASE),
        re.compile(r"beyond my knowledge", re.IGNORECASE),
    ]

    @classmethod
    def verify(cls, raw: str, source_count: int) -> GuardrailedAnswer | None:
        """Verify LLM output against guardrail rules.

        Returns ``GuardrailedAnswer`` if valid, ``None`` if guardrails reject.
        """
        if not isinstance(raw, str):
            return None
        try:
            parsed = json.loads(cls._clean_json(raw))
            validated = GuardrailedAnswer(**parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

        # Reject empty / boilerplate answers when we have sources
        if validated.answer and source_count > 0:
            if len(validated.answer.strip()) < 20:
                return None
            for pattern in cls.BOILERPLATE_PATTERNS:
                if pattern.search(validated.answer):
                    return None

        return validated

    @staticmethod
    def _clean_json(raw: str) -> str:
        cleaned = raw.strip()
        # Strip markdown fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()
