"""Groundedness verifier: claim extraction + LLM NLI scoring (with text-overlap fallback).

Annotate-only by design — never blocks answers, only reports scores.
Fail-open: on LLM error, falls back to text-overlap heuristic.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from data_engineering_copilot.domain.models import Answer, DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

_NLI_PROMPT = (
    "You are a groundedness verifier. Given an answer and supporting context, "
    "determine which claims in the answer are supported by the context.\n\n"
    "For each claim in the answer, output a JSON array of objects:\n"
    '[{{"claim": "...", "supported": true/false, "evidence": "..."}}]\n'
    "Return ONLY the JSON array, no preamble.\n\n"
    "ANSWER:\n{answer}\n\n"
    "CONTEXT (excerpted from documentation):\n{context}\n\n"
    "JSON array:"
)


@dataclass(frozen=True)
class ClaimAnnotation:
    claim: str
    supported: bool
    supporting_chunk_ids: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class GroundednessResult:
    overall_score: float
    annotations: tuple[ClaimAnnotation, ...]


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentence-like claims."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip().split()) >= 3]


def _tokenize(text: str) -> set[str]:
    """Lowercase word-level tokenization for overlap scoring."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _overlap_score(claim_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Jaccard-ish overlap: |intersection| / |claim|."""
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


class GroundednessVerifier:
    """Groundedness checker using LLM NLI (with text-overlap fallback).

    Primary mode: LLM-based claim verification via ``async_verify()``.
    Fallback: text-overlap heuristic via ``verify()`` (used when LLM unavailable).
    """

    def __init__(
        self,
        llm_client: object | None,
        enabled: bool = True,
        min_support_score: float = 0.3,
    ) -> None:
        self._llm_client = llm_client
        self._enabled = enabled
        self._min_support_score = min_support_score

    def extract_claims(self, answer_text: str) -> list[str]:
        """Extract individual claims from the answer text."""
        if not self._enabled or not answer_text.strip():
            return []
        return _extract_sentences(answer_text)

    async def async_verify(
        self,
        answer: Answer,
        context_chunks: list[RetrievedChunk],
    ) -> tuple[bool, list[str]]:
        """LLM-based groundedness verification (fail-open).

        Takes an ``Answer`` object and top-5 ``RetrievedChunk`` contexts.
        Returns ``(supported, unsupported_claims)``.
        On any error, falls back to the text-overlap heuristic.
        """
        if not self._enabled:
            return (True, [])

        top_5 = context_chunks[:5]
        answer_text = answer.text
        context_docs: list[DocumentChunk] = [c.chunk for c in top_5]

        if self._llm_client is None:
            return self._fallback_tuple(answer_text, context_docs)

        try:
            context_excerpt = "\n".join(
                f"[{c.source_name}] {c.text[:500]}" for c in context_docs
            )
            prompt = _NLI_PROMPT.format(
                answer=answer_text[:2000],
                context=context_excerpt[:3000],
            )
            raw = await self._llm_client.generate(prompt)

            cleaned = raw.strip()
            fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
            if fence_match:
                cleaned = fence_match.group(1).strip()

            claims_data = json.loads(cleaned)
            if not isinstance(claims_data, list):
                raise ValueError("LLM returned non-array JSON")

            annotations: list[ClaimAnnotation] = []
            for item in claims_data:
                claim = str(item.get("claim", "")).strip()
                supported = bool(item.get("supported", False))
                score = 1.0 if supported else 0.0
                annotations.append(
                    ClaimAnnotation(
                        claim=claim,
                        supported=supported,
                        supporting_chunk_ids=(),
                        score=score,
                    )
                )

            overall = sum(a.score for a in annotations) / len(annotations) if annotations else 0.0
            unsupported = [a.claim for a in annotations if not a.supported]
            return (overall >= 0.5, unsupported)
        except Exception as exc:
            logger.warning("LLM groundedness verification failed, falling back to text-overlap: %s", exc)
            return self._fallback_tuple(answer_text, context_docs)

    def _fallback_tuple(
        self, answer_text: str, context_chunks: list[DocumentChunk]
    ) -> tuple[bool, list[str]]:
        """Fallback: text-overlap heuristic returning ``(supported, unsupported_claims)``."""
        result = self.verify(answer_text, context_chunks)
        return (result.overall_score >= 0.5, [a.claim for a in result.annotations if not a.supported])

    async def async_verify_with_score(
        self,
        answer: Answer,
        context_chunks: list[RetrievedChunk],
    ) -> tuple[bool, list[str], float]:
        """LLM-based groundedness verification returning (supported, unsupported, score).

        Combines async verification + score computation in a single LLM call.
        Falls back to text-overlap heuristic on error.
        """
        if not self._enabled:
            return (True, [], 1.0)

        top_5 = context_chunks[:5]
        answer_text = answer.text
        context_docs: list[DocumentChunk] = [c.chunk for c in top_5]

        if self._llm_client is None:
            result = self.verify(answer_text, context_docs)
            return (
                result.overall_score >= 0.5,
                [a.claim for a in result.annotations if not a.supported],
                result.overall_score,
            )

        try:
            context_excerpt = "\n".join(
                f"[{c.source_name}] {c.text[:500]}" for c in context_docs
            )
            prompt = _NLI_PROMPT.format(
                answer=answer_text[:2000],
                context=context_excerpt[:3000],
            )
            raw = await self._llm_client.generate(prompt)

            cleaned = raw.strip()
            fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
            if fence_match:
                cleaned = fence_match.group(1).strip()

            claims_data = json.loads(cleaned)
            if not isinstance(claims_data, list):
                raise ValueError("LLM returned non-array JSON")

            annotations: list[ClaimAnnotation] = []
            for item in claims_data:
                claim = str(item.get("claim", "")).strip()
                supported = bool(item.get("supported", False))
                score = 1.0 if supported else 0.0
                annotations.append(
                    ClaimAnnotation(
                        claim=claim,
                        supported=supported,
                        supporting_chunk_ids=(),
                        score=score,
                    )
                )

            overall = sum(a.score for a in annotations) / len(annotations) if annotations else 0.0
            unsupported = [a.claim for a in annotations if not a.supported]
            return (overall >= 0.5, unsupported, overall)
        except Exception as exc:
            logger.warning("LLM groundedness verification failed, falling back to text-overlap: %s", exc)
            result = self.verify(answer_text, context_docs)
            return (
                result.overall_score >= 0.5,
                [a.claim for a in result.annotations if not a.supported],
                result.overall_score,
            )

    def verify(
        self,
        answer_text: str,
        context_chunks: list[DocumentChunk],
    ) -> GroundednessResult:
        """Verify groundedness of answer against context chunks.

        Returns per-claim annotations and an overall score.
        Annotate-only: does not block or filter.
        """
        if not self._enabled:
            return GroundednessResult(overall_score=1.0, annotations=())

        claims = self.extract_claims(answer_text)
        if not claims:
            return GroundednessResult(overall_score=0.0, annotations=())

        chunk_token_sets = {c.chunk_id: _tokenize(c.text) for c in context_chunks}
        annotations: list[ClaimAnnotation] = []

        for claim in claims:
            claim_tokens = _tokenize(claim)
            best_score = 0.0
            supporting_ids: list[str] = []

            for chunk_id, chunk_tokens in chunk_token_sets.items():
                score = _overlap_score(claim_tokens, chunk_tokens)
                if score >= self._min_support_score:
                    supporting_ids.append(chunk_id)
                best_score = max(best_score, score)

            annotations.append(
                ClaimAnnotation(
                    claim=claim,
                    supported=best_score >= self._min_support_score,
                    supporting_chunk_ids=tuple(supporting_ids),
                    score=round(best_score, 4),
                )
            )

        overall = sum(a.score for a in annotations) / len(annotations) if annotations else 0.0
        return GroundednessResult(
            overall_score=round(overall, 4),
            annotations=tuple(annotations),
        )
