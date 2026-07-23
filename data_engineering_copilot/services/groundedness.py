"""Groundedness verifier: claim extraction + text-overlap scoring.

Annotate-only by design — never blocks answers, only reports scores.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from data_engineering_copilot.domain.models import DocumentChunk

logger = logging.getLogger(__name__)


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
    """Lightweight groundedness checker using text-overlap heuristics.

    No LLM or NLI model required. Scores each claim against context chunks.
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
