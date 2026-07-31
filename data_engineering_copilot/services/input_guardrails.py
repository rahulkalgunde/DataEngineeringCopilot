"""Input guardrails against indirect prompt injection via retrieved documents.

Retrieved chunks are untrusted content that gets inserted verbatim into the
LLM prompt. A crawled page containing embedded instructions (e.g. "ignore
previous instructions") could hijack the model. This module scans retrieved
chunks and drops the ones that look like injection attempts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from data_engineering_copilot.domain.models import RetrievedChunk
from data_engineering_copilot.services.prompt_injection import INJECTION_THRESHOLD, detect_prompt_injection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputGuardrailResult:
    """Result of scanning a set of retrieved chunks."""

    kept: list[RetrievedChunk]
    rejected_count: int


class InputGuardrails:
    """Filters retrieved chunks that contain prompt-injection patterns.

    Cheap, regex-based defense-in-depth. Intentionally conservative: only
    chunks whose raw text scores at or above the injection threshold are
    dropped, and a note is logged so operators can review false positives.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def scan_chunks(self, chunks: list[RetrievedChunk]) -> InputGuardrailResult:
        """Return chunks that passed the injection scan.

        When disabled (or when no chunks are provided) this is a pass-through.
        """
        if not self._enabled or not chunks:
            return InputGuardrailResult(kept=chunks, rejected_count=0)

        kept: list[RetrievedChunk] = []
        rejected = 0
        for chunk in chunks:
            text = chunk.chunk.text
            if not text:
                kept.append(chunk)
                continue
            score = detect_prompt_injection(text[:4000])
            if score >= INJECTION_THRESHOLD:
                rejected += 1
                logger.warning(
                    "indirect_prompt_injection_blocked score=%.2f source=%s chunk=%s",
                    score,
                    chunk.chunk.source_name,
                    chunk.chunk.chunk_id,
                )
            else:
                kept.append(chunk)

        if rejected:
            logger.info(
                "input_guardrails kept=%d rejected=%d",
                len(kept),
                rejected,
            )
        return InputGuardrailResult(kept=kept, rejected_count=rejected)
