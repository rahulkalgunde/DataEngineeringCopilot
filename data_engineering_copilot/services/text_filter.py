from __future__ import annotations

import logging
import re
from dataclasses import replace

from data_engineering_copilot.domain.models import DocumentChunk

logger = logging.getLogger(__name__)


class ChunkFilter:
    def __init__(
        self,
        enabled: bool = True,
        min_word_count: int = 15,
        min_alpha_ratio: float = 0.5,
        max_repetition_ratio: float = 0.3,
    ) -> None:
        self._enabled = enabled
        self.min_word_count = min_word_count
        self.min_alpha_ratio = min_alpha_ratio
        self.max_repetition_ratio = max_repetition_ratio

        self._noise_patterns = [
            re.compile(r"^\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+(INFO|WARN|DEBUG|ERROR)\s+.*$", re.MULTILINE),
            re.compile(r"org\.apache\.spark\.[a-zA-Z0-9\._]+", re.MULTILINE),
            re.compile(r"[\{\}\[\]\(\)\<\>]"),
            re.compile(r"\n{3,}"),
        ]

    def extract(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        if not self._enabled:
            return chunks
        result: list[DocumentChunk] = []
        dropped = 0
        for chunk in chunks:
            processed = self._process_one(chunk)
            if processed is not None:
                result.append(processed)
            else:
                dropped += 1
        if dropped:
            logger.info("chunk_filter dropped=%d chunks", dropped)
        return result

    def _process_one(self, chunk: DocumentChunk) -> DocumentChunk | None:
        is_sparse, metrics = self._is_sparse(chunk.text)
        if is_sparse:
            logger.debug(
                "chunk_filter.dropped chunk_id=%s reason=%s word_count=%d alpha_ratio=%.2f repetition_ratio=%.2f",
                chunk.chunk_id,
                metrics.get("reason", "unknown"),
                metrics.get("word_count", 0),
                metrics.get("alpha_ratio", 0.0),
                metrics.get("repetition_ratio", 0.0),
            )
            return None
        cleaned = self._clean_text(chunk.text)
        return replace(chunk, text=cleaned, word_count=len(cleaned.split()))

    def _clean_text(self, text: str) -> str:
        cleaned = text
        for pattern in self._noise_patterns:
            cleaned = pattern.sub(" ", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n+", "\n", cleaned).strip()
        return cleaned

    def _is_sparse(self, raw_text: str) -> tuple[bool, dict]:
        if not raw_text or not raw_text.strip():
            return True, {"reason": "empty"}

        cleaned = self._clean_text(raw_text)
        words = cleaned.split()
        word_count = len(words)

        if word_count == 0:
            return True, {"reason": "no_words"}

        char_count = len(raw_text)
        alpha_count = sum(c.isalnum() for c in raw_text)
        alpha_ratio = alpha_count / char_count if char_count > 0 else 0.0

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        unique_lines = set(lines)
        repetition_ratio = 1.0 - (len(unique_lines) / len(lines)) if lines else 0.0

        metrics = {
            "word_count": word_count,
            "alpha_ratio": round(alpha_ratio, 2),
            "repetition_ratio": round(repetition_ratio, 2),
        }

        if word_count < self.min_word_count:
            return True, {**metrics, "reason": "low_word_count"}
        if alpha_ratio < self.min_alpha_ratio:
            return True, {**metrics, "reason": "low_alpha_density"}
        if repetition_ratio > self.max_repetition_ratio:
            return True, {**metrics, "reason": "high_repetition"}

        return False, metrics

    def is_sparse(self, raw_text: str) -> bool:
        is_sparse, _ = self._is_sparse(raw_text)
        return is_sparse

    def process_chunk(self, raw_chunk: str) -> str | None:
        is_sparse, _ = self._is_sparse(raw_chunk)
        if is_sparse:
            return None
        return self._clean_text(raw_chunk)
