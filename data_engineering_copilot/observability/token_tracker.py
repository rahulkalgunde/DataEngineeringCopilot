"""LLM token usage tracker for observability."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)


class TokenTracker:
    """Thread-safe LLM token usage accumulator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._calls = 0
        self._by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "unknown",
    ) -> None:
        with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._calls += 1
            self._by_model[model]["prompt_tokens"] += prompt_tokens
            self._by_model[model]["completion_tokens"] += completion_tokens

    def get_usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(
                total_prompt_tokens=self._prompt_tokens,
                total_completion_tokens=self._completion_tokens,
                total_calls=self._calls,
                by_model=dict(self._by_model),
            )

    def reset(self) -> None:
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._calls = 0
            self._by_model.clear()


class RetrievalTracker:
    """Track retrieval score distributions for quality monitoring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scores: list[float] = []
        self._query_count: int = 0

    def record_retrieval(self, scores: list[float], query: str = "") -> None:
        """Record retrieval scores for a single query."""
        with self._lock:
            self._scores.extend(scores)
            self._query_count += 1

    def get_distribution(self) -> dict[str, float]:
        """Return score distribution statistics."""
        with self._lock:
            if not self._scores:
                return {
                    "mean": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                    "queries": 0,
                }
            sorted_scores = sorted(self._scores)
            n = len(sorted_scores)
            return {
                "mean": sum(sorted_scores) / n,
                "p50": sorted_scores[n // 2],
                "p95": sorted_scores[int(n * 0.95)],
                "p99": sorted_scores[int(n * 0.99)],
                "min": sorted_scores[0],
                "max": sorted_scores[-1],
                "count": n,
                "queries": self._query_count,
            }

    def reset(self) -> None:
        with self._lock:
            self._scores.clear()
            self._query_count = 0
