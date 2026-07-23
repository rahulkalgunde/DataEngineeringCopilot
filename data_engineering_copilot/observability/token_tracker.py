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
