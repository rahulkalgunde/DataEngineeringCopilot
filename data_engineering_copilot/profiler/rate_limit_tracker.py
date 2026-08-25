"""Rate-limit & provider quota tracker for embedding/LLM HTTP services."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RateLimitEvent:
    """A single rate-limit event captured from an HTTP response."""

    timestamp: float
    provider: str  # "ollama", "openrouter", "openai"
    status_code: int
    retry_after: float | None = None
    remaining: int | None = None
    limit: int | None = None
    reset_at: float | None = None


_RATELIMIT_HEADERS = {
    "openrouter": {
        "remaining": "x-ratelimit-remaining",
        "limit": "x-ratelimit-limit",
        "reset": "x-ratelimit-reset",
    },
    "openai": {
        "remaining": "x-ratelimit-remaining-requests",
        "limit": "x-ratelimit-limit-requests",
        "reset": "x-ratelimit-reset-requests",
    },
}


def _extract_ratelimit_headers(provider: str, headers: dict[str, str]) -> dict[str, Any]:
    """Extract rate-limit headers for a known provider."""
    result: dict[str, Any] = {}
    info = _RATELIMIT_HEADERS.get(provider)
    if info is None:
        return result
    remaining_str = headers.get(info["remaining"])
    limit_str = headers.get(info["limit"])
    reset_str = headers.get(info["reset"])
    if remaining_str is not None:
        with contextlib.suppress(ValueError, TypeError):
            result["remaining"] = int(remaining_str)
    if limit_str is not None:
        with contextlib.suppress(ValueError, TypeError):
            result["limit"] = int(limit_str)
    if reset_str is not None:
        with contextlib.suppress(ValueError, TypeError):
            result["reset_at"] = time.time() + int(reset_str)
    result["retry_after"] = _parse_retry_after(headers)
    return result


def _parse_retry_after(headers: dict[str, str]) -> float | None:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


class RateLimitTracker:
    """Accumulates rate-limit events and computes provider saturation."""

    def __init__(self) -> None:
        self._events: list[RateLimitEvent] = []

    def record_response(
        self,
        provider: str,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> RateLimitEvent | None:
        """Record an HTTP response, capturing rate-limit info if present."""
        headers = headers or {}
        is_429 = status_code == 429
        rl_info = _extract_ratelimit_headers(provider, headers) if provider in _RATELIMIT_HEADERS else {}

        event = RateLimitEvent(
            timestamp=time.time(),
            provider=provider,
            status_code=status_code,
            retry_after=rl_info.get("retry_after") if is_429 else None,
            remaining=rl_info.get("remaining"),
            limit=rl_info.get("limit"),
            reset_at=rl_info.get("reset_at"),
        )
        self._events.append(event)

        if is_429:
            return event
        return None

    @property
    def total_429s(self) -> int:
        return sum(1 for e in self._events if e.status_code == 429)

    @property
    def events(self) -> list[RateLimitEvent]:
        return list(self._events)

    def provider_429_count(self, provider: str) -> int:
        return sum(1 for e in self._events if e.provider == provider and e.status_code == 429)

    def get_provider_saturation(self, provider: str) -> float:
        """Return saturation 0.0–1.0 based on remaining quota headers or 429 ratio."""
        relevant = [e for e in self._events if e.provider == provider]
        if not relevant:
            return 0.0
        remaining_values = [e.remaining for e in relevant if e.remaining is not None]
        limit_values = [e.limit for e in relevant if e.limit is not None]
        if remaining_values and limit_values:
            avg_remaining = sum(remaining_values) / len(remaining_values)
            avg_limit = sum(limit_values) / len(limit_values)
            if avg_limit > 0:
                return 1.0 - (avg_remaining / avg_limit)
        total = len(relevant)
        rate_429 = sum(1 for e in relevant if e.status_code == 429)
        return rate_429 / total if total > 0 else 0.0

    def is_throttled(self, provider: str, threshold: float = 0.1) -> bool:
        """Return True if provider is likely throttled based on 429 ratio."""
        return self.get_provider_saturation(provider) > threshold
