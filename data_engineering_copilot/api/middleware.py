"""ASGI rate limiting and prompt injection detection middleware."""

from __future__ import annotations

import json
import logging
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from data_engineering_copilot.services.rate_limiter import DEFAULT_LIMITS, RateLimiter

logger = logging.getLogger(__name__)

# Prompt injection detection patterns
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|above|prior|the\s+above)\s+(instructions|prompts|directions)", re.IGNORECASE
    ),
    re.compile(r"you\s+are\s+(now|free|an?\s+AI\s+named|DAN)", re.IGNORECASE),
    re.compile(r"system\s+prompt|developer\s+mode|prompt\s+injection", re.IGNORECASE),
    re.compile(r"(REVEAL|LEAK|DUMP|DISPLAY|OUTPUT)\s+(ALL\s+)?(INSTRUCTIONS|PROMPT|SYSTEM|CONSTRAINTS)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(DAN|an?\s+AI\s+without|if\s+you\s+are)", re.IGNORECASE),
    re.compile(r"bypass|jailbreak|breach|compromise", re.IGNORECASE),
]


def _detect_prompt_injection(text: str) -> float:
    """Return 0.0–1.0 injection likelihood. >0.5 suggests rejection."""
    text_lower = text.lower()
    score = 0.0
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text_lower):
            score += 0.3
    return min(1.0, score)


def _client_ip(request: Request) -> str:
    """Extract client IP from the direct connection."""
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _rate_limit_key(request: Request) -> str:
    """Derive a rate-limit key from API key or client IP.

    API keys get their own quota; unauthenticated requests are grouped by IP.
    """
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ")
    if api_key:
        return api_key[:48]
    return _client_ip(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that applies per-path rate limiting.

    Uses API key as the rate-limit identifier when available,
    falling back to client IP for unauthenticated requests.
    Only enforces limits for paths listed in ``DEFAULT_LIMITS``.
    Other paths pass through without rate limiting.
    """

    def __init__(self, app, **kwargs) -> None:
        super().__init__(app)
        self._limiters: dict[str, RateLimiter] = {}
        for path, (max_calls, period) in DEFAULT_LIMITS.items():
            self._limiters[path] = RateLimiter(path=path, max_calls=max_calls, period_seconds=period)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Prompt injection detection for /ask endpoints
        if path in ("/api/v1/ask", "/api/v1/ask/stream") and request.method == "POST":
            body_bytes = await request.body()
            try:
                body = json.loads(body_bytes)
                question = body.get("question", "")
                if isinstance(question, str) and question.strip():
                    injection_score = _detect_prompt_injection(question)
                    if injection_score > 0.5:
                        logger.warning(
                            "prompt_injection_detected path=%s score=%.2f question=%r",
                            path,
                            injection_score,
                            question[:80],
                        )
                        return JSONResponse(
                            status_code=400,
                            content={
                                "detail": "Question rejected: detected potential prompt injection. "
                                "Rephrase your question without system-level instructions."
                            },
                        )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        limiter = self._limiters.get(path)
        if limiter is None:
            return await call_next(request)

        rl_key = _rate_limit_key(request)
        if not limiter.allow(rl_key):
            logger.warning("rate_limit_exceeded path=%s key=%s", path, rl_key)
            period = getattr(limiter, "_period_seconds", 60)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(period)},
            )
        return await call_next(request)
