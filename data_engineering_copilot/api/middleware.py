"""ASGI rate limiting middleware with per-route scoping."""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from data_engineering_copilot.services.rate_limiter import DEFAULT_LIMITS, RateLimiter

logger = logging.getLogger(__name__)


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

        limiter = self._limiters.get(path)
        if limiter is None:
            return await call_next(request)

        rl_key = _rate_limit_key(request)
        if not limiter.allow(rl_key):
            logger.warning("rate_limit_exceeded path=%s key=%s", path, rl_key)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        return await call_next(request)
