"""ASGI rate limiting middleware with per-route scoping."""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from data_engineering_copilot.services.rate_limiter import DEFAULT_LIMITS, RateLimiter

logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    """Extract client IP from X-Forwarded-For or direct connection."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that applies per-path rate limiting.

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

        # Only rate-limit configured paths
        limiter = self._limiters.get(path)
        if limiter is None:
            return await call_next(request)

        client_ip = _client_ip(request)
        if not limiter.allow(client_ip):
            logger.warning("rate_limit_exceeded path=%s ip=%s", path, client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        return await call_next(request)
