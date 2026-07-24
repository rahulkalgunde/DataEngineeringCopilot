"""API authentication middleware via API key.

Checks X-API-Key header or Authorization: Bearer token against the
API_KEY environment variable. No-op if API_KEY is not set (dev mode).
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests via X-API-Key or Authorization: Bearer header."""

    EXEMPT_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc", "/metrics"}

    def __init__(self, app: Callable, api_key: str | None = None) -> None:
        super().__init__(app)
        self._api_key = api_key or os.environ.get("API_KEY", "")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        if not self._api_key:
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key")
        if not provided_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                provided_key = auth_header[7:]

        if not provided_key or not hmac.compare_digest(provided_key, self._api_key):
            logger.warning(
                "Auth failed path=%s ip=%s",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
