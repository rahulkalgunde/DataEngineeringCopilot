"""API authentication middleware via API key.

Checks X-API-Key header or Authorization: Bearer token against the
API_KEY environment variable. No-op if API_KEY is not set (dev mode).

When RBAC is enabled (``rbac_enabled=True``), the middleware also resolves
the API key to a ``UserPermissions`` object and stores it on
``request.state.user_permissions``. Routes can then enforce document-level
access control by reading ``allowed_sources`` from the resolved permissions.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from data_engineering_copilot.domain.models import UserPermissions

logger = logging.getLogger(__name__)


def _build_rbac_map(rbac_users_json: str) -> dict[str, UserPermissions]:
    """Parse inline JSON into an API-key-prefix → UserPermissions mapping."""
    if not rbac_users_json:
        return {}
    try:
        raw = json.loads(rbac_users_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid RBAC_USERS_JSON, RBAC disabled")
        return {}
    result: dict[str, UserPermissions] = {}
    for key_prefix, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        result[key_prefix] = UserPermissions(
            api_key_prefix=key_prefix,
            allowed_sources=tuple(entry.get("allowed_sources", [])),
            role=entry.get("role", "reader"),
        )
    return result


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests via X-API-Key or Authorization: Bearer header.

    Optionally resolves the key to RBAC permissions stored on
    ``request.state.user_permissions``.
    """

    EXEMPT_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc", "/metrics"}

    def __init__(
        self,
        app: Callable,
        api_key: str | None = None,
        rbac_enabled: bool = False,
        rbac_users_json: str = "",
    ) -> None:
        super().__init__(app)
        self._api_key = api_key or os.environ.get("API_KEY", "")
        self._rbac_enabled = rbac_enabled
        self._rbac_map = _build_rbac_map(rbac_users_json) if rbac_enabled else {}

    def _resolve_permissions(self, provided_key: str | None) -> UserPermissions | None:
        """Look up API key prefix in the RBAC map."""
        if not provided_key or not self._rbac_map:
            return None
        # Try prefixes from longest to shortest for best match
        for prefix in sorted(self._rbac_map, key=len, reverse=True):
            if provided_key.startswith(prefix):
                return self._rbac_map[prefix]
        return None

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

        # RBAC: resolve key to user permissions
        if self._rbac_enabled:
            perms = self._resolve_permissions(provided_key)
            if perms is not None:
                request.state.user_permissions = perms

        return await call_next(request)
