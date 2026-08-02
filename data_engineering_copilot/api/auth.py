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
        self._api_key = api_key or ""
        self._rbac_enabled = rbac_enabled
        self._rbac_map = _build_rbac_map(rbac_users_json) if rbac_enabled else {}
        if not self._api_key:
            logger.warning(
                "SECURITY: API_KEY not set — authentication is DISABLED. "
                "All requests pass through unauthenticated. Set the API_KEY "
                "environment variable to enable authentication."
            )

    def _resolve_permissions(self, provided_key: str | None) -> UserPermissions | None:
        """Look up API key in the RBAC map by exact match."""
        if not provided_key or not self._rbac_map:
            return None
        return self._rbac_map.get(provided_key)

    def _audit(
        self,
        event: str,
        request: Request,
        provided_key: str | None,
        extra: dict | None = None,
    ) -> None:
        """Emit a structured audit event for authentication outcomes.

        ``key_prefix`` is truncated to the first 8 characters so the full key
        is never logged while still allowing cross-referencing in log analysis.
        """
        fields = {
            "event": event,
            "path": request.url.path,
            "ip": request.client.host if request.client else "unknown",
            "key_prefix": (provided_key or "")[:8],
        }
        if extra:
            fields.update(extra)
        logger.info("%s path=%s ip=%s key_prefix=%s", event, fields["path"], fields["ip"], fields["key_prefix"])

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
            self._audit("auth_failed", request, provided_key)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        # RBAC: resolve key to user permissions. When RBAC is enabled, a key
        # with no mapped permissions is an authorization failure (fail-closed),
        # never an implicit "everything allowed".
        if self._rbac_enabled:
            perms = self._resolve_permissions(provided_key)
            if perms is None:
                from data_engineering_copilot.domain.exceptions import AuthorizationError

                self._audit("auth_denied_no_permissions", request, provided_key)
                raise AuthorizationError("API key has no configured permissions")
            request.state.user_permissions = perms

        self._audit("auth_success", request, provided_key)
        return await call_next(request)
