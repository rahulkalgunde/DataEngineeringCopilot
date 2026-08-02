"""Unit tests for ApiKeyAuthMiddleware — API key sourcing and RBAC enforcement."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from data_engineering_copilot.api.auth import ApiKeyAuthMiddleware
from data_engineering_copilot.domain.exceptions import AuthorizationError


def _make_request(path: str = "/protected", headers: dict | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
        "scheme": "http",
    }
    return Request(scope)


async def _ok_next(_request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


def _make_middleware(**kwargs) -> ApiKeyAuthMiddleware:
    return ApiKeyAuthMiddleware(app=MagicMock(), **kwargs)


class TestApiKeyFromSettings:
    def test_api_key_comes_from_constructor_not_os_environ(self, monkeypatch) -> None:
        """The middleware must not read os.environ directly (F-11)."""
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("API_KEY", "os-env-key-should-not-win")

        mw = _make_middleware(api_key="settings-key")

        assert mw._api_key == "settings-key"

    def test_empty_api_key_disables_auth(self) -> None:
        mw = _make_middleware(api_key="")
        assert mw._api_key == ""


class TestUnmappedKeyRaisesAuthorizationError:
    @pytest.mark.asyncio
    async def test_unmapped_key_raises_authorization_error(self) -> None:
        """With RBAC enabled, a valid but unmapped key must raise (403), not pass (F-11)."""
        mw = _make_middleware(
            api_key="test-key-123",
            rbac_enabled=True,
            rbac_users_json='{"sk-mapped": {"allowed_sources": ["A"], "role": "reader"}}',
        )
        request = _make_request(headers={"X-API-Key": "test-key-123"})

        with pytest.raises(AuthorizationError):
            await mw.dispatch(request, _ok_next)

    @pytest.mark.asyncio
    async def test_mapped_key_passes_through(self) -> None:
        mw = _make_middleware(
            api_key="test-key-123",
            rbac_enabled=True,
            rbac_users_json='{"test-key-123": {"allowed_sources": ["A"], "role": "reader"}}',
        )
        request = _make_request(headers={"X-API-Key": "test-key-123"})

        response = await mw.dispatch(request, _ok_next)
        assert response.status_code == 200
        assert getattr(request.state, "user_permissions", None) is not None

    @pytest.mark.asyncio
    async def test_wrong_key_returns_401_not_403(self) -> None:
        mw = _make_middleware(
            api_key="test-key-123",
            rbac_enabled=True,
            rbac_users_json='{"test-key-123": {"allowed_sources": ["A"], "role": "reader"}}',
        )
        request = _make_request(headers={"X-API-Key": "wrong-key"})

        response = await mw.dispatch(request, _ok_next)
        assert response.status_code == 401
