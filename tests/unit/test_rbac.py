"""Unit tests for document-level RBAC (source filter enforcement)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from data_engineering_copilot.api.auth import ApiKeyAuthMiddleware, _build_rbac_map
from data_engineering_copilot.api.routes import _resolve_source_filter
from data_engineering_copilot.domain.models import UserPermissions


class TestBuildRbacMap:
    def test_empty_json_returns_empty(self) -> None:
        assert _build_rbac_map("") == {}

    def test_invalid_json_returns_empty(self) -> None:
        assert _build_rbac_map("not json") == {}

    def test_valid_json_creates_permissions(self) -> None:
        rbac_json = json.dumps(
            {
                "sk-abc123": {"allowed_sources": ["Spark", "Delta"], "role": "reader"},
                "sk-admin1": {"allowed_sources": [], "role": "admin"},
            }
        )
        result = _build_rbac_map(rbac_json)
        assert len(result) == 2
        assert result["sk-abc123"].allowed_sources == ("Spark", "Delta")
        assert result["sk-abc123"].role == "reader"
        assert result["sk-admin1"].role == "admin"

    def test_defaults_role_to_reader(self) -> None:
        rbac_json = json.dumps({"sk-x": {"allowed_sources": ["A"]}})
        result = _build_rbac_map(rbac_json)
        assert result["sk-x"].role == "reader"


class TestResolveSourceFilter:
    def _make_request(self, user_permissions: UserPermissions | None = None) -> MagicMock:
        request = MagicMock()
        request.state.user_permissions = user_permissions
        return request

    def test_no_permissions_passthrough(self) -> None:
        request = self._make_request(None)
        assert _resolve_source_filter(request, ["A"]) == ["A"]
        assert _resolve_source_filter(request, None) is None

    def test_admin_bypasses_filter(self) -> None:
        perms = UserPermissions(api_key_prefix="sk-admin", role="admin", allowed_sources=("A",))
        request = self._make_request(perms)
        assert _resolve_source_filter(request, ["B"]) == ["B"]
        assert _resolve_source_filter(request, None) is None

    def test_reader_with_no_client_filter(self) -> None:
        perms = UserPermissions(api_key_prefix="sk-x", role="reader", allowed_sources=("Spark", "Delta"))
        request = self._make_request(perms)
        result = _resolve_source_filter(request, None)
        assert result == ["Spark", "Delta"]

    def test_reader_intersects_with_client_filter(self) -> None:
        perms = UserPermissions(api_key_prefix="sk-x", role="reader", allowed_sources=("Spark", "Delta"))
        request = self._make_request(perms)
        result = _resolve_source_filter(request, ["Spark", "Airflow"])
        assert result == ["Spark"]

    def test_reader_empty_allowed_sources_passthrough(self) -> None:
        perms = UserPermissions(api_key_prefix="sk-x", role="reader", allowed_sources=())
        request = self._make_request(perms)
        assert _resolve_source_filter(request, ["A"]) == ["A"]


class TestRbacMiddlewareResolvePermissions:
    def _make_middleware(self, rbac_json: str) -> ApiKeyAuthMiddleware:
        app = MagicMock()
        return ApiKeyAuthMiddleware(
            app,
            api_key="test-key-123",
            rbac_enabled=True,
            rbac_users_json=rbac_json,
        )

    def test_resolve_exact_match(self) -> None:
        rbac_json = json.dumps(
            {
                "sk-abc-def-xyz": {"allowed_sources": ["B"], "role": "admin"},
                "sk-abc": {"allowed_sources": ["A"], "role": "reader"},
            }
        )
        mw = self._make_middleware(rbac_json)
        perms = mw._resolve_permissions("sk-abc-def-xyz")
        assert perms is not None
        assert perms.allowed_sources == ("B",)
        assert perms.role == "admin"

    def test_resolve_parent_prefix_does_not_match(self) -> None:
        rbac_json = json.dumps({"sk-abc": {"allowed_sources": ["A"]}})
        mw = self._make_middleware(rbac_json)
        assert mw._resolve_permissions("sk-abc-def") is None

    def test_resolve_no_match(self) -> None:
        rbac_json = json.dumps({"sk-abc": {"allowed_sources": ["A"]}})
        mw = self._make_middleware(rbac_json)
        assert mw._resolve_permissions("sk-xyz") is None

    def test_resolve_none_key(self) -> None:
        rbac_json = json.dumps({"sk-abc": {"allowed_sources": ["A"]}})
        mw = self._make_middleware(rbac_json)
        assert mw._resolve_permissions(None) is None
