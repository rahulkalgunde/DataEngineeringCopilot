"""Tests for api/routes.py pure functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.api.routes import (
    _build_cache_scope,
    _extract_user_session,
    _resolve_source_filter,
)


class TestResolveSourceFilter:
    def test_passes_through_when_rbac_disabled(self) -> None:
        result = _resolve_source_filter(None, ["src1", "src2"], rbac_enabled=False)
        assert result == ["src1", "src2"]

    def test_passes_through_none_when_rbac_disabled(self) -> None:
        result = _resolve_source_filter(None, None, rbac_enabled=False)
        assert result is None

    def test_admin_bypasses_rbac(self) -> None:
        mock_request = MagicMock()
        mock_perms = MagicMock()
        mock_perms.role = "admin"
        mock_request.state.user_permissions = mock_perms

        result = _resolve_source_filter(mock_request, ["src1"], rbac_enabled=True)
        assert result == ["src1"]

    def test_reader_restricted_to_allowed_sources(self) -> None:
        mock_request = MagicMock()
        mock_perms = MagicMock()
        mock_perms.role = "reader"
        mock_perms.allowed_sources = ["src1", "src2"]
        mock_request.state.user_permissions = mock_perms

        result = _resolve_source_filter(mock_request, ["src1", "src3"], rbac_enabled=True)
        assert result == ["src1"]

    def test_reader_with_no_client_filter_gets_allowed_sources(self) -> None:
        mock_request = MagicMock()
        mock_perms = MagicMock()
        mock_perms.role = "reader"
        mock_perms.allowed_sources = ["src1", "src2"]
        mock_request.state.user_permissions = mock_perms

        result = _resolve_source_filter(mock_request, None, rbac_enabled=True)
        assert result == ["src1", "src2"]

    def test_no_permissions_raises_authorization_error(self) -> None:
        from data_engineering_copilot.domain.exceptions import AuthorizationError

        mock_request = MagicMock()
        mock_request.state.user_permissions = None

        with pytest.raises(AuthorizationError, match="no resolved permissions"):
            _resolve_source_filter(mock_request, ["src1"], rbac_enabled=True)

    def test_reader_with_empty_allowed_sources_raises(self) -> None:
        from data_engineering_copilot.domain.exceptions import AuthorizationError

        mock_request = MagicMock()
        mock_perms = MagicMock()
        mock_perms.role = "reader"
        mock_perms.allowed_sources = []
        mock_request.state.user_permissions = mock_perms

        with pytest.raises(AuthorizationError, match="no permitted sources"):
            _resolve_source_filter(mock_request, ["src1"], rbac_enabled=True)

    def test_empty_intersection_raises_authorization_error(self) -> None:
        from data_engineering_copilot.domain.exceptions import AuthorizationError

        mock_request = MagicMock()
        mock_perms = MagicMock()
        mock_perms.role = "reader"
        mock_perms.allowed_sources = ["src1"]
        mock_request.state.user_permissions = mock_perms

        with pytest.raises(AuthorizationError, match="not permitted"):
            _resolve_source_filter(mock_request, ["src2"], rbac_enabled=True)


class TestBuildCacheScope:
    def test_builds_cache_scope_with_defaults(self) -> None:
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "default"
        mock_request.state.user_permissions = None

        with (
            patch("data_engineering_copilot.api.routes.settings") as mock_settings,
            patch("data_engineering_copilot.config.settings.resolve_active_generation", return_value="gen1"),
            patch("data_engineering_copilot.evaluation.provenance.answer_config_fingerprint", return_value="fp1"),
        ):
            mock_settings.active_embedding_model_name.return_value = "model1"
            mock_settings.active_collection_name = "col1"
            mock_settings.collection_name = "col1"

            result = _build_cache_scope(mock_request, None)

        assert result.tenant_id == "default"
        assert result.role == "anonymous"
        assert result.source_filter == ()
        assert result.embedding_model == "model1"
        assert result.index_generation == "gen1"

    def test_builds_cache_scope_with_permissions(self) -> None:
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "tenant1"
        mock_perms = MagicMock()
        mock_perms.role = "admin"
        mock_request.state.user_permissions = mock_perms

        with (
            patch("data_engineering_copilot.api.routes.settings") as mock_settings,
            patch("data_engineering_copilot.config.settings.resolve_active_generation", return_value="gen1"),
            patch("data_engineering_copilot.evaluation.provenance.answer_config_fingerprint", return_value="fp1"),
        ):
            mock_settings.active_embedding_model_name.return_value = "model1"
            mock_settings.active_collection_name = "col1"
            mock_settings.collection_name = "col1"

            result = _build_cache_scope(mock_request, ["src1"])

        assert result.tenant_id == "tenant1"
        assert result.role == "admin"
        assert result.source_filter == ("src1",)


class TestExtractUserSession:
    def test_extracts_from_headers(self) -> None:
        mock_request = MagicMock()
        mock_request.headers.get.side_effect = lambda key: {"X-User-ID": "user1", "X-Session-ID": "sess1"}.get(key)
        mock_request.query_params.get.return_value = None

        user_id, session_id = _extract_user_session(mock_request)
        assert user_id == "user1"
        assert session_id == "sess1"

    def test_falls_back_to_query_params(self) -> None:
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None
        mock_request.query_params.get.side_effect = lambda key: {"user_id": "user1", "session_id": "sess1"}.get(key)

        user_id, session_id = _extract_user_session(mock_request)
        assert user_id == "user1"
        assert session_id == "sess1"

    def test_returns_none_when_missing(self) -> None:
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None
        mock_request.query_params.get.return_value = None

        user_id, session_id = _extract_user_session(mock_request)
        assert user_id is None
        assert session_id is None
