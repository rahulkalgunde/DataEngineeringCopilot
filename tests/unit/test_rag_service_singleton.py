"""Tests for services/rag_service_singleton.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset the singleton before and after each test."""
    import data_engineering_copilot.services.rag_service_singleton as mod

    mod.reset_rag_service()
    yield
    mod.reset_rag_service()


class TestGetRagServiceIfInitialized:
    def test_returns_none_when_not_initialized(self) -> None:
        from data_engineering_copilot.services.rag_service_singleton import (
            get_rag_service_if_initialized,
        )

        assert get_rag_service_if_initialized() is None

    def test_returns_instance_after_get(self) -> None:
        from data_engineering_copilot.services.rag_service_singleton import (
            get_rag_service,
            get_rag_service_if_initialized,
        )

        mock_service = AsyncMock()
        mock_service.reranker = None
        with patch(
            "data_engineering_copilot.factory.build_rag_service",
            return_value=mock_service,
        ):
            import asyncio

            asyncio.run(get_rag_service())

        assert get_rag_service_if_initialized() is mock_service


class TestResetRagService:
    def test_resets_instance(self) -> None:
        from data_engineering_copilot.services.rag_service_singleton import (
            get_rag_service_if_initialized,
            reset_rag_service,
        )

        mock_service = AsyncMock()
        import data_engineering_copilot.services.rag_service_singleton as mod

        mod._instance = mock_service
        assert get_rag_service_if_initialized() is mock_service

        reset_rag_service()
        assert get_rag_service_if_initialized() is None
