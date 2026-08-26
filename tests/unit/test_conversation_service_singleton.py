"""Tests for services/conversation_service_singleton.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset the singleton before and after each test."""
    import data_engineering_copilot.services.conversation_service_singleton as mod

    mod.reset_conversation_service()
    yield
    mod.reset_conversation_service()


class TestGetConversationServiceIfInitialized:
    def test_returns_none_when_not_initialized(self) -> None:
        from data_engineering_copilot.services.conversation_service_singleton import (
            get_conversation_service_if_initialized,
        )

        assert get_conversation_service_if_initialized() is None

    def test_returns_instance_after_get(self) -> None:
        from data_engineering_copilot.services.conversation_service_singleton import (
            get_conversation_service,
            get_conversation_service_if_initialized,
        )

        mock_service = AsyncMock()
        with patch(
            "data_engineering_copilot.factory.build_conversation_service",
            return_value=mock_service,
        ):
            import asyncio

            asyncio.run(get_conversation_service())

        assert get_conversation_service_if_initialized() is mock_service


class TestResetConversationService:
    def test_resets_instance(self) -> None:
        from data_engineering_copilot.services.conversation_service_singleton import (
            get_conversation_service_if_initialized,
            reset_conversation_service,
        )

        mock_service = AsyncMock()
        import data_engineering_copilot.services.conversation_service_singleton as mod

        mod._instance = mock_service
        assert get_conversation_service_if_initialized() is mock_service

        reset_conversation_service()
        assert get_conversation_service_if_initialized() is None
