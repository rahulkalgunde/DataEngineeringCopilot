"""Singleton wrapper for ConversationService.

Mirrors ``rag_service_singleton.py``: a thread-safe lazily-built singleton so
the chat store's Postgres pool and Redis client are shared across requests
instead of re-created per call.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_engineering_copilot.services.conversation_rag import ConversationService

_instance: ConversationService | None = None
_lock = threading.Lock()


async def get_conversation_service() -> ConversationService:
    """Get or create the singleton ConversationService instance."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                from data_engineering_copilot.config.settings import settings
                from data_engineering_copilot.factory import build_conversation_service

                _instance = build_conversation_service(app_settings=settings)
    return _instance


def get_conversation_service_if_initialized() -> ConversationService | None:
    """Return the singleton if created, else None (safe during shutdown)."""
    return _instance


def reset_conversation_service() -> None:
    """Reset the singleton (for testing or config changes)."""
    global _instance
    with _lock:
        _instance = None
