"""Singleton wrapper for AsyncRagService to avoid per-request instantiation.

Every /ask request previously called build_rag_service(), creating fresh
LLM clients, embedders, vector stores, and reranker models. This module
provides a thread-safe singleton that reuses the same service instance.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_engineering_copilot.services.async_rag import AsyncRagService

_instance: AsyncRagService | None = None
_lock = threading.Lock()


def get_rag_service() -> AsyncRagService:
    """Get or create a singleton AsyncRagService instance.

    Thread-safe via threading.Lock. The instance is created once
    and reused across all requests, avoiding per-request model loading.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                from data_engineering_copilot.factory import build_rag_service

                _instance = build_rag_service()
    return _instance


def reset_rag_service() -> None:
    """Reset the singleton (for testing or config changes)."""
    global _instance
    with _lock:
        _instance = None
