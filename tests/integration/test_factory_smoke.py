"""Integration tests for factory smoke tests."""

from __future__ import annotations

import pytest

from data_engineering_copilot.config.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


async def test_build_async_ingestion_service_creates_collection(redis_url):
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url=settings.qdrant_url,
        collection_name="test_factory_smoke",
        embedding_dim=settings.embedding_dim,
        hybrid_search=settings.hybrid_search_enabled,
    )

    await store.initialize()

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    assert "test_factory_smoke" in names

    client.delete_collection("test_factory_smoke")


async def test_build_rag_service_returns_valid_service(redis_url):
    from data_engineering_copilot.factory import build_rag_service
    from data_engineering_copilot.services.async_rag import AsyncRagService

    rag_service = build_rag_service()
    assert isinstance(rag_service, AsyncRagService)
