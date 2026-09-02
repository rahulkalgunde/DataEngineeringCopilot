"""Test _upsert_with_retry retries with exponential backoff."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore


@pytest.mark.asyncio
async def test_upsert_with_retry_retries_3():
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test", hybrid_search=False)
    store._client = AsyncMock()
    store._client.upsert.side_effect = [Exception("yellow"), Exception("yellow"), None]
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await store._upsert_with_retry("test", points=AsyncMock(), max_retries=3)
        assert store._client.upsert.call_count == 3
        # sleep called twice for first two failures with 30*2^attempt
        assert mock_sleep.await_count == 2
        assert mock_sleep.await_args_list[0].args[0] == 30
        assert mock_sleep.await_args_list[1].args[0] == 60


@pytest.mark.asyncio
async def test_upsert_with_retry_succeeds_first_try():
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test", hybrid_search=False)
    store._client = AsyncMock()
    store._client.upsert.return_value = None
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await store._upsert_with_retry("test", points=AsyncMock(), max_retries=3)
        assert store._client.upsert.call_count == 1
        mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_with_retry_raises_after_exhausted():
    store = AsyncQdrantVectorStore(url="http://localhost:6333", collection_name="test", hybrid_search=False)
    store._client = AsyncMock()
    store._client.upsert.side_effect = Exception("persistent yellow")
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(Exception, match="persistent yellow"):
            await store._upsert_with_retry("test", points=AsyncMock(), max_retries=3)
        assert store._client.upsert.call_count == 3
