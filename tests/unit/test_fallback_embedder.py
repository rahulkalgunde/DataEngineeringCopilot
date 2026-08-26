"""Tests for FallbackEmbedder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.infrastructure.fallback_embedder import FallbackEmbedder


def _make_mock_chain(has_execute: bool = True) -> MagicMock:
    mock = MagicMock(spec=["execute", "close"] if has_execute else ["embed_texts", "close"])
    if has_execute:
        mock.execute = AsyncMock(return_value=[[1.0] * 2048])
    else:
        mock.embed_texts = AsyncMock(return_value=[[1.0] * 2048])
    mock.close = AsyncMock()
    return mock


class TestFallbackEmbedder:
    @pytest.mark.asyncio
    async def test_embed_texts_with_execute(self) -> None:
        chain = _make_mock_chain(has_execute=True)
        emb = FallbackEmbedder(chain)
        result = await emb.embed_texts(["hello"])
        assert len(result) == 1
        chain.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_without_execute(self) -> None:
        chain = _make_mock_chain(has_execute=False)
        emb = FallbackEmbedder(chain)
        result = await emb.embed_texts(["hello"])
        assert len(result) == 1
        chain.embed_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_query_with_execute(self) -> None:
        chain = _make_mock_chain(has_execute=True)
        chain.execute.return_value = [[2.0] * 2048]
        emb = FallbackEmbedder(chain)
        result = await emb.embed_query("test")
        assert len(result) == 2048
        assert result[0] == 2.0

    @pytest.mark.asyncio
    async def test_embed_query_without_execute(self) -> None:
        chain = _make_mock_chain(has_execute=False)
        chain.embed_texts.return_value = [[3.0] * 2048]
        emb = FallbackEmbedder(chain)
        result = await emb.embed_query("test")
        assert len(result) == 2048

    @pytest.mark.asyncio
    async def test_embed_query_empty_raises(self) -> None:
        chain = _make_mock_chain(has_execute=True)
        chain.execute.return_value = []
        emb = FallbackEmbedder(chain)
        with pytest.raises(ValueError, match="no embedding"):
            await emb.embed_query("test")

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        chain = _make_mock_chain(has_execute=True)
        emb = FallbackEmbedder(chain)
        await emb.close()
        chain.close.assert_called_once()

    def test_inner_property(self) -> None:
        chain = _make_mock_chain(has_execute=True)
        emb = FallbackEmbedder(chain)
        assert emb.inner is chain
