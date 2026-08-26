"""Tests for GraphExtractor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.services.graph_extractor import GraphExtractor


class TestGraphExtractor:
    @pytest.mark.asyncio
    async def test_extract_and_store_parses_json(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = '[{"source": "A", "target": "B", "relation": "uses"}]'
        store = MagicMock()
        ext = GraphExtractor(llm, store)
        await ext.extract_and_store("test text")
        store.add_edge.assert_called_once_with("A", "B", "uses")

    @pytest.mark.asyncio
    async def test_extract_and_store_handles_markdown_fence(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = '```json\n[{"source": "X", "target": "Y", "relation": "r"}]\n```'
        store = MagicMock()
        ext = GraphExtractor(llm, store)
        await ext.extract_and_store("text")
        store.add_edge.assert_called_once_with("X", "Y", "r")

    @pytest.mark.asyncio
    async def test_extract_and_store_skips_invalid(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "not json"
        store = MagicMock()
        ext = GraphExtractor(llm, store)
        await ext.extract_and_store("text")
        store.add_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_and_store_skips_incomplete(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = '[{"source": "A", "target": "B"}]'
        store = MagicMock()
        ext = GraphExtractor(llm, store)
        await ext.extract_and_store("text")
        store.add_edge.assert_not_called()
