from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.graph_store import GraphStore
from data_engineering_copilot.services.graph_extractor import GraphExtractor
from data_engineering_copilot.services.graph_traversal import GraphTraversalService


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return ""

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        yield await self.generate(prompt)

    @property
    def last_usage(self) -> LLMUsage:
        return LLMUsage()


def test_graph_store_basic() -> None:
    store = GraphStore(":memory:")

    node_id = store.add_node("Spark", "framework")
    assert node_id > 0

    # Test duplicate node addition doesn't error and returns same ID
    node_id_2 = store.add_node("Spark", "framework")
    assert node_id == node_id_2

    store.add_edge("Spark", "Celery", "uses")

    neighbors = store.get_neighbors("Spark")
    assert len(neighbors) == 1
    assert neighbors[0] == ("spark", "uses", "celery")


@pytest.mark.asyncio
async def test_graph_extractor() -> None:
    store = GraphStore(":memory:")
    llm_response = """
    [
      {"source": "Spark Streaming", "target": "YARN", "relation": "runs_on"}
    ]
    """
    llm = FakeLLM([llm_response])
    extractor = GraphExtractor(llm, store)

    await extractor.extract_and_store("Spark Streaming runs on YARN.")

    neighbors = store.get_neighbors("Spark Streaming")
    assert len(neighbors) == 1
    assert neighbors[0] == ("spark streaming", "runs_on", "yarn")


@pytest.mark.asyncio
async def test_graph_traversal() -> None:
    store = GraphStore(":memory:")
    store.add_edge("Spark", "YARN", "runs_on")

    llm_response = '["spark"]'
    llm = FakeLLM([llm_response])
    traversal = GraphTraversalService(llm, store)

    topo_ctx = await traversal.get_topological_context("How to run Spark?")
    assert "Topological & Entity Relationships found in Knowledge Graph:" in topo_ctx
    assert "(- (spark) --[runs_on]--> (yarn)" in topo_ctx or "(spark) --[runs_on]--> (yarn)" in topo_ctx
