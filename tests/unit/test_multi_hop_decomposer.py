from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, LLMUsage, RetrievedChunk
from data_engineering_copilot.services.multi_hop_decomposer import MultiHopDecomposer, QueryStep


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


@pytest.mark.asyncio
async def test_plan_query_multi_hop() -> None:
    llm_response = """
    ```json
    {
      "is_multi_hop": true,
      "steps": [
        {
          "step_id": 1,
          "query": "Spark 3.2 streaming ingestion",
          "depends_on": []
        },
        {
          "step_id": 2,
          "query": "Custom migration framework",
          "depends_on": [1]
        }
      ]
    }
    ```
    """
    llm = FakeLLM([llm_response])
    decomposer = MultiHopDecomposer(llm)

    plan = await decomposer.plan_query("Compare Spark 3.2 vs Custom migration framework")
    assert plan.is_multi_hop is True
    assert len(plan.steps) == 2
    assert plan.steps[0].step_id == 1
    assert plan.steps[0].query == "Spark 3.2 streaming ingestion"
    assert plan.steps[1].depends_on == [1]


@pytest.mark.asyncio
async def test_plan_query_single_hop() -> None:
    llm_response = """
    {
      "is_multi_hop": false,
      "steps": []
    }
    """
    llm = FakeLLM([llm_response])
    decomposer = MultiHopDecomposer(llm)

    plan = await decomposer.plan_query("What is Spark?")
    assert plan.is_multi_hop is False
    assert len(plan.steps) == 0


@pytest.mark.asyncio
async def test_execute_step() -> None:
    llm = FakeLLM(["Refined custom framework query", "Summary of results"])
    decomposer = MultiHopDecomposer(llm)

    # Mock RAG Service
    rag_service = MagicMock()
    rag_service.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
    rag_service._rrf_profile_for = MagicMock(return_value="equal")

    # Mock retrieved chunks
    chunk = DocumentChunk(
        chunk_id="c1",
        content_hash="h1",
        url="http://spark.org",
        title="Spark Docs",
        text="Spark 3.2 handles streaming with micro-batches.",
        source_name="spark",
        word_count=10,
        chunk_index=0,
        total_chunks=1,
    )
    retrieved = RetrievedChunk(chunk=chunk, distance=0.1, confidence=0.9)
    rag_service.vector_store.query = AsyncMock(return_value=[retrieved])

    step = QueryStep(step_id=2, query="Custom framework query", depends_on=[1])
    previous_results = {1: "Spark 3.2 handles streaming with micro-batches."}

    summary = await decomposer.execute_step(step, previous_results, rag_service)

    assert summary == "Summary of results"
    # Check that query was refined because it depends on step 1
    assert "Given the previous steps context" in llm.calls[0]
    # Check summary generation call
    assert "Provide a concise summary answering this sub-query" in llm.calls[1]
