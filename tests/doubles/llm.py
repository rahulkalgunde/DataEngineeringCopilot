"""Deterministic LLM doubles for pipeline-logic tests (no network, no infra).

Reuse these instead of hand-rolling ``MagicMock(spec=LLMClientProtocol)`` or
local stub classes in every test module.  Behavior is explicit and stable, so
assertions on answer structure, citations, confidence, and call counts are
hermetic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.domain.protocols import LLMClientProtocol

STUB_ANSWER = (
    "Apache Spark is a unified analytics engine for large-scale data processing. "
    "It provides high-level APIs in Scala, Java, Python, and R. Spark SQL enables "
    "structured data processing with DataFrames, and Delta Lake brings ACID "
    "transactions to big data workloads. PySpark is the Python API for Spark."
)

STUB_GAP_ANSWER = (
    "I cannot answer this question based on the provided documentation, which covers "
    "only Apache Spark, Delta Lake, and Airflow topics."
)


async def _token_stream(text: str) -> AsyncIterator[str]:
    for token in text.split(" "):
        yield f"{token} "


class StubLLM(LLMClientProtocol):
    """Deterministic LLM: returns a fixed answer, or a gap-acknowledgement
    when the prompt contains *gap_trigger*.

    Tracks ``call_count`` so tests can assert on how many LLM round-trips a
    pipeline performed (e.g. cache-hit tests).
    """

    def __init__(self, answer: str = STUB_ANSWER, gap_trigger: str = "capital of France") -> None:
        self.answer = answer
        self.gap_trigger = gap_trigger
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs: object) -> str:
        self.call_count += 1
        if self.gap_trigger and self.gap_trigger in prompt:
            return STUB_GAP_ANSWER
        return self.answer

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        return _token_stream(await self.generate(prompt))

    @property
    def last_usage(self) -> LLMUsage:
        return LLMUsage(prompt_tokens=0, completion_tokens=0, model="stub")

    async def close(self) -> None:
        pass


class StaticLLM(LLMClientProtocol):
    """Returns a fixed answer on every call and counts invocations.

    Use when a test only cares about *whether* the LLM was called (and how
    many times), not about varying responses.
    """

    def __init__(self, answer: str = STUB_ANSWER) -> None:
        self.answer = answer
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs: object) -> str:
        self.call_count += 1
        return self.answer

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        return _token_stream(await self.generate(prompt))

    @property
    def last_usage(self) -> LLMUsage:
        return LLMUsage(prompt_tokens=0, completion_tokens=0, model="static")

    async def close(self) -> None:
        pass
