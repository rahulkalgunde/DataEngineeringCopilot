import asyncio
from unittest.mock import AsyncMock, MagicMock

from data_engineering_copilot.domain.models import RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService


class _Unavailable:
    def is_available(self):
        return False

    def initialize(self):
        async def _load():
            await asyncio.sleep(0.05)

        return _load()


class _Boom:
    def is_available(self):
        return False

    def initialize(self):
        async def _bad():
            raise RuntimeError("download failed")

        return _bad()


def _make_service():
    # Match the canonical construction used by tests/unit/test_async_rag.py
    # (same pattern as tests/unit/test_rag_rerank_budget.py).
    return AsyncRagService(
        config=RagConfig(),
        vector_store=MagicMock(query=AsyncMock(return_value=[])),
        llm_client=MagicMock(generate=AsyncMock(return_value="answer")),
        embedder=MagicMock(embed_query=AsyncMock(return_value=[0.1] * 2048)),
        reranker=None,
        telemetry=None,
        cache=None,
    )


async def test_warmup_task_started_and_completes():
    service = _make_service()
    service.reranker = _Unavailable()
    from data_engineering_copilot.factory import _warmup_reranker

    service.reranker_warmup_task = asyncio.create_task(_warmup_reranker(service.reranker, budget_seconds=1.0))
    assert service.reranker_warmup_task is not None
    await service.reranker_warmup_task
    assert service.reranker_warmup_task.done()
    assert not service.reranker_warmup_task.cancelled()


async def test_warmup_fails_open_when_init_errors():
    from data_engineering_copilot.factory import _warmup_reranker

    await _warmup_reranker(_Boom(), budget_seconds=1.0)  # must NOT raise
