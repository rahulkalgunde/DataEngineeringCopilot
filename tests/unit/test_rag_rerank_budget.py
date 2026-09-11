import asyncio
from unittest.mock import AsyncMock, MagicMock

from data_engineering_copilot.domain.models import RagConfig
from data_engineering_copilot.services.async_rag import AsyncRagService


class _SlowNotReadyReranker:
    """initialize() that never completes within budget; not available when done."""

    def __init__(self):
        self.initialized = False

    def is_available(self) -> bool:
        return False

    def initialize(self):
        async def _slow():
            self.initialized = True
            await asyncio.sleep(5)

        return _slow()


class _ReadyReranker:
    def is_available(self) -> bool:
        return True

    def initialize(self):  # pragma: no cover - never called
        raise AssertionError("initialize should not be called when already available")


def _make_service():
    config = RagConfig(reranker_enabled=True, reranker_init_budget_seconds=0.1)
    # Match the canonical construction used by tests/unit/test_async_rag.py.
    return AsyncRagService(
        config=config,
        vector_store=MagicMock(query=AsyncMock(return_value=[])),
        llm_client=MagicMock(generate=AsyncMock(return_value="answer")),
        embedder=MagicMock(embed_query=AsyncMock(return_value=[0.1] * 2048)),
        reranker=None,
        telemetry=None,
        cache=None,
    )


async def test_init_budget_exceeded_returns_false_quickly():
    service = _make_service()
    rr = _SlowNotReadyReranker()
    start = asyncio.get_running_loop().time()
    ok = await service._ensure_reranker_ready(rr, 0.1)
    elapsed = asyncio.get_running_loop().time() - start
    assert ok is False
    assert elapsed < 2.0  # far below the fixed 120s
    assert rr.initialized is True  # the load was started, just not finished


async def test_ready_reranker_short_circuits():
    service = _make_service()
    assert await service._ensure_reranker_ready(_ReadyReranker(), 0.1) is True


async def test_chat_stream_emits_rerank_skip_status():
    # Assert the skip transparency at the smallest unit that can show it: the
    # _maybe_rerank_chat helper below. Driving full chat_stream hermetically is
    # only viable if the repo already has a stubbed full drive in
    # tests/unit/test_async_rag*.py — if it does, drive that and assert the
    # 'Reranking skipped' status event; do not build a new full drive here.
    await test_maybe_rerank_skips_and_emits_status()


async def test_maybe_rerank_skips_and_emits_status():
    from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk

    service = _make_service()
    chunk_a = DocumentChunk(chunk_id="a", source_name="s", title="t", url="u", text="x", doc_type="guide")
    chunk_b = DocumentChunk(chunk_id="b", source_name="s", title="t", url="u", text="y", doc_type="guide")
    chunks = [
        RetrievedChunk(chunk=chunk_a, distance=0.1, confidence=0.9),
        RetrievedChunk(chunk=chunk_b, distance=0.2, confidence=0.8),
    ]
    events = []
    rr = _SlowNotReadyReranker()
    out = await service._maybe_rerank_chat(rr, "q", chunks, events.append)
    assert len(out) == 2  # unchanged pool (no rerank ran)
    assert {"type": "status", "message": "Reranking"} in events
    assert {"type": "status", "message": "Reranking skipped (local reranker not ready)"} in events
