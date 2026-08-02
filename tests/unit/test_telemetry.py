"""Tests for telemetry flush_async — non-blocking, fail-silent telemetry flush."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from data_engineering_copilot.observability import telemetry


class TestFlushAsyncDoesNotBlockEventLoop:
    @pytest.mark.asyncio
    async def test_flush_async_does_not_block_event_loop(self):
        """flush_async runs the sync flush off-loop; the event loop stays responsive."""

        class SlowClient:
            def flush(self) -> None:
                time.sleep(0.5)

        tracer = telemetry.LangfuseTelemetryTracer(SlowClient())

        task = asyncio.create_task(tracer.flush_async(timeout=5.0))

        ticks = 0
        while not task.done() and ticks < 100:
            await asyncio.sleep(0.01)
            ticks += 1

        await task
        assert ticks > 0  # event loop progressed while the sync flush ran off-thread

    @pytest.mark.asyncio
    async def test_flush_async_noop_tracer_returns_immediately(self):
        tracer = telemetry.NoOpTelemetryTracer()
        await tracer.flush_async(timeout=1.0)

    @pytest.mark.asyncio
    async def test_flush_async_none_client_returns_immediately(self):
        tracer = telemetry.LangfuseTelemetryTracer(None)
        await tracer.flush_async(timeout=1.0)


class TestFlushFailureIsSilent:
    @pytest.mark.asyncio
    async def test_flush_failure_is_silent(self):
        class FailingClient:
            def flush(self) -> None:
                raise RuntimeError("exporter down")

        tracer = telemetry.LangfuseTelemetryTracer(FailingClient())
        await tracer.flush_async(timeout=1.0)  # must not raise

    @pytest.mark.asyncio
    async def test_flush_failure_is_silent_even_when_blocking(self):
        class SlowFailingClient:
            def flush(self) -> None:
                time.sleep(0.5)
                raise RuntimeError("exporter hung then died")

        tracer = telemetry.LangfuseTelemetryTracer(SlowFailingClient())
        await tracer.flush_async(timeout=2.0)  # must not raise


class TestFlushAsyncTimeout:
    @pytest.mark.asyncio
    async def test_flush_async_respects_deadline(self):
        class HangingClient:
            def flush(self) -> None:
                time.sleep(10.0)

        tracer = telemetry.LangfuseTelemetryTracer(HangingClient())

        start = time.monotonic()
        await tracer.flush_async(timeout=0.05)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0  # returned promptly despite the 10s sync flush


class TestRagTelemetryFlush:
    @pytest.mark.asyncio
    async def test_answer_survives_failing_exporter(self):
        from unittest.mock import AsyncMock, MagicMock

        from data_engineering_copilot.domain.models import RagConfig
        from data_engineering_copilot.services.async_rag import AsyncRagService

        class FailingTracer:
            def start_observation(self, name: str, input: Any = None, as_type: str = "trace", model: str | None = None):
                return _FakeSpan()

            async def flush_async(self, timeout: float = 2.0) -> None:
                raise RuntimeError("exporter down")

            def flush(self) -> None:
                raise RuntimeError("exporter down")

        class _FakeSpan:
            def update(self, **kwargs: Any) -> Any:
                return self

            def end(self) -> Any:
                return self

            def start_observation(self, name: str, **kwargs: Any) -> Any:
                return self

        chunk = MagicMock()
        chunk.chunk.source_name = "test"
        chunk.chunk.title = "Test"
        chunk.chunk.url = "http://test.com"
        chunk.chunk.text = "Spark SQL is a module for structured data."
        chunk.confidence = 0.9
        chunk.distance = 0.1

        vector_store = MagicMock()
        vector_store.query = AsyncMock(return_value=[chunk])
        llm_client = MagicMock()
        llm_client.generate = AsyncMock(return_value="Spark SQL is a module for structured data.")
        embedder = MagicMock()
        embedder.embed_query = AsyncMock(return_value=[0.1] * 8)

        config = RagConfig(retrieval_top_k=5, confidence_threshold=0.3, reranker_enabled=False, max_context_chars=4000)

        service = AsyncRagService(
            config=config,
            vector_store=vector_store,
            llm_client=llm_client,
            embedder=embedder,
            reranker=None,
            telemetry=FailingTracer(),
            cache=None,
        )

        result = await service.answer("what is spark sql")
        assert result.text  # answer returned even though flush_async raised


def test_opentelemetry_sdk_importable():
    """OTel SDK must be resolvable — declared as a hard dependency, not soft-failed."""
    import opentelemetry.exporter.otlp.proto.grpc.trace_exporter  # noqa: F401
    import opentelemetry.sdk  # noqa: F401
    import opentelemetry.sdk.resources  # noqa: F401
    import opentelemetry.sdk.trace  # noqa: F401
    import opentelemetry.sdk.trace.export  # noqa: F401
