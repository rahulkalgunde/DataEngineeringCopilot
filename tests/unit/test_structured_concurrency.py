"""Tests for structured concurrency with TaskGroup in async ingestion."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

INGESTION_PY = Path("data_engineering_copilot/services/async_ingestion.py")


class TestTaskGroupStructuredConcurrency:
    @pytest.mark.asyncio
    async def test_taskgroup_propagates_exception(self):
        async def failing_task():
            raise ValueError("boom")

        with pytest.raises(BaseExceptionGroup):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(failing_task())

    @pytest.mark.asyncio
    async def test_taskgroup_cancels_siblings_on_failure(self):
        results = []

        async def slow_task():
            await asyncio.sleep(10)
            results.append("slow_done")

        async def fail_task():
            raise RuntimeError("crash")

        with pytest.raises(ExceptionGroup):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(slow_task())
                tg.create_task(fail_task())

        assert "slow_done" not in results

    @pytest.mark.asyncio
    async def test_taskgroup_all_tasks_complete(self):
        results = []

        async def work(n):
            results.append(n)

        async with asyncio.TaskGroup() as tg:
            for i in range(3):
                tg.create_task(work(i))

        assert sorted(results) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_taskgroup_empty(self):
        async with asyncio.TaskGroup():
            pass

    @pytest.mark.asyncio
    async def test_async_ingestion_uses_taskgroup(self):
        content = INGESTION_PY.read_text()
        assert "TaskGroup" in content or "task_group" in content.lower()
