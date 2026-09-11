import asyncio

import pytest

from data_engineering_copilot.api.stream_budget import generator_with_global_budget


async def test_fast_generator_passes_through():
    async def gen():
        yield 1
        yield 2

    out = []
    async for item in generator_with_global_budget(gen(), 5.0):
        out.append(item)
    assert out == [1, 2]


async def test_stalled_generator_aborted_on_budget():
    aborted = False

    async def gen():
        nonlocal aborted
        try:
            yield "a"
            await asyncio.sleep(60)
            yield "b"
        finally:
            aborted = True

    agen = gen()
    out = []
    with pytest.raises(asyncio.TimeoutError):
        async for item in generator_with_global_budget(agen, 0.1):
            out.append(item)
    assert out == ["a"]
    assert aborted is True


async def test_slow_but_within_budget_completes():
    async def gen():
        yield 1
        await asyncio.sleep(0.05)
        yield 2

    out = []
    async for item in generator_with_global_budget(gen(), 1.0):
        out.append(item)
    assert out == [1, 2]
