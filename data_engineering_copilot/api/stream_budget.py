"""Route-layer wall-clock budget for streaming SSE generators."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator


async def generator_with_global_budget[T](
    agen: AsyncGenerator[T, None], total_budget_seconds: float
) -> AsyncGenerator[T, None]:
    """Yield ``agen`` items, aborting the generator on wall-clock budget breach.

    The deadline is computed once at start and checked before every fetch;
    a single stalled ``__anext__`` is additionally bounded by ``wait_for``.
    On breach the underlying generator is cancelled and closed, then
    ``asyncio.TimeoutError`` is raised.
    """
    if total_budget_seconds is None or total_budget_seconds <= 0:
        async for item in agen:
            yield item
        return

    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_budget_seconds
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            await _abort(agen)
            raise TimeoutError("streaming budget exceeded")
        try:
            item = await asyncio.wait_for(agen.__anext__(), timeout=remaining)
        except TimeoutError:
            await _abort(agen)
            raise
        except StopAsyncIteration:
            return
        yield item


async def _abort[T](agen: AsyncGenerator[T, None]) -> None:
    with contextlib.suppress(Exception):  # aclose may raise if the generator was mid-await
        await agen.aclose()
