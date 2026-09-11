import asyncio
import logging
import sys
import types

import pytest

from data_engineering_copilot.services.reranker import CrossEncoderReranker


class _SlowLoader:
    """Doubles sentence_transformers.CrossEncoder; completes after 0.15s."""

    def model_name(self):
        return "fake-reranker"


async def test_two_concurrent_initializes_load_once():
    loads = 0

    async def fake_load(self):
        nonlocal loads
        loads += 1
        await asyncio.sleep(0.15)
        self.model = object()

    rr = CrossEncoderReranker(model_name="fake-reranker")
    rr._load_model = fake_load.__get__(rr, CrossEncoderReranker)

    a = asyncio.create_task(rr.initialize())
    b = asyncio.create_task(rr.initialize())
    await asyncio.gather(a, b)

    assert rr.model is not None
    assert loads == 1


async def test_cancelled_waiter_does_not_kill_background_load():
    rr = CrossEncoderReranker(model_name="fake-reranker")

    async def fake_load(self):
        await asyncio.sleep(0.2)
        self.model = object()

    rr._load_model = fake_load.__get__(rr, CrossEncoderReranker)

    waiter = asyncio.create_task(rr.initialize())
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    await asyncio.sleep(0.25)  # background load still completes
    assert rr.model is not None


async def test_transient_load_failure_retries(monkeypatch, caplog) -> None:
    """A failed load clears the init handle so the next request retries."""
    trials: list[int] = []

    class _FlakyCrossEncoder:
        def __init__(self, *args, **kwargs) -> None:
            trials.append(1)
            if len(trials) == 1:
                raise OSError("transient model download failure")
            self.loaded = True

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.CrossEncoder = _FlakyCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    rr = CrossEncoderReranker(model_name="fake-reranker")
    with caplog.at_level(logging.WARNING):
        await rr.initialize()
    assert rr.model is None
    assert len(trials) == 1

    with caplog.at_level(logging.WARNING):
        await rr.initialize()
    assert rr.model is not None
    assert rr.model.loaded
    assert len(trials) == 2
