"""Tests for CircuitBreaker fail-fast mechanism."""

from __future__ import annotations

import asyncio

import pytest

from data_engineering_copilot.infrastructure.llm_client import CircuitBreaker, CircuitBreakerError


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_successful_call_returns_result(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        result = await cb.call(lambda: _async_ok("hello"))
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_successful_call_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)
        with pytest.raises(ValueError):
            await cb.call(lambda: _async_fail(ValueError("first")))
        with pytest.raises(ValueError):
            await cb.call(lambda: _async_fail(ValueError("second")))
        assert cb._state == "open"
        cb._last_failure_time = 0  # force recovery
        cb._state = "half-open"
        result = await cb.call(lambda: _async_ok("recovered"))
        assert result == "recovered"
        assert cb._state == "closed"
        assert cb._failures == 0

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)
        with pytest.raises(ValueError):
            await cb.call(lambda: _async_fail(ValueError("fail 1")))
        assert cb._state == "closed"
        with pytest.raises(ValueError):
            await cb.call(lambda: _async_fail(ValueError("fail 2")))
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_rejects_in_open_state(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)
        cb._state = "open"
        cb._failures = 2
        cb._last_failure_time = 9999999999.0  # far in the future
        with pytest.raises(CircuitBreakerError, match="(?i)circuit breaker open"):
            await cb.call(lambda: _async_ok("should not reach"))

    @pytest.mark.asyncio
    async def test_half_open_allows_probe_after_recovery(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        with pytest.raises(ValueError):
            await cb.call(lambda: _async_fail(ValueError("fail 1")))
        with pytest.raises(ValueError):
            await cb.call(lambda: _async_fail(ValueError("fail 2")))
        assert cb._state == "open"
        await asyncio.sleep(0.02)
        result = await cb.call(lambda: _async_ok("probe"))
        assert result == "probe"
        assert cb._state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb._state = "half-open"
        with pytest.raises(ValueError):
            await cb.call(lambda: _async_fail(ValueError("probe fail")))
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_any_exception_counts_as_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)
        with pytest.raises(RuntimeError):
            await cb.call(lambda: _async_fail(RuntimeError("any error")))
        assert cb._state == "open"
        with pytest.raises(CircuitBreakerError):
            await cb.call(lambda: _async_ok("should not reach"))

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_thread_safe(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

        async def concurrent_fail():
            with pytest.raises(ValueError):
                await cb.call(lambda: _async_fail(ValueError("concurrent")))

        await asyncio.gather(concurrent_fail(), concurrent_fail(), concurrent_fail())
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_not_serialized(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        gate = asyncio.Event()
        started = 0

        async def slow_ok():
            nonlocal started
            started += 1
            await gate.wait()
            return "ok"

        # Fire 5 calls concurrently. If calls were serialized by a lock held
        # across the coroutine execution, only one would start before the gate
        # is released.
        tasks = [asyncio.create_task(cb.call(slow_ok)) for _ in range(5)]
        await asyncio.sleep(0.05)
        assert started == 5
        gate.set()
        results = await asyncio.gather(*tasks)
        assert results == ["ok"] * 5

    @pytest.mark.asyncio
    async def test_half_open_allows_single_probe_only(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb._state = "half-open"

        async def slow_probe():
            await asyncio.sleep(0.02)
            return "ok"

        # First caller claims the probe slot.
        probe_task = asyncio.create_task(cb.call(slow_probe))
        await asyncio.sleep(0.01)
        # Concurrent caller while probing must be rejected, not fire a second probe.
        with pytest.raises(CircuitBreakerError):
            await cb.call(slow_probe)
        assert await probe_task == "ok"
        assert cb._state == "closed"


async def _async_ok(value: str = "ok") -> str:
    return value


async def _async_fail(exc: Exception) -> str:
    raise exc
