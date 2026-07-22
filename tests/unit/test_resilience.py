"""Tests for circuit breaker and bulkhead resilience patterns."""

from __future__ import annotations

import time

import pytest

from data_engineering_copilot.infrastructure.resilience import (
    Bulkhead,
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerState,
    bulkhead,
    circuit_breaker,
)


class TestCircuitBreakerStateMachine:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_closed_allows_calls(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert cb.can_execute() is True

    def test_increment_failure_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 2

    def test_threshold_reached_opens_circuit(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 3

    def test_open_rejects_calls(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.can_execute() is False

    def test_open_raises_on_execute(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        for _ in range(3):
            cb.record_failure()
        with pytest.raises(CircuitBreakerOpen):
            cb.execute()

    def test_recovery_timeout_transitions_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.can_execute() is True

    def test_half_open_allows_single_call(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.can_execute() is True

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerDecorator:
    def test_decorator_allows_successful_calls(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        @circuit_breaker(cb)
        def succeed():
            return "ok"

        assert succeed() == "ok"

    def test_decorator_records_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        @circuit_breaker(cb)
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()
        assert cb.failure_count == 1

    def test_decorator_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)

        @circuit_breaker(cb)
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fail()
        with pytest.raises(ValueError):
            fail()
        assert cb.state == CircuitBreakerState.OPEN

        with pytest.raises(CircuitBreakerOpen):
            fail()

    def test_decorator_recovers_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        call_count = 0

        @circuit_breaker(cb)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("boom")
            return "recovered"

        with pytest.raises(ValueError):
            flaky()
        with pytest.raises(ValueError):
            flaky()
        assert cb.state == CircuitBreakerState.OPEN

        time.sleep(0.15)
        result = flaky()
        assert result == "recovered"
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_decorator_with_async_function(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        @circuit_breaker(cb)
        async def succeed():
            return "async_ok"

        result = await succeed()
        assert result == "async_ok"

    @pytest.mark.asyncio
    async def test_decorator_records_async_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        @circuit_breaker(cb)
        async def fail():
            raise ValueError("async_boom")

        with pytest.raises(ValueError, match="async_boom"):
            await fail()
        assert cb.failure_count == 1


class TestBulkhead:
    def test_allows_concurrent_up_to_limit(self):
        bh = Bulkhead(max_concurrency=2)
        assert bh.acquire(blocking=False) is True
        assert bh.acquire(blocking=False) is True

    def test_rejects_when_at_limit(self):
        bh = Bulkhead(max_concurrency=2)
        bh.acquire(blocking=False)
        bh.acquire(blocking=False)
        with pytest.raises(Bulkhead.Full):
            bh.acquire(blocking=False)

    def test_release_allows_new_call(self):
        bh = Bulkhead(max_concurrency=2)
        handle1 = bh.acquire(blocking=False)
        bh.acquire(blocking=False)
        assert handle1 is not None
        bh.release(handle1)
        assert bh.acquire(blocking=False) is not None

    def test_context_manager_releases_automatically(self):
        bh = Bulkhead(max_concurrency=1)
        with bh:
            assert bh.available == 0
        assert bh.available == 1

    def test_available_count(self):
        bh = Bulkhead(max_concurrency=3)
        assert bh.available == 3
        h1 = bh.acquire(blocking=False)
        assert bh.available == 2
        bh.release(h1)
        assert bh.available == 3

    def test_acquire_blocking_raises_timeout(self):
        bh = Bulkhead(max_concurrency=1)
        bh.acquire(blocking=False)
        with pytest.raises(TimeoutError):
            bh.acquire(blocking=True, timeout=0.05)


class TestBulkheadDecorator:
    def test_decorator_allows_concurrent_calls(self):
        bh = Bulkhead(max_concurrency=2)

        @bulkhead(bh)
        def work(x):
            return x * 2

        assert work(5) == 10
        assert work(6) == 12

    def test_decorator_rejects_when_full(self):
        bh = Bulkhead(max_concurrency=1)

        @bulkhead(bh)
        def work():
            return "ok"

        handle = bh.acquire(blocking=False)
        with pytest.raises(Bulkhead.Full):
            work()
        bh.release(handle)

    @pytest.mark.asyncio
    async def test_decorator_with_async(self):
        bh = Bulkhead(max_concurrency=2)

        @bulkhead(bh)
        async def work(x):
            return x * 2

        result = await work(5)
        assert result == 10


class TestBulkheadFullException:
    def test_is_runtime_error(self):
        assert issubclass(Bulkhead.Full, RuntimeError)
