"""Resilience patterns: Circuit Breaker and Bulkhead.

Provides reusable decorators and context managers for isolating
slow/broken dependencies and preventing retry storms.
"""

from __future__ import annotations

import asyncio
import enum
import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# --------------------------------------------------------------------------- #
# Circuit Breaker
# --------------------------------------------------------------------------- #


class CircuitBreakerState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """State-machine circuit breaker.

    Parameters
    ----------
    failure_threshold : int
        Number of consecutive failures before the circuit opens.
    recovery_timeout : float
        Seconds to wait in OPEN state before transitioning to HALF_OPEN.
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 5.0) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def can_execute(self) -> bool:
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state is CircuitBreakerState.OPEN:
                return False
            if self._state is CircuitBreakerState.HALF_OPEN:
                return True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state is CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                logger.info("circuit_breaker.closed reason=half_open_success")

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    "circuit_breaker.opened failure_count=%d threshold=%d",
                    self._failure_count,
                    self._failure_threshold,
                )
            elif self._state is CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                logger.warning("circuit_breaker.reopened reason=half_open_failure")

    def execute(self) -> None:
        """Raise CircuitBreakerOpen if the circuit is open."""
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state is CircuitBreakerState.OPEN:
                raise CircuitBreakerOpen(f"Circuit breaker is OPEN. Retry after {self._recovery_timeout}s.")

    def _maybe_transition_to_half_open(self) -> None:
        if self._state is CircuitBreakerState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = CircuitBreakerState.HALF_OPEN
                logger.info("circuit_breaker.half_open")


def circuit_breaker(cb: CircuitBreaker) -> Callable[[F], F]:
    """Decorator that wraps a function with circuit breaker protection.

    Works with both sync and async callables.
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                cb.execute()
                try:
                    result = await func(*args, **kwargs)
                    cb.record_success()
                    return result
                except CircuitBreakerOpen:
                    raise
                except Exception:
                    cb.record_failure()
                    raise

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                cb.execute()
                try:
                    result = func(*args, **kwargs)
                    cb.record_success()
                    return result
                except CircuitBreakerOpen:
                    raise
                except Exception:
                    cb.record_failure()
                    raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# --------------------------------------------------------------------------- #
# Bulkhead
# --------------------------------------------------------------------------- #


class Bulkhead:
    """Semaphore-based concurrency limiter.

    Parameters
    ----------
    max_concurrency : int
        Maximum number of concurrent calls allowed.
    """

    class Full(RuntimeError):
        """Raised when the bulkhead rejects a call due to full capacity."""

    def __init__(self, max_concurrency: int = 4) -> None:
        self._max_concurrency = max_concurrency
        self._semaphore = threading.Semaphore(max_concurrency)
        self._available = max_concurrency
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        with self._lock:
            return self._available

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> Any:
        """Acquire a slot in the bulkhead.

        Returns a handle that must be passed to ``release()``.
        Raises ``Bulkhead.Full`` if non-blocking and capacity is exhausted.
        Raises ``TimeoutError`` if blocking with timeout and slot not acquired.
        """
        if blocking:
            acquired = self._semaphore.acquire(timeout=timeout)
            if not acquired:
                raise TimeoutError(f"Bulkhead full: could not acquire slot within {timeout}s")
            with self._lock:
                self._available -= 1
            return True
        else:
            acquired = self._semaphore.acquire(blocking=False)
            if not acquired:
                raise Bulkhead.Full(f"Bulkhead full: {self._max_concurrency} concurrent calls in progress")
            with self._lock:
                self._available -= 1
            return True

    def release(self, handle: Any = None) -> None:
        """Release a previously acquired slot."""
        self._semaphore.release()
        with self._lock:
            self._available += 1

    def __enter__(self) -> Bulkhead:
        self.acquire(blocking=True)
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


def bulkhead(bh: Bulkhead) -> Callable[[F], F]:
    """Decorator that wraps a function with bulkhead protection.

    Works with both sync and async callables.
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                bh.acquire(blocking=False)
                try:
                    return await func(*args, **kwargs)
                finally:
                    bh.release()

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                bh.acquire(blocking=False)
                try:
                    return func(*args, **kwargs)
                finally:
                    bh.release()

            return sync_wrapper  # type: ignore[return-value]

    return decorator
