"""Telemetry adapter conforming to TelemetryTracerProtocol.

Provides ``LangfuseTelemetryTracer`` (primary, via Langfuse SDK),
``OTelTelemetryTracer`` (fallback, via OpenTelemetry), and ``NoOpTelemetryTracer``
(graceful no-op when neither is available).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class NoOpTelemetryTracer:
    """No-op telemetry tracer used when Langfuse is unavailable."""

    def start_observation(
        self,
        name: str,
        input: Any = None,
        as_type: str = "trace",
        model: str | None = None,
    ) -> _NoOpObservation:
        return _NoOpObservation()

    def flush(self) -> None:
        pass

    async def flush_async(self, timeout: float = 2.0) -> None:
        return None


class _NoOpObservation:
    """No-op observation returned by ``NoOpTelemetryTracer``."""

    def update(self, **kwargs: Any) -> _NoOpObservation:
        return self

    def end(self) -> _NoOpObservation:
        return self

    def start_observation(
        self,
        name: str,
        **kwargs: Any,
    ) -> _NoOpObservation:
        return _NoOpObservation()


class LangfuseTelemetryTracer:
    """Wraps a LangfuseCompat instance as a ``TelemetryTracerProtocol``."""

    def __init__(self, langfuse_compat: Any) -> None:
        self._client = langfuse_compat

    def start_observation(
        self,
        name: str,
        input: Any = None,
        as_type: str = "trace",
        model: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if model is not None:
            kwargs["model"] = model
        return self._client.start_observation(name=name, as_type=as_type, **kwargs)

    def score(self, trace_id: str, name: str, value: float, data_type: str = "NUMERIC", **kwargs) -> None:
        """Score a trace with a metric."""
        if self._client is not None:
            self._client.score(trace_id=trace_id, name=name, value=value, data_type=data_type, **kwargs)

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()

    async def flush_async(self, timeout: float = 2.0) -> None:
        if self._client is None:
            return None
        try:
            await asyncio.wait_for(asyncio.to_thread(self._client.flush), timeout=timeout)
        except Exception:
            logger.debug("Telemetry flush_async timed out or failed (silent)", exc_info=True)


def build_telemetry_tracer() -> NoOpTelemetryTracer | LangfuseTelemetryTracer | Any:
    """Factory that returns the best available tracer.

    Priority: Langfuse → OTel → NoOp.

    Langfuse is preferred because it provides richer observability (scoring,
    prompt management, cost tracking, user feedback) compared to raw OTel.
    OTel is used as a fallback when Langfuse is unavailable.
    """
    # Try Langfuse first — richer features, scoring, prompt management
    try:
        from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

        instance = get_langfuse_instance()
        if instance is not None:
            logger.info("Using Langfuse tracer (primary)")
            return LangfuseTelemetryTracer(instance)
    except Exception as exc:
        logger.debug("Langfuse telemetry unavailable: %s", exc)

    # Fall back to OpenTelemetry
    try:
        from data_engineering_copilot.observability.otel_telemetry import OTelTelemetryTracer

        otel = OTelTelemetryTracer()
        if otel._tracer is not None:
            logger.info("Using OpenTelemetry tracer (fallback)")
            return otel  # type: ignore[return-value]
    except Exception as exc:
        logger.debug("OTel telemetry unavailable: %s", exc)

    logger.info("Using NoOp tracer")
    return NoOpTelemetryTracer()
