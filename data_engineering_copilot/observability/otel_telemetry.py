"""OpenTelemetry-based telemetry tracer implementing TelemetryTracerProtocol.

Creates real OTel spans that appear in any OTLP-compatible backend
(Grafana Tempo, Jaeger, SigNoz, Datadog, etc.).

Falls back gracefully if ``opentelemetry`` is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_tracer: Any = None


def _ensure_tracer() -> Any:
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("data-engineering-copilot")
    except Exception as exc:
        logger.debug("OpenTelemetry unavailable: %s", exc)
        _tracer = None
    return _tracer


class OTelTelemetryTracer:
    """OpenTelemetry-based tracer implementing TelemetryTracerProtocol."""

    def __init__(self) -> None:
        self._tracer = _ensure_tracer()

    def start_observation(
        self,
        name: str,
        input: Any = None,
        as_type: str = "trace",
        model: str | None = None,
    ) -> _OTelSpan:
        if self._tracer is None:
            return _OTelSpan(None)
        span = self._tracer.start_span(name)
        if input is not None:
            span.set_attribute("app.input", str(input)[:2000])
        if model is not None:
            span.set_attribute("app.model", model)
        span.set_attribute("app.span_type", as_type)
        return _OTelSpan(span)

    def flush(self) -> None:
        pass


class _OTelSpan:
    """Wrapper around an OTel span that conforms to the observation protocol."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def update(self, output: Any = None, level: str = "INFO", **kwargs: Any) -> _OTelSpan:
        if self._span is None:
            return self
        if output is not None:
            self._span.set_attribute("app.output", str(output)[:5000])
        if level == "ERROR":
            from opentelemetry import trace as otel_trace

            self._span.set_status(
                otel_trace.Status(otel_trace.StatusCode.ERROR, str(output))
            )
        return self

    def end(self) -> None:
        if self._span is not None:
            self._span.end()

    def start_observation(
        self,
        name: str,
        **kwargs: Any,
    ) -> _OTelSpan:
        if self._span is None:
            return _OTelSpan(None)
        child = self._tracer_ref().start_span(name) if self._tracer_ref() else None
        return _OTelSpan(child)

    def _tracer_ref(self) -> Any:
        return _tracer
