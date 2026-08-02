"""OpenTelemetry-based telemetry tracer implementing TelemetryTracerProtocol.

Creates real OTel spans that appear in any OTLP-compatible backend
(Grafana Tempo, Jaeger, SigNoz, Datadog, etc.).

Relies on ``BatchSpanProcessor`` internal retry/buffering to handle
collector unavailability — no proactive TCP reachability check.
Falls back gracefully if ``opentelemetry`` is not installed.
"""

from __future__ import annotations

import contextlib
import logging
import os
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
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

        # Additive resource metadata so spans are attributed to this service,
        # release, and environment in the telemetry backend.
        resource = Resource.create(
            {
                "service.name": "data-engineering-copilot",
                "service.version": os.environ.get("IMAGE_GIT_SHA", "unknown"),
                "deployment.environment": os.environ.get("APP_ENV", "development"),
            }
        )

        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("data-engineering-copilot")
        logger.info("OpenTelemetry tracer initialized for endpoint=%s", endpoint)
    except Exception as exc:
        logger.warning("OpenTelemetry unavailable, falling back to NoOp tracer: %s", exc)
        _tracer = None
    return _tracer


def extract_w3c_context(headers: dict[str, str] | None) -> Any:
    """Extract W3C trace context from incoming HTTP headers.

    Uses ``opentelemetry.propagate.extract`` (traceparent/tracestate) so that
    spans created while the returned context is attached continue the upstream
    trace. Returns ``None`` when telemetry is unavailable or no valid headers
    are present — callers should then just attach the current (root) context.
    """
    if _tracer is None:
        return None
    try:
        from opentelemetry import context as otel_context
        from opentelemetry.propagate import extract

        carrier = {}
        if headers:
            for key, value in headers.items():
                if key.lower() in ("traceparent", "tracestate"):
                    carrier[key] = value
        # With no trace headers, extract() returns the current context (root),
        # which is exactly what we want: continue a trace if one is active.
        return extract(carrier) if carrier else otel_context.get_current()
    except Exception:
        return None


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

    async def flush_async(self, timeout: float = 2.0) -> None:
        return None


class _OTelSpan:
    """Wrapper around an OTel span that conforms to the observation protocol."""

    def __init__(self, span: Any) -> None:
        self._span = span
        self._span_context: Any = None
        if span is not None:
            # Make the span current for the duration of its lifetime so nested
            # observations become children (enables full trace trees).
            try:
                from opentelemetry import trace as otel_trace

                self._span_context = otel_trace.use_span(span).__enter__()
            except Exception:
                self._span_context = None

    def update(self, output: Any = None, level: str = "INFO", **kwargs: Any) -> _OTelSpan:
        if self._span is None:
            return self
        if output is not None:
            self._span.set_attribute("app.output", str(output)[:5000])
        if level == "ERROR":
            from opentelemetry import trace as otel_trace

            self._span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(output)))
        return self

    def end(self) -> None:
        if self._span is not None:
            if self._span_context is not None:
                with contextlib.suppress(Exception):
                    self._span_context.__exit__(None, None, None)
                self._span_context = None
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
