"""Tests for OTelTelemetryTracer — OpenTelemetry tracer wrapper.

Manages the module-level ``_tracer`` global carefully to avoid
cross-test contamination.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_engineering_copilot.observability import otel_telemetry


@pytest.fixture(autouse=True)
def _reset_otel_tracer():
    otel_telemetry._tracer = None
    yield
    otel_telemetry._tracer = None


class TestOTelTelemetryTracer:
    """OTelTelemetryTracer (otel_telemetry.py:40)."""

    def test_init_without_opentelemetry_creates_tracer_none(self):
        with patch("data_engineering_copilot.observability.otel_telemetry._ensure_tracer", return_value=None):
            tracer = otel_telemetry.OTelTelemetryTracer()
            assert tracer._tracer is None

    def test_init_with_opentelemetry_creates_tracer(self):
        mock_tracer = MagicMock()
        with patch("data_engineering_copilot.observability.otel_telemetry._ensure_tracer", return_value=mock_tracer):
            tracer = otel_telemetry.OTelTelemetryTracer()
            assert tracer._tracer is mock_tracer

    def test_start_observation_without_tracer_returns_noop_span(self):
        with patch("data_engineering_copilot.observability.otel_telemetry._ensure_tracer", return_value=None):
            tracer = otel_telemetry.OTelTelemetryTracer()
            span = tracer.start_observation(name="test")
            assert span._span is None

    def test_start_observation_with_tracer_creates_span(self):
        mock_otel_tracer = MagicMock()
        mock_span = MagicMock()
        mock_otel_tracer.start_span.return_value = mock_span
        with patch(
            "data_engineering_copilot.observability.otel_telemetry._ensure_tracer", return_value=mock_otel_tracer
        ):
            tracer = otel_telemetry.OTelTelemetryTracer()
            tracer.start_observation(name="test", input="hello", as_type="generation", model="test-model")
            mock_otel_tracer.start_span.assert_called_once_with("test")
            mock_span.set_attribute.assert_any_call("app.input", "hello")
            mock_span.set_attribute.assert_any_call("app.model", "test-model")
            mock_span.set_attribute.assert_any_call("app.span_type", "generation")

    def test_flush_is_noop(self):
        tracer = otel_telemetry.OTelTelemetryTracer()
        tracer.flush()


class TestOTelSpan:
    """_OTelSpan (otel_telemetry.py:67)."""

    def test_update_without_span_is_noop(self):
        span = otel_telemetry._OTelSpan(None)
        result = span.update(output="test", level="ERROR")
        assert result is span

    def test_update_with_span_sets_attributes(self):
        mock_span = MagicMock()
        span = otel_telemetry._OTelSpan(mock_span)
        span.update(output="result", level="INFO")
        mock_span.set_attribute.assert_called_once_with("app.output", "result")

    def test_update_error_calls_set_status(self):
        mock_span = MagicMock()
        span = otel_telemetry._OTelSpan(mock_span)
        span.update(output="error msg", level="ERROR")
        mock_span.set_status.assert_called_once()

    def test_end_without_span_is_noop(self):
        span = otel_telemetry._OTelSpan(None)
        span.end()

    def test_end_with_span_calls_end(self):
        mock_span = MagicMock()
        span = otel_telemetry._OTelSpan(mock_span)
        span.end()
        mock_span.end.assert_called_once()

    def test_start_observation_without_span_returns_noop(self):
        span = otel_telemetry._OTelSpan(None)
        child = span.start_observation(name="child")
        assert child._span is None

    def test_start_observation_with_span_uses_module_tracer(self):
        mock_tracer = MagicMock()
        mock_child_span = MagicMock()
        mock_tracer.start_span.return_value = mock_child_span
        otel_telemetry._tracer = mock_tracer
        span = otel_telemetry._OTelSpan(MagicMock())
        child = span.start_observation(name="child")
        mock_tracer.start_span.assert_called_once_with("child")
        assert child._span is mock_child_span


class TestEnsureTracer:
    """_ensure_tracer (otel_telemetry.py:19)."""

    def test_returns_cached_tracer(self):
        otel_telemetry._tracer = "cached"
        result = otel_telemetry._ensure_tracer()
        assert result == "cached"

    def test_opentelemetry_import_failure_returns_none(self):
        otel_telemetry._tracer = None
        with patch("builtins.__import__", side_effect=ImportError("no opentelemetry")):
            result = otel_telemetry._ensure_tracer()
        assert result is None

    def test_opentelemetry_import_success(self):
        otel_telemetry._tracer = None
        mock_tracer = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        mock_modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=MagicMock()),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(OTLPSpanExporter=MagicMock()),
        }
        with patch.dict("sys.modules", mock_modules):
            result = otel_telemetry._ensure_tracer()
        assert result is not None

    def test_opentelemetry_exception_falls_through(self):
        otel_telemetry._tracer = None
        with (
            patch.object(otel_telemetry, "_tracer", None),
            patch("opentelemetry.sdk.trace.TracerProvider", side_effect=RuntimeError("init failed")),
        ):
            result = otel_telemetry._ensure_tracer()
        assert result is None
