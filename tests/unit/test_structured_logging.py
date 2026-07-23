"""Tests for structured logger and LLM token tracking."""
from __future__ import annotations

import json
import logging

import pytest
from data_engineering_copilot.observability.structured_logging import (
    StructuredLogger,
    parse_log_record,
)


class TestStructuredLogger:
    def test_emits_json_format(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="test.structured"):
            sl = StructuredLogger("test.structured")
            sl.info("test_event", query="hello", count=5)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        # Message should be JSON parseable
        data = json.loads(record.getMessage())
        assert data["event"] == "test_event"
        assert data["query"] == "hello"
        assert data["count"] == 5
        assert "timestamp" in data

    def test_error_includes_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger="test.err"):
            sl = StructuredLogger("test.err")
            try:
                raise ValueError("boom")
            except ValueError:
                sl.error("failure", exc_info=True)

        data = json.loads(caplog.records[0].getMessage())
        assert data["event"] == "failure"
        assert "exc_type" in data
        assert data["exc_type"] == "ValueError"

    def test_child_logger_inherits(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="test.parent"):
            sl = StructuredLogger("test.parent.child")
            sl.info("child_event", x=1)

        data = json.loads(caplog.records[0].getMessage())
        assert data["event"] == "child_event"


class TestParseLogRecord:
    def test_valid_json(self):
        raw = '{"event": "test", "ts": "2024-01-01"}'
        result = parse_log_record(raw)
        assert result == {"event": "test", "ts": "2024-01-01"}

    def test_invalid_json_returns_raw(self):
        result = parse_log_record("not json at all")
        assert result == {"raw": "not json at all"}
