"""Tests for structlog integration with stdlib logging.

TDD: these tests define the expected behavior BEFORE implementation.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path

import pytest
import structlog

import data_engineering_copilot.config.logging as logging_mod

# ---------------------------------------------------------------------------
# 1. structlog configuration
# ---------------------------------------------------------------------------


class TestStructlogConfig:
    """setup_logging() should configure structlog processors."""

    def test_structlog_configured_after_setup(self, tmp_path: Path) -> None:
        """After setup_logging(), structlog should be configured."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            logging_mod.setup_logging(log_dir=str(tmp_path), log_filename="test.log")

            # structlog.get_logger should return a logger
            log = structlog.get_logger("test.module")
            assert log is not None
        finally:
            root.handlers.clear()

    def test_structlog_json_in_production(self, tmp_path: Path) -> None:
        """With LOG_LEVEL=INFO (production), log output should be JSON."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            os.environ["LOG_LEVEL"] = "INFO"
            importlib.reload(logging_mod)
            logging_mod.setup_logging(log_dir=str(tmp_path), log_filename="test.log")

            log = structlog.get_logger("test.json")
            log.info("test_event", key="value")

            log_file = tmp_path / "test.log"
            content = log_file.read_text()
            for line in content.strip().split("\n"):
                if "test_event" in line:
                    parsed = json.loads(line)
                    assert parsed["event"] == "test_event"
                    assert parsed["key"] == "value"
                    assert "timestamp" in parsed
                    assert parsed["level"] == "info"
                    return
            pytest.fail("No JSON log line containing 'test_event' found")
        finally:
            os.environ.pop("LOG_LEVEL", None)
            root.handlers.clear()

    def test_structlog_console_in_debug(self, tmp_path: Path) -> None:
        """With LOG_LEVEL=DEBUG, console should use human-readable format."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            os.environ["LOG_LEVEL"] = "DEBUG"
            importlib.reload(logging_mod)
            logging_mod.setup_logging(log_dir=str(tmp_path), log_filename="test.log")

            log = structlog.get_logger("test.console")
            log.info("hello_console")

            log_file = tmp_path / "test.log"
            content = log_file.read_text()
            # In DEBUG, the console handler should have readable text
            assert "hello_console" in content
        finally:
            os.environ.pop("LOG_LEVEL", None)
            root.handlers.clear()

    def test_structlog_includes_log_level(self, tmp_path: Path) -> None:
        """structlog output should include log level."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            os.environ["LOG_LEVEL"] = "INFO"
            importlib.reload(logging_mod)
            logging_mod.setup_logging(log_dir=str(tmp_path), log_filename="test.log")

            log = structlog.get_logger("test.level")
            log.warning("test_warn")

            log_file = tmp_path / "test.log"
            content = log_file.read_text()
            for line in content.strip().split("\n"):
                if "test_warn" in line:
                    parsed = json.loads(line)
                    assert parsed["level"] == "warning"
                    return
            pytest.fail("No log line containing 'test_warn' found")
        finally:
            root.handlers.clear()


# ---------------------------------------------------------------------------
# 2. structlog logger acquisition
# ---------------------------------------------------------------------------


class TestStructlogLogger:
    """Modules should use structlog.get_logger()."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """structlog.get_logger() should return a BoundLogger or proxy."""
        log = structlog.get_logger("test.acquire")
        assert log is not None

    def test_logger_includes_context(self, tmp_path: Path) -> None:
        """structlog logger should support bind() for context."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            os.environ["LOG_LEVEL"] = "INFO"
            importlib.reload(logging_mod)
            logging_mod.setup_logging(log_dir=str(tmp_path), log_filename="test.log")

            log = structlog.get_logger("test.ctx").bind(task_id="abc-123")
            log.info("context_test")

            log_file = tmp_path / "test.log"
            content = log_file.read_text()
            for line in content.strip().split("\n"):
                if "context_test" in line:
                    parsed = json.loads(line)
                    assert parsed["task_id"] == "abc-123"
                    return
            pytest.fail("No log line found")
        finally:
            os.environ.pop("LOG_LEVEL", None)
            root.handlers.clear()


# ---------------------------------------------------------------------------
# 3. Module-level structlog logger attribute
# ---------------------------------------------------------------------------


class TestStructuredFields:
    """Key modules should define a module-level 'log' via structlog.get_logger."""

    def test_worker_uses_structlog(self) -> None:
        from data_engineering_copilot.workers import tasks
        assert hasattr(tasks, "log"), "tasks.py should define a 'log' structlog logger"

    def test_api_routes_uses_structlog(self) -> None:
        from data_engineering_copilot.api import routes
        assert hasattr(routes, "log"), "routes.py should define a 'log' structlog logger"

    def test_crawler_uses_structlog(self) -> None:
        from data_engineering_copilot.infrastructure import crawler
        assert hasattr(crawler, "log"), "crawler.py should define a 'log' structlog logger"

    def test_ingestion_uses_structlog(self) -> None:
        from data_engineering_copilot.services import ingestion
        assert hasattr(ingestion, "log"), "ingestion.py should define a 'log' structlog logger"
