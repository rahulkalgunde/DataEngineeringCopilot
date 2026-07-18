"""Tests for logging improvements: LOG_LEVEL env var, worker/API logging,
print() removal, logger naming, and dead code cleanup.

TDD: these tests define the expected behavior BEFORE implementation.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. LOG_LEVEL env var support in setup_logging()
# ---------------------------------------------------------------------------

class TestLogLevelEnvVar:
    """setup_logging() should respect the LOG_LEVEL environment variable."""

    def test_defaults_to_info_when_env_not_set(self, tmp_path: Path) -> None:
        """Without LOG_LEVEL, root logger should be INFO."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            os.environ.pop("LOG_LEVEL", None)
            from data_engineering_copilot.config.logging import setup_logging
            setup_logging(log_dir=str(tmp_path), log_filename="test.log")
            assert root.level == logging.INFO
        finally:
            root.handlers.clear()
            os.environ.pop("LOG_LEVEL", None)

    def test_respects_debug_env_var(self, tmp_path: Path) -> None:
        """LOG_LEVEL=DEBUG should set root logger to DEBUG."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            os.environ["LOG_LEVEL"] = "DEBUG"
            importlib.reload(importlib.import_module("data_engineering_copilot.config.logging"))
            from data_engineering_copilot.config.logging import setup_logging as reload_setup
            reload_setup(log_dir=str(tmp_path), log_filename="test.log")
            assert root.level == logging.DEBUG
        finally:
            root.handlers.clear()
            os.environ.pop("LOG_LEVEL", None)

    def test_respects_warning_env_var(self, tmp_path: Path) -> None:
        """LOG_LEVEL=WARNING should set root logger to WARNING."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            os.environ["LOG_LEVEL"] = "WARNING"
            importlib.reload(importlib.import_module("data_engineering_copilot.config.logging"))
            from data_engineering_copilot.config.logging import setup_logging as reload_setup
            reload_setup(log_dir=str(tmp_path), log_filename="test.log")
            assert root.level == logging.WARNING
        finally:
            root.handlers.clear()
            os.environ.pop("LOG_LEVEL", None)

    def test_lowercase_env_var_works(self, tmp_path: Path) -> None:
        """LOG_LEVEL=debug (lowercase) should work."""
        root = logging.getLogger()
        root.handlers.clear()
        try:
            os.environ["LOG_LEVEL"] = "debug"
            importlib.reload(importlib.import_module("data_engineering_copilot.config.logging"))
            from data_engineering_copilot.config.logging import setup_logging as reload_setup
            reload_setup(log_dir=str(tmp_path), log_filename="test.log")
            assert root.level == logging.DEBUG
        finally:
            root.handlers.clear()
            os.environ.pop("LOG_LEVEL", None)


# ---------------------------------------------------------------------------
# 2. Worker module logging
# ---------------------------------------------------------------------------

class TestWorkerLogging:
    """workers/tasks.py should have a logger and use it at key points."""

    def test_tasks_module_has_logger(self):
        """tasks.py must define a module-level logger."""
        from data_engineering_copilot.workers import tasks
        assert hasattr(tasks, "log"), "workers/tasks.py must define a 'log' attribute (structlog)"

    def test_tasks_logger_is_correct_name(self):
        """tasks.py 'log' attribute should be a structlog BoundLoggerLazyProxy."""
        from data_engineering_copilot.workers import tasks
        assert type(tasks.log).__name__ == "BoundLoggerLazyProxy"

    def test_async_ingest_task_has_logging_calls(self):
        """async_ingest_task function body should contain log.info and log.exception."""
        import ast
        from pathlib import Path

        tasks_path = Path(__file__).parent.parent.parent / "data_engineering_copilot/workers/tasks.py"
        source = tasks_path.read_text()
        tree = ast.parse(source)

        # Find the async_ingest_task function
        func_source_lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "async_ingest_task":
                func_source_lines = source.splitlines()[node.lineno - 1 : node.end_lineno]
                break

        func_text = "\n".join(func_source_lines)
        assert "log.info" in func_text, "async_ingest_task should call log.info"
        assert "log.exception" in func_text or "log.error" in func_text, \
            "async_ingest_task should call log.exception or log.error"


# ---------------------------------------------------------------------------
# 3. API routes logging
# ---------------------------------------------------------------------------

class TestApiRoutesLogging:
    """api/routes.py should log key operations."""

    def test_routes_logger_used(self):
        """routes.py should log at least one message during dispatch."""
        from data_engineering_copilot.api import routes
        assert hasattr(routes, "log"), "routes.py must define a 'log' attribute (structlog)"

    @patch("data_engineering_copilot.api.routes.get_redis_client")
    @patch("data_engineering_copilot.api.routes.async_ingest_task")
    def test_ingest_dispatch_logged(self, mock_task, mock_redis, caplog):
        """POST /api/v1/ingest should log the dispatch event."""
        mock_task.delay.return_value = MagicMock(id="task-123", state="PENDING")
        mock_redis.return_value = MagicMock()

        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.app import app
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="data_engineering_copilot.api.routes"):
            client.post("/api/v1/ingest", json={"source_names": ["Test"], "max_pages": 10})

        assert any("ingest" in record.message.lower() or "dispatch" in record.message.lower()
                    or "task" in record.message.lower()
                    for record in caplog.records), \
            "Ingest dispatch should be logged"

    @patch("data_engineering_copilot.api.routes.get_redis_client")
    @patch("data_engineering_copilot.api.routes.celery_app")
    def test_cancel_logged(self, mock_celery, mock_redis, caplog):
        """POST /cancel should log the cancellation."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis.return_value = mock_client

        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.app import app
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="data_engineering_copilot.api.routes"):
            client.post("/api/v1/ingest/some-task-id/cancel")

        assert any("cancel" in record.message.lower() for record in caplog.records), \
            "Cancel should be logged"


# ---------------------------------------------------------------------------
# 4. print() removal from production code
# ---------------------------------------------------------------------------

class TestNoPrintInProduction:
    """Production modules should use logger, not print()."""

    @pytest.mark.parametrize("module_path", [
        "data_engineering_copilot/services/ingestion.py",
        "data_engineering_copilot/infrastructure/crawler.py",
    ])
    def test_no_print_calls(self, module_path: str) -> None:
        """These modules should have no print() calls."""
        import ast
        full_path = Path(__file__).parent.parent.parent / module_path
        source = full_path.read_text()
        tree = ast.parse(source)

        print_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print" or isinstance(func, ast.Attribute) and func.attr == "print":
                    print_calls.append(node.lineno)

        assert not print_calls, (
            f"{module_path} has print() calls at lines {print_calls}. "
            "Use logger instead."
        )


# ---------------------------------------------------------------------------
# 5. CLI logger naming
# ---------------------------------------------------------------------------

class TestCliLoggerName:
    """cli.py should use __name__ for its logger, not a hardcoded string."""

    def test_cli_uses_module_logger_name(self):
        """The logger name should be 'data_engineering_copilot.cli', not 'data_engineering_copilot.main'."""
        from data_engineering_copilot import cli
        assert cli.logger.name == "data_engineering_copilot.cli"


# ---------------------------------------------------------------------------
# 6. UI dead code and wrong filename
# ---------------------------------------------------------------------------

class TestUiCleanup:
    """streamlit_app.py should reference the correct log filename."""

    def test_log_filename_is_app_log(self, tmp_path: Path) -> None:
        """The health tab should reference 'app.log', not 'application.log'."""
        import inspect

        from data_engineering_copilot.ui import streamlit_app
        source = inspect.getsource(streamlit_app)
        assert "application.log" not in source, (
            "streamlit_app.py references 'application.log' but the actual file is 'app.log'"
        )

    def test_ingestion_log_path_removed(self) -> None:
        """IngestionManager should no longer have ingestion_log_path method."""
        from data_engineering_copilot.ui.streamlit_app import IngestionManager
        assert not hasattr(IngestionManager, "ingestion_log_path"), (
            "IngestionManager.ingestion_log_path is dead code — remove it"
        )
