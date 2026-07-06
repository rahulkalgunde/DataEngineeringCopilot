from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from logger_config import setup_logging


def _root_handler_types() -> set[type]:
    root = logging.getLogger()
    return {type(h) for h in root.handlers}


def _root_handler_count() -> int:
    return len(logging.getLogger().handlers)


def test_setup_logging_creates_log_file_and_handlers(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_file = log_dir / "app.log"

    setup_logging(log_dir=str(log_dir), log_filename="app.log")

    assert log_file.exists()
    handler_types = _root_handler_types()
    assert logging.StreamHandler in handler_types
    assert logging.handlers.RotatingFileHandler in handler_types


def test_setup_logging_writes_message_to_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_file = log_dir / "app.log"

    setup_logging(log_dir=str(log_dir), log_filename="app.log")

    test_logger = logging.getLogger("test_writer")
    test_logger.info("test log entry %s", 42)

    content = log_file.read_text(encoding="utf-8")
    assert "test log entry 42" in content


def test_setup_logging_idempotent_no_duplicate_handlers(tmp_path: Path) -> None:
    root = logging.getLogger()
    # Clear handlers to start clean
    root.handlers.clear()

    try:
        log_dir = tmp_path / "logs"

        setup_logging(log_dir=str(log_dir), log_filename="app.log")
        first_count = _root_handler_count()

        setup_logging(log_dir=str(log_dir), log_filename="app.log")
        second_count = _root_handler_count()

        # Should not have doubled the handlers
        assert second_count == first_count
    finally:
        root.handlers.clear()


def test_setup_logging_rotating_handler_config(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    setup_logging(log_dir=str(log_dir), log_filename="app.log")

    root = logging.getLogger()
    rotating_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(rotating_handlers) == 1
    handler = rotating_handlers[0]
    assert handler.maxBytes == 10 * 1024 * 1024
    assert handler.backupCount == 5
    assert handler.encoding == "utf8"


def test_setup_logging_custom_filename(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_file = log_dir / "custom.log"

    setup_logging(log_dir=str(log_dir), log_filename="custom.log")

    assert log_file.exists()


def test_setup_logging_creates_log_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "nested" / "logs"
    assert not log_dir.exists()

    setup_logging(log_dir=str(log_dir), log_filename="app.log")

    assert log_dir.exists()
    assert (log_dir / "app.log").exists()


def test_setup_logging_formats_include_both_handlers(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    setup_logging(log_dir=str(log_dir), log_filename="app.log")

    root = logging.getLogger()
    stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    file_handlers = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]

    assert len(stream_handlers) >= 1
    assert len(file_handlers) >= 1

    # Verify stream handler uses simple formatter
    assert stream_handlers[0].formatter is not None
    assert stream_handlers[0].formatter._fmt == "[%(levelname)s] - %(message)s"

    # Verify file handler uses standard formatter (with timestamp, module, etc.)
    assert file_handlers[0].formatter is not None
    assert "asctime" in file_handlers[0].formatter._fmt