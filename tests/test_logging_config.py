from __future__ import annotations

import logging

from data_engineering_copilot.logging_config import LOGGER_NAME, configure_logging


def test_configure_logging_creates_application_log_without_duplicate_handlers(tmp_path) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    original_handlers = list(logger.handlers)
    logger.handlers.clear()

    try:
        first_path = configure_logging(tmp_path)
        second_path = configure_logging(tmp_path)
        logger.info("test log entry")

        assert first_path == tmp_path / "logs" / "application.log"
        assert second_path == first_path
        assert first_path.exists()
        assert len(logger.handlers) == 1
        assert "test log entry" in first_path.read_text(encoding="utf-8")
    finally:
        logger.handlers.clear()
        logger.handlers.extend(original_handlers)
