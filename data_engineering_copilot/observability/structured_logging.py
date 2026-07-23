"""Structured JSON logger for production observability."""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone


class StructuredLogger:
    """Emits log records as single-line JSON objects."""

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def _emit(self, level: int, event: str, kwargs: dict, exc_info: bool = False) -> None:
        data: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        if exc_info:
            import sys

            _, exc_value, _ = sys.exc_info()
            if exc_value is not None:
                data["exc_type"] = type(exc_value).__name__
                data["exc_message"] = str(exc_value)
                data["exc_traceback"] = traceback.format_exc()
        self._logger.log(level, json.dumps(data, default=str))

    def info(self, event: str, **kwargs: object) -> None:
        self._emit(logging.INFO, event, kwargs)

    def warning(self, event: str, **kwargs: object) -> None:
        self._emit(logging.WARNING, event, kwargs)

    def error(self, event: str, exc_info: bool = False, **kwargs: object) -> None:
        self._emit(logging.ERROR, event, kwargs, exc_info=exc_info)

    def debug(self, event: str, **kwargs: object) -> None:
        self._emit(logging.DEBUG, event, kwargs)


def parse_log_record(raw: str) -> dict:
    """Parse a JSON log record, returning raw dict on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"raw": raw}
