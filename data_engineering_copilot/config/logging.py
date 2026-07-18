import logging
import logging.config
import os

import structlog


def setup_logging(default_level=logging.INFO, log_dir="logs", log_filename="app.log"):
    """Configure stdlib + structlog for structured JSON logging.

    The root log level can be overridden via the ``LOG_LEVEL`` environment
    variable (e.g. ``LOG_LEVEL=DEBUG``).  When not set, *default_level* is
    used.

    In production (INFO+), output is JSON for log aggregators.
    In debug mode, output is human-readable console format.
    """
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if env_level and hasattr(logging, env_level):
        default_level = getattr(logging, env_level)

    # Ensure the logs directory exists safely
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_filename)
    is_debug = default_level <= logging.DEBUG

    # --- stdlib logging configuration ---
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structlog_json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            },
            "structlog_console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=False),
                ],
            },
            "plain": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(funcName)s:%(lineno)d) - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": default_level,
                "formatter": "structlog_console" if is_debug else "structlog_json",
                "stream": "ext://sys.stdout",
            },
            "file_rolling": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": default_level,
                "formatter": "structlog_json",
                "filename": log_path,
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "root": {"handlers": ["console", "file_rolling"], "level": default_level},
    }

    logging.config.dictConfig(logging_config)

    # --- structlog configuration ---
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
