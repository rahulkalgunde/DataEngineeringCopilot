import logging
import logging.config
import os

def setup_logging(default_level=logging.INFO, log_dir="logs", log_filename="app.log"):
    """
    Sets up a production-grade logging configuration with automatic file rolling.
    """
    # Ensure the logs directory exists safely
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, log_filename)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,  # Keeps non-proprietary library logs active
        "formatters": {
            "standard": {
                # Clean, scannable structural pattern
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(funcName)s:%(lineno)d) - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "[%(levelname)s] - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": default_level,
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file_rolling": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": default_level,
                "formatter": "standard",
                "filename": log_path,
                # 10 MB per file maximum threshold
                "maxBytes": 10 * 1024 * 1024, 
                # Keep up to 5 history files before auto-purging the oldest
                "backupCount": 5,
                "encoding": "utf8"
            }
        },
        "root": {
            "handlers": ["console", "file_rolling"],
            "level": default_level
        }
    }

    logging.config.dictConfig(logging_config)