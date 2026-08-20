"""
app/core/logging.py
────────────────────
Structured JSON logging configuration for picasso-rag-chat.

Call ``configure_logging()`` once at application startup (in app/main.py
lifespan). All modules should use ``logging.getLogger(__name__)``.
"""
from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any


# ── Formatters ────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Minimal JSON-line formatter for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        import json
        import traceback

        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if record.exc_info:
            payload["exc"] = "".join(traceback.format_exception(*record.exc_info))

        # Carry any extra fields attached via logger.info("...", extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "msg", "args", "exc_info", "exc_text", "stack_info",
                "levelname", "levelno", "pathname", "filename", "module",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name", "message",
                "taskName",
            ) and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


# ── Public API ─────────────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> None:
    """Set up application-wide structured logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": _JsonFormatter,
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "json",
            },
        },
        "root": {
            "level": numeric_level,
            "handlers": ["stdout"],
        },
        # Quiet noisy third-party loggers
        "loggers": {
            "uvicorn.access": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "openai": {"level": "WARNING"},
            "sqlalchemy.engine": {"level": "WARNING"},
        },
    }

    logging.config.dictConfig(config)
    logging.getLogger(__name__).info(
        "Logging configured", extra={"log_level": level}
    )
