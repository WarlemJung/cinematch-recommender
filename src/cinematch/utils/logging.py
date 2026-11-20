"""Structured logging configuration for CineMatch."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            payload.update(record.extra_data)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root and application loggers consistently."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger("cinematch")
    root_logger.setLevel(numeric_level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    if json_format:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a scoped logger under the cinematch namespace."""
    if not name.startswith("cinematch"):
        name = f"cinematch.{name}"
    return logging.getLogger(name)
