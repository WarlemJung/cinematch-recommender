"""Centralized error handling for the Flask API without leaking internal stack traces."""

from __future__ import annotations

from typing import Any
from flask import Flask, Response, jsonify
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from cinematch.utils.exceptions import CineMatchError
from cinematch.utils.logging import get_logger

logger = get_logger("api.errors")


def build_error_response(
    message: str,
    error_code: str,
    status_code: int,
    details: list[dict[str, Any]] | None = None,
) -> tuple[Response, int]:
    """Create a standardized JSON error payload."""
    payload: dict[str, Any] = {
        "error": message,
        "error_code": error_code,
        "status_code": status_code,
    }
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status_code


def register_error_handlers(app: Flask) -> None:
    """Attach domain, validation, and generic exception handlers to the Flask instance."""

    @app.errorhandler(CineMatchError)
    def handle_cinematch_error(exc: CineMatchError) -> tuple[Response, int]:
        logger.warning(
            "Domain error [%s] (%d): %s",
            exc.error_code,
            exc.status_code,
            exc.message,
        )
        return build_error_response(
            message=exc.message,
            error_code=exc.error_code,
            status_code=exc.status_code,
        )

    @app.errorhandler(ValidationError)
    def handle_pydantic_validation_error(exc: ValidationError) -> tuple[Response, int]:
        sanitized_errors: list[dict[str, Any]] = []
        for err in exc.errors():
            sanitized_errors.append(
                {
                    "field": ".".join(str(loc) for loc in err.get("loc", ())),
                    "message": str(err.get("msg", "Invalid value")),
                    "type": str(err.get("type", "value_error")),
                }
            )
        logger.warning("Request parameter validation failed: %s", sanitized_errors)
        return build_error_response(
            message="One or more request parameters are invalid.",
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=sanitized_errors,
        )

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException) -> tuple[Response, int]:
        code = exc.code or 500
        return build_error_response(
            message=exc.description or "HTTP error occurred.",
            error_code=f"HTTP_{code}",
            status_code=code,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_exception(exc: Exception) -> tuple[Response, int]:
        logger.exception("Unhandled server exception: %s", exc)
        return build_error_response(
            message="An unexpected internal server error occurred.",
            error_code="INTERNAL_SERVER_ERROR",
            status_code=500,
        )
