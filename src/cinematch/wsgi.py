"""Production WSGI entrypoint for Gunicorn and WSGI-compliant servers."""

from __future__ import annotations

from threading import Lock
from flask import Flask

from cinematch.config import AppConfig
from cinematch.main import create_app

_WSGI_LOCK = Lock()
_WSGI_APP: Flask | None = None


def get_wsgi_app(force_reload: bool = False) -> Flask:
    """
    Return the process-wide singleton Flask WSGI application.

    Ensures recommendation models and sparse TF-IDF matrices are initialized
    at most once per process, preventing duplicate memory allocation.
    """
    global _WSGI_APP
    with _WSGI_LOCK:
        if _WSGI_APP is None or force_reload:
            config = AppConfig.from_env()
            _WSGI_APP = create_app(config=config)
        return _WSGI_APP


def __getattr__(name: str) -> Flask:
    """Lazy module-level attribute access for `gunicorn cinematch.wsgi:app`."""
    if name in {"app", "application"}:
        return get_wsgi_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
