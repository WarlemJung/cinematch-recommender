"""
Entrypoint wrapper for CineMatch web application.

Delegates to the modular Application Factory in `src/cinematch/main.py` with safe production defaults.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cinematch.config import AppConfig  # noqa: E402
from cinematch.main import create_app  # noqa: E402


def main() -> None:
    """Start the CineMatch Flask application using environment configuration (debug=False by default)."""
    config = AppConfig.from_env()
    application = create_app(config=config)
    application.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()
