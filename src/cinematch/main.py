"""Application Factory and local development entrypoint for the CineMatch web application."""

from __future__ import annotations

from pathlib import Path
from flask import Flask
from flask_cors import CORS

from cinematch.api.errors import register_error_handlers
from cinematch.api.routes import create_api_blueprint
from cinematch.config import AppConfig
from cinematch.data.loader import LoadedDataset, MovieDataLoader
from cinematch.services.movie_service import MovieService
from cinematch.services.recommendation_service import RecommendationService
from cinematch.utils.exceptions import CineMatchError
from cinematch.utils.logging import configure_logging, get_logger


def _resolve_frontend_paths() -> tuple[Path, Path]:
    """Resolve frontend template and static directories with fallback to legacy root folders."""
    repo_root = Path(__file__).resolve().parents[2]
    frontend_templates = repo_root / "frontend" / "templates"
    frontend_static = repo_root / "frontend" / "static"
    if frontend_templates.exists() and frontend_static.exists():
        return frontend_templates, frontend_static
    return repo_root / "templates", repo_root / "static"


def create_app(
    config: AppConfig | None = None,
    loaded_dataset: LoadedDataset | None = None,
) -> Flask:
    """
    Create and configure the Flask application instance using dependency injection.

    If `loaded_dataset` is omitted, loads and validates the dataset from `config.dataset_path`.
    If the dataset is missing or fails validation at startup, or if model fitting fails,
    the application boots in degraded mode so `/api/health` reports HTTP 503 cleanly.
    """
    app_config = config or AppConfig.from_env()
    app_config.validate()
    configure_logging(level=app_config.log_level, json_format=app_config.log_json)
    logger = get_logger("main")

    template_dir, static_dir = _resolve_frontend_paths()
    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.config["APP_CONFIG"] = app_config

    CORS(
        app,
        resources={r"/api/*": {"origins": app_config.cors_origins}},
    )

    @app.after_request
    def add_security_headers(response):  # type: ignore[no-untyped-def]
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    register_error_handlers(app)

    dataset = loaded_dataset
    if dataset is None:
        try:
            loader = MovieDataLoader(
                dataset_path=app_config.dataset_path,
                ratings_path=app_config.ratings_path,
                sample_size=app_config.sample_size,
                random_state=app_config.random_state,
            )
            dataset = loader.load()
        except CineMatchError as exc:
            logger.error("Application started without loaded dataset: %s", exc.message)
            dataset = None

    app.config["MOVIE_SERVICE"] = None
    app.config["RECOMMENDATION_SERVICE"] = None

    if dataset is not None:
        try:
            movie_service = MovieService(
                movies_df=dataset.movies_df,
                tmdb_image_base_url=app_config.tmdb_image_base_url,
            )
            rec_service = RecommendationService(
                dataset=dataset,
                config=app_config,
                movie_service=movie_service,
            )
            app.config["MOVIE_SERVICE"] = movie_service
            app.config["RECOMMENDATION_SERVICE"] = rec_service
            logger.info("CineMatch services initialized with %d movies.", movie_service.catalog_size)
        except Exception as exc:
            logger.error("Failed to initialize recommendation models; entering degraded mode: %s", exc)

    app.register_blueprint(create_api_blueprint())
    return app


def main() -> None:
    """Local development CLI entrypoint. In production, use `gunicorn cinematch.wsgi:app`."""
    config = AppConfig.from_env()
    app = create_app(config=config)
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
