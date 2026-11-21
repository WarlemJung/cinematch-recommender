"""Centralized application configuration with environment variable validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from cinematch.utils.exceptions import ConfigurationError

load_dotenv()


def _resolve_default_dataset_path() -> Path:
    """
    Resolve default dataset path following the standardized `data/movies_metadata.csv` convention,
    with fallback to root `movies_metadata.csv` if `data/movies_metadata.csv` does not exist yet.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_candidate = repo_root / "data" / "movies_metadata.csv"
    if data_candidate.exists():
        return data_candidate
    root_candidate = repo_root / "movies_metadata.csv"
    if root_candidate.exists():
        return root_candidate
    return data_candidate


def _resolve_default_ratings_path() -> Path | None:
    """Locate optional ratings.csv or ratings_small.csv if available."""
    repo_root = Path(__file__).resolve().parents[2]
    for candidate_rel in ("data/ratings.csv", "data/ratings_small.csv", "ratings.csv", "ratings_small.csv"):
        candidate = repo_root / candidate_rel
        if candidate.exists():
            return candidate
    return None


def _parse_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_int(val: str | None) -> int | None:
    if val is None or val.strip() == "" or val.strip().lower() == "none":
        return None
    try:
        parsed = int(val.strip())
        return parsed if parsed > 0 else None
    except ValueError as exc:
        raise ConfigurationError(f"Invalid integer configuration value: '{val}'") from exc


@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration object for the CineMatch application."""

    env: str = "production"
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5000", "http://127.0.0.1:5000"])

    # WSGI / Gunicorn production settings
    gunicorn_workers: int = 2
    gunicorn_timeout: int = 120
    gunicorn_graceful_timeout: int = 30

    # Data settings
    dataset_path: Path = field(default_factory=_resolve_default_dataset_path)
    ratings_path: Path | None = field(default_factory=_resolve_default_ratings_path)
    sample_size: int | None = None
    random_state: int = 42

    # Demographic model parameters
    demographic_quantile: float = 0.90
    demographic_min_votes: int | None = None

    # Content-based model feature weights
    weight_overview: int = 2
    weight_genres: int = 3
    weight_tagline: int = 2
    weight_collection: int = 2
    weight_production: int = 1
    weight_keywords: int = 2
    tfidf_max_features: int = 25000

    # Hybrid model weights
    hybrid_weight_content: float = 0.75
    hybrid_weight_demographic: float = 0.15
    hybrid_weight_collaborative: float = 0.10

    # API & pagination limits
    default_top_n: int = 12
    max_top_n: int = 50
    default_page_size: int = 20
    max_page_size: int = 100

    # External Visual Metadata (TMDB)
    tmdb_api_key: str | None = None
    tmdb_image_base_url: str = "https://image.tmdb.org/t/p/w500"

    def validate(self) -> None:
        """Validate configuration invariants and raise ConfigurationError on invalid settings."""
        if self.env.lower() == "production" and self.debug:
            raise ConfigurationError("DEBUG mode cannot be enabled when ENV is 'production'.")
        if not (0.0 <= self.demographic_quantile <= 1.0):
            raise ConfigurationError("demographic_quantile must be between 0.0 and 1.0.")
        if self.max_top_n < 1 or self.default_top_n < 1 or self.default_top_n > self.max_top_n:
            raise ConfigurationError("Invalid top_n bounds in configuration.")
        if self.gunicorn_workers < 1 or self.gunicorn_timeout < 1 or self.gunicorn_graceful_timeout < 1:
            raise ConfigurationError("Gunicorn workers and timeouts must be positive integers.")
        if self.hybrid_weight_content < 0 or self.hybrid_weight_demographic < 0 or self.hybrid_weight_collaborative < 0:
            raise ConfigurationError("Hybrid recommendation weights must be non-negative.")
        total_hybrid = (
            self.hybrid_weight_content + self.hybrid_weight_demographic + self.hybrid_weight_collaborative
        )
        if total_hybrid <= 0:
            raise ConfigurationError("Sum of hybrid weights must be strictly positive.")

    @classmethod
    def from_env(cls) -> AppConfig:
        """Construct and validate AppConfig from environment variables."""
        env_name = os.getenv("CINEMATCH_ENV", os.getenv("FLASK_ENV", "production")).strip().lower()
        debug_flag = _parse_bool(os.getenv("CINEMATCH_DEBUG"), default=False)
        if env_name == "production":
            debug_flag = False

        raw_origins = os.getenv("CINEMATCH_CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")
        cors_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

        dataset_env = os.getenv("CINEMATCH_DATASET_PATH")
        dataset_path = Path(dataset_env) if dataset_env else _resolve_default_dataset_path()

        ratings_env = os.getenv("CINEMATCH_RATINGS_PATH")
        ratings_path = Path(ratings_env) if ratings_env else _resolve_default_ratings_path()

        try:
            config = cls(
                env=env_name,
                host=os.getenv("CINEMATCH_HOST", "0.0.0.0"),
                port=int(os.getenv("CINEMATCH_PORT", "5000")),
                debug=debug_flag,
                log_level=os.getenv("CINEMATCH_LOG_LEVEL", "INFO"),
                log_json=_parse_bool(os.getenv("CINEMATCH_LOG_JSON"), default=False),
                cors_origins=cors_origins,
                gunicorn_workers=int(os.getenv("GUNICORN_WORKERS", "2")),
                gunicorn_timeout=int(os.getenv("GUNICORN_TIMEOUT", "120")),
                gunicorn_graceful_timeout=int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30")),
                dataset_path=dataset_path,
                ratings_path=ratings_path,
                sample_size=_parse_optional_int(os.getenv("CINEMATCH_SAMPLE_SIZE")),
                random_state=int(os.getenv("CINEMATCH_RANDOM_STATE", "42")),
                demographic_quantile=float(os.getenv("CINEMATCH_DEMOGRAPHIC_QUANTILE", "0.90")),
                demographic_min_votes=_parse_optional_int(os.getenv("CINEMATCH_DEMOGRAPHIC_MIN_VOTES")),
                tfidf_max_features=int(os.getenv("CINEMATCH_TFIDF_MAX_FEATURES", "25000")),
                hybrid_weight_content=float(os.getenv("CINEMATCH_HYBRID_WEIGHT_CONTENT", "0.75")),
                hybrid_weight_demographic=float(os.getenv("CINEMATCH_HYBRID_WEIGHT_DEMOGRAPHIC", "0.15")),
                hybrid_weight_collaborative=float(os.getenv("CINEMATCH_HYBRID_WEIGHT_COLLABORATIVE", "0.10")),
                default_top_n=int(os.getenv("CINEMATCH_DEFAULT_TOP_N", "12")),
                max_top_n=int(os.getenv("CINEMATCH_MAX_TOP_N", "50")),
                default_page_size=int(os.getenv("CINEMATCH_DEFAULT_PAGE_SIZE", "20")),
                max_page_size=int(os.getenv("CINEMATCH_MAX_PAGE_SIZE", "100")),
                tmdb_api_key=os.getenv("TMDB_API_KEY") or None,
                tmdb_image_base_url=os.getenv("TMDB_IMAGE_BASE_URL", "https://image.tmdb.org/t/p/w500"),
            )
        except ValueError as exc:
            raise ConfigurationError(f"Failed to parse environment configuration: {exc}") from exc

        config.validate()
        return config
