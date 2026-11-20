# Changelog

All notable changes to the **CineMatch (`cinematch-recommender`)** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-11-20

### Added
- **Modular Clean Architecture (`src/cinematch/`)**: Separated configuration, data validation/preprocessing, ML models, services, and REST API blueprints with Application Factory (`create_app`) and dependency injection.
- **Hybrid Recommendation Engine (`HybridRecommender`)**: Combines Content-Based Filtering, Demographic Filtering, and optional Collaborative Filtering with dynamic weight normalization and natural-language explanations (`explanation`) per movie.
- **Collaborative Filtering Module (`CollaborativeRecommender`)**: Item-based / TruncatedSVD latent factor model over real user-item interaction logs with explicit cold-start standby handling (never fabricates synthetic user ratings).
- **Quantitative Evaluation Pipeline (`src/cinematch/models/evaluation.py` & `scripts/train_models.py`)**: Leakage-free benchmark measuring `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Catalog Coverage`, `Intra-List Similarity`, and `Recommendation Diversity`.
- **Data Quality Pipeline (`src/cinematch/data/` & `scripts/validate_dataset.py`)**: Schema validation, safe numeric/date coercion, corrupted ID removal, deduplication by `movie_id`, reproducible sampling with `random_state`, and `DataQualityReport`.
- **Expanded REST API**: Added `/api/health`, `/api/genres`, `/api/movies/search`, `/api/movies/<movie_id>`, `/api/recommend/<movie_id>`, and `/api/recommendations/hybrid/<movie_id>` with Pydantic v2 parameter validation and pagination.
- **Modernized Dark SaaS/Streaming Frontend**: Live search autocomplete, genre/year/rating filters, ML mode switcher, movie details modal, `localStorage` favorites watchlist, hybrid score breakdowns, and pagination.
- **Automated Testing & CI/CD**: Unit and integration test suites (`pytest`), static analysis (`ruff`, `mypy`), hardened non-root `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow (`.github/workflows/ci.yml`).
- **Comprehensive Technical Documentation**: Added `docs/INITIAL_TECHNICAL_AUDIT.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/DATA_DICTIONARY.md`, `docs/MODEL_EVALUATION.md`, `docs/DEPLOYMENT.md`, and `docs/FINAL_TECHNICAL_AUDIT.md`.

### Changed
- **Algorithmic Memory Optimization**: Replaced the dense quadratic $N \times N$ similarity matrix (`linear_kernel(tfidf_matrix, tfidf_matrix)`, which consumed ~3.2 GB RAM for 20k movies) with an on-demand sparse CSR vector product (`linear_kernel(tfidf_matrix[idx], tfidf_matrix)`), reducing TF-IDF memory consumption to **12.30 MB** across the entire catalog of **45,430 valid movies** (a 99.92% memory reduction vs. dense full-catalog storage).
- **Enriched Content-Based Features**: Evolved TF-IDF representation from raw `overview` only to namespace-weighted tokens across `overview`, `genres`, `tagline`, `belongs_to_collection`, `production_companies`, and `original_language`.

### Fixed
- **Duplicate Title HTTP 500 Crash**: Fixed bug where querying any movie with a duplicate title (e.g., `"Heat"`, `"Hamlet"`, `"Sabrina"`) returned a `pd.Series` of indices and crashed `sorted()` with `ValueError: The truth value of an array with more than one element is ambiguous`.

### Security
- **Eliminated DOM XSS Vulnerabilities**: Removed all dynamic `innerHTML` injections in `static/js/script.js`, replacing them with safe `document.createElement` and `textContent` operations plus strict regex validation for TMDB poster URLs.
- **Hardened Production Defaults**: Disabled hardcoded `debug=True`, restricted CORS origins via `CINEMATCH_CORS_ORIGINS`, added security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`), and sanitized error responses to prevent internal stack trace exposure.
