# Changelog

All notable changes to the **CineMatch (`cinematch-recommender`)** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.1] - 2026-11-20

### Added
- **Production WSGI Entrypoint (`src/cinematch/wsgi.py`)**: Thread/process-safe singleton WSGI entrypoint (`cinematch.wsgi:app` / `get_wsgi_app()`) integrated with **Gunicorn** (`--preload`, configurable `GUNICORN_WORKERS`, `GUNICORN_TIMEOUT`, and `GUNICORN_GRACEFUL_TIMEOUT`).
- **Hybrid Weight Sensitivity Benchmark (`evaluate_hybrid_weight_sensitivity`)**: Added systematic weight sensitivity analysis and reproducible JSON/CSV export (`--seed`, `--sensitivity`, `--output-json`, `--output-csv`) to `scripts/train_models.py` and `src/cinematch/models/evaluation.py`.
- **Corrective & Final Production Hardening Audits**: Added `docs/audits/CORRECTIVE_AUDIT_V2.0.1.md` and `docs/audits/FINAL_AUDIT_V2.0.1.md`.
- **Multi-Version Python CI Matrix**: Expanded `.github/workflows/ci.yml` to run across Python `3.11`, `3.12`, and `3.13`.

### Changed
- **Docker Dataset Standardization (`data/movies_metadata.csv`)**: Unified dataset path convention to `data/movies_metadata.csv` across `docker-compose.yml` (`./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro`), `.env.example`, `AppConfig`, and documentation.
- **Query-Scaled & Thematic-Gated Hybrid Scoring (`src/cinematch/models/hybrid.py`)**: Calibrated query-level content similarity scaling and added a thematic relevance gate so high-rating unrelated blockbusters with zero content/collaborative similarity cannot override genuinely similar movies.
- **Full Coverage Including Application Factory**: Removed `src/cinematch/main.py` from `[tool.coverage.run] omit` in `pyproject.toml` and expanded automated tests to 31 unit/integration tests with **86.42% total coverage** (`97%` on `main.py`, `100%` on `wsgi.py`).

### Fixed
- **Collaborative Filtering Distinct-User Deduplication (`src/cinematch/models/collaborative.py`)**: Deduplicated multiple ratings by the same `(user_id, movie_id)` pair (retaining the latest rating by `timestamp` when available, or last log entry) and changed minimum interaction qualification to count distinct users (`nunique`) instead of raw rows.
- **Collaborative Cosine Similarity Calibration**: Replaced the linear `(sim + 1) / 2` mapping with non-negative clamping `max(0.0, cos)`, ensuring zero or negative latent correlations never produce artificial positive preference scores.
- **Prevented Duplicate Cold-Start Initialization**: Added `is_fitted` tracking on `CollaborativeRecommender` so `HybridRecommender.fit()` does not re-run collaborative fitting when initialized in cold-start standby mode.
- **Evaluation Oracle Genre Verification (`src/cinematch/models/evaluation.py`)**: Implemented explicit genre intersection verification alongside held-out `collection_name` membership in `build_relevance_oracle`, aligning the benchmark code 100% with the documented methodology.

---

## [2.0.0] - 2026-11-20

### Added
- **Modular Clean Architecture (`src/cinematch/`)**: Separated configuration, data validation/preprocessing, ML models, services, and REST API blueprints with Application Factory (`create_app`) and dependency injection.
- **Hybrid Recommendation Engine (`HybridRecommender`)**: Combines Content-Based Filtering, Demographic Filtering, and optional Collaborative Filtering with dynamic weight normalization and natural-language explanations (`explanation`) per movie.
- **Collaborative Filtering Module (`CollaborativeRecommender`)**: Item-based / TruncatedSVD latent factor model over real user-item interaction logs with explicit cold-start standby handling.
- **Quantitative Evaluation Pipeline (`src/cinematch/models/evaluation.py` & `scripts/train_models.py`)**: Leakage-free benchmark measuring `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Catalog Coverage`, `Intra-List Similarity`, and `Recommendation Diversity`.
- **Data Quality Pipeline (`src/cinematch/data/` & `scripts/validate_dataset.py`)**: Schema validation, safe numeric/date coercion, corrupted ID removal, deduplication by `movie_id`, reproducible sampling with `random_state`, and `DataQualityReport`.
- **Expanded REST API & Dark Streaming Frontend**: Endpoints `/api/health`, `/api/genres`, `/api/movies/search`, `/api/movies/<movie_id>`, `/api/recommend/<movie_id>`, and `/api/recommendations/hybrid/<movie_id>` with Pydantic v2 validation and XSS-safe frontend.
