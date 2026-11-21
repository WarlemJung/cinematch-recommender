# 🎬 CineMatch — Hybrid Movie Recommendation Engine

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B%20%7C%203.13-3776AB?logo=python&logoColor=white)
![Flask 3.1](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Sparse%20TF--IDF%20%26%20SVD-F7931E?logo=scikit-learn&logoColor=white)
![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Ruff & Mypy](https://img.shields.io/badge/Code%20Quality-Ruff%20%2B%20Mypy-261230)
![License: MIT](https://img.shields.io/badge/License-MIT-00f2ff)

> 🇧🇷 **[Leia a documentação em Português (README_pt.md)](README_pt.md)**

**CineMatch (`cinematch-recommender`)** is a production-grade **Hybrid Movie Recommendation Engine** built with Python, Scikit-Learn, SciPy sparse matrices, and Flask. It combines **Demographic Filtering (IMDB Bayesian Weighted Rating)**, **Sparse Content-Based Filtering (Multi-Field Weighted TF-IDF)**, and **Collaborative Filtering (TruncatedSVD Latent Factors with Cold-Start Fallback)** into an explainable ensemble served through a validated REST API and a modern dark-mode streaming interface.

---

## 🎯 1. Problem It Solves

Traditional prototype movie recommenders suffer from three critical engineering flaws:
1. **Quadratic Memory Blowup ($O(N^2)$):** Materializing a dense pairwise cosine similarity matrix (`linear_kernel(tfidf, tfidf)`) for 45,466 movies requires **~16.5 GB of RAM** (or ~3.2 GB even when truncated to 20,000 rows). CineMatch keeps the TF-IDF representation in **sparse CSR format (`float32`)** and computes similarity vectors **on demand in $O(\text{nnz}_{\text{query}} \cdot \bar{d})$**, processing the entire **45,430-movie catalog in just 12.30 MB of RAM** (a **99.92% memory reduction**).
2. **Single-Signal Blindness:** Content-only models often recommend poorly rated sequels, while popularity charts ignore user taste. CineMatch blends thematic similarity, Bayesian vote-weighted acclaim, and collaborative user-item signals with **natural-language explanations** for every recommendation.
3. **Unvalidated & Corrupted Data:** Raw movie metadata contains shifted CSV rows, duplicate IDs, missing synopses, and thousands of duplicate movie titles (e.g., remakes like *"Heat"* or *"Hamlet"*) that crash naive indexers. CineMatch enforces a strict data validation and disambiguation pipeline.

---

## ✨ 2. Key Features

- **Explainable Hybrid Recommendation Engine:** Generates human-readable explanations (e.g., shared franchise, thematic plot similarity, matching genres, shared studio, and weighted critical rating) alongside individual component scores.
- **Memory-Safe Sparse Content-Based Filtering:** Namespace-weighted tokens across `overview`, `genres`, `tagline`, `belongs_to_collection`, `production_companies`, and `original_language` without materializing an $N \times N$ matrix.
- **Bayesian Demographic Filtering (Top Charts):** Vectorized IMDB Weighted Rating with configurable vote quantile and multi-faceted filters (genre, release year range, minimum rating).
- **Isolated Collaborative Filtering with Cold-Start Protection:** Item-based / TruncatedSVD latent factor model trained exclusively on real user-item interactions (`ratings.csv`) when available, automatically redistributing ensemble weights during cold-start without fabricating synthetic ratings.
- **Quantitative Leakage-Free Evaluation Pipeline:** Built-in evaluation benchmark calculating `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Catalog Coverage`, `Intra-List Similarity (ILS)`, and `Recommendation Diversity`.
- **Hardened Security (AppSec):** Zero dynamic `innerHTML` usage (100% DOM XSS-safe via `textContent`), strict TMDB poster URL regex validation, Pydantic v2 parameter validation, restricted CORS, `debug=False` enforcement in production, and sanitized error responses.
- **Modern Streaming Dark UI:** Live search autocomplete, genre/year/rating filter bar, ML mode selector, rich movie details modal, `localStorage`-persisted Favorites watchlist, and catalog pagination.

---

## 🏗️ 3. Architecture

```text
cinematch-recommender/
├── src/
│   └── cinematch/
│       ├── main.py                  # Flask Application Factory (create_app)
│       ├── wsgi.py                  # Production WSGI singleton entrypoint (Gunicorn)
│       ├── config.py                # Immutable environment configuration & validation
│       ├── api/                     # Blueprint routes, Pydantic v2 schemas & error handlers
│       ├── data/                    # CSV loader, schema validation, cleaning & DataQualityReport
│       ├── models/                  # Demographic, Sparse Content-Based, Collaborative, Hybrid & Evaluation
│       ├── services/                # MovieService & RecommendationService orchestration
│       └── utils/                   # Domain exception hierarchy & structured JSON logging
├── frontend/
│   ├── templates/index.html         # Accessible dark-mode streaming web interface
│   └── static/                      # CSS design system & XSS-safe vanilla JS controller
├── scripts/
│   ├── validate_dataset.py          # Dataset integrity CLI & DataQualityReport generator
│   └── train_models.py              # Model fitting, memory telemetry & quantitative benchmark CLI
├── tests/
│   ├── unit/                        # Unit tests (data, math formulas, models, evaluation, config)
│   └── integration/                 # Full REST API & error-handling integration tests
├── docs/                            # Technical audits, architecture, API, data dictionary & evaluation
├── Dockerfile                       # Non-root production container with healthcheck
├── docker-compose.yml               # Single-command container orchestration
└── pyproject.toml                   # Packaging, pytest, coverage, ruff & mypy configuration
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component diagrams and design decisions.

---

## 🛠️ 4. Technologies Used

- **Language:** Python 3.11+ / 3.13
- **Machine Learning & Data:** `scikit-learn`, `scipy` (CSR sparse matrices), `pandas`, `numpy`
- **API & Validation:** `Flask`, `flask-cors`, `pydantic` (v2), `python-dotenv`
- **Frontend:** Semantic HTML5, Modern CSS3 (Custom Properties, Grid, Glassmorphism), Vanilla ES6+ JavaScript
- **Quality & DevOps:** `pytest`, `pytest-cov`, `ruff`, `mypy`, Docker, GitHub Actions CI

---

## 🧠 5. Algorithms Implemented

### 5.1. Demographic Filtering (IMDB Weighted Rating)
$$\text{WR} = \left(\frac{v}{v + m} \times R\right) + \left(\frac{m}{v + m} \times C\right)$$
Where $R$ is the movie's average rating, $v$ is its vote count, $m$ is the minimum votes threshold (90th percentile by default, $m = 160$ votes on the full catalog), and $C$ is the mean rating across the entire catalog ($C = 5.6182$).

### 5.2. Sparse Content-Based Filtering
Builds a weighted document per movie using namespace-isolated tokens (`genre_*`, `collection_*`, `studio_*`, `lang_*`) combined with free-text `overview` and `tagline`. Vectorized into a sparse $45,430 \times 25,000$ `float32` CSR matrix (**12.30 MB**). Similarity is computed on demand for the queried movie row $q$:
$$\mathbf{s}(q) = \mathbf{X}_{q, :} \cdot \mathbf{X}^T$$

### 5.3. Collaborative Filtering
Applies user-mean centering and `TruncatedSVD` dimensionality reduction over real user-item rating matrices (`userId, movieId, rating`). If no external ratings file is configured or an item lacks interaction history, the model enters *Cold-Start Standby* without synthesizing interactions.

### 5.4. Hybrid Weighted Ensemble
$$\text{Score}_{\text{hybrid}}(q, i) = \hat{w}_c S_{\text{content}}(q, i) + \hat{w}_d S_{\text{demographic}}(i) + \hat{w}_{cf} S_{\text{collaborative}}(q, i)$$
Weights $(\hat{w}_c, \hat{w}_d, \hat{w}_{cf})$ are automatically normalized to sum to `1.0` across the active models for query $q$.

---

## 🖥️ 6. Visual Preview

The web interface features:
- **Interactive Filter Bar:** Filter by Genre, Release Year range, Minimum Rating, and switch live between **Hybrid Ensemble** and **Content-Based TF-IDF**.
- **Autocomplete Search:** Instant title search with release year and rating disambiguation for duplicate titles.
- **Explainable Recommendation Cards:** Displays match percentage, individual model score chips (`Conteúdo`, `Demográfico`, `Colab`), and a natural-language explanation banner.
- **Persisted Favorites Watchlist & Movie Details Modal:** Save favorite films in `localStorage` and inspect full cast/studio/collection metadata.

---

## 📦 7. Installation & Configuration

### 7.1. Clone & Install Dependencies
```bash
git clone https://github.com/WarlemJung/cinematch-recommender.git
cd cinematch-recommender
pip install -r requirements.txt
```

### 7.2. Dataset Setup
Place `movies_metadata.csv` (from [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)) in the standardized `data/movies_metadata.csv` directory.
Optionally, place `ratings_small.csv` or `ratings.csv` in `data/` and set `CINEMATCH_RATINGS_PATH` to enable Collaborative Filtering.

### 7.3. Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

---

## 🚀 8. Running Locally & With Docker

### 8.1. Validate Dataset & Train/Evaluate Models
```bash
python scripts/validate_dataset.py
python scripts/train_models.py --k 10 --max-queries 200 --seed 42 --sensitivity
```

### 8.2. Start the Web Application (Production WSGI vs. Local Dev)
```bash
# Production WSGI server (Linux / macOS / Container):
PYTHONPATH=src gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --preload cinematch.wsgi:app

# Local development server (Windows / macOS / Linux):
python -m cinematch.main
```
Open `http://localhost:5000` in your browser.

### 8.3. Run with Docker Compose
```bash
docker compose up --build -d
```

---

## 🔌 9. REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web user interface |
| `GET` | `/api/health` | Service health check, sparse matrix memory usage & data quality report |
| `GET` | `/api/trending` | Paginated top movies by Weighted Rating (supports `genre`, `min_year`, `max_year`, `min_rating`) |
| `GET` | `/api/genres` | List of catalog genres and movie counts |
| `GET` | `/api/movies/search` | Paginated title search (`?q=...`) with multi-filter support |
| `GET` | `/api/movies/<movie_id>` | Detailed metadata for a specific movie ID |
| `GET` | `/api/recommend/<movie_id>` | Content-based recommendations by movie ID (also supports `/api/recommend?title=...`) |
| `GET` | `/api/recommendations/hybrid/<movie_id>` | Explainable hybrid recommendations by movie ID (also supports `/api/recommendations/hybrid?title=...`) |

Full payload examples are documented in [docs/API.md](docs/API.md).

---

## 🧪 10. Automated Testing & Quality Checks

```bash
# Run unit and integration test suites with coverage enforcement (>= 80%)
python -m pytest --cov=src/cinematch --cov-report=term-missing --cov-fail-under=80 -v

# Run Ruff linter and Mypy static type checker
python -m ruff check src tests scripts app.py recommender.py
python -m mypy src tests scripts app.py recommender.py
```

---

## 📊 11. Quantitative Evaluation Results

Measured on the full validated catalog (`N = 45,430` movies, `K = 10`, `Q = 200` queries, `seed = 42`) using the **Feature-Holdout Anti-Leakage Protocol** (where `belongs_to_collection` is withheld from training features and combined with genre intersection strictly as the ground-truth relevance oracle):

| Algorithm | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage | Intra-List Similarity | Diversity |
|---|---|---|---|---|---|---|---|
| **Demographic Filtering** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0002` | `0.0171` | **`0.9829`** |
| **Content-Based Filtering** | `0.2045` | `0.4231` | `0.4244` | `0.3531` | `0.0350` | `0.2508` | `0.7492` |
| **Hybrid Recommendation** | **`0.2070`** | **`0.4358`** | **`0.4290`** | **`0.3558`** | **`0.0354`** | `0.2485` | `0.7515` |

See [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) for full methodology, hybrid weight sensitivity analysis, and formulas.

---

## ⚠️ 12. Known Limitations

1. **External User Ratings Optionality:** `movies_metadata.csv` contains aggregated vote counts/averages rather than individual user rating logs. Unless a MovieLens-compatible `ratings.csv` is configured via `CINEMATCH_RATINGS_PATH`, Collaborative Filtering remains in cold-start standby mode and the Hybrid engine redistributes weights across Content-Based and Demographic components.
2. **English-Centric Synopses:** Plot overviews in `movies_metadata.csv` are primarily in English, though structured tokens (genres, collections, studios, original language) operate language-independently.

---

## 📄 13. License & Credits

- **License:** Distributed under the [MIT License](LICENSE).
- **Author:** Created by **Warlem Jung**.
- **Dataset Attribution:** Metadata sourced from *The Movies Dataset* (TMDB / GroupLens MovieLens).
