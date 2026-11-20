"""CLI script to fit all recommendation models, measure memory usage, and run quantitative evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cinematch.config import AppConfig  # noqa: E402
from cinematch.data.loader import MovieDataLoader  # noqa: E402
from cinematch.models.evaluation import evaluate_catalog_models  # noqa: E402
from cinematch.services.movie_service import MovieService  # noqa: E402
from cinematch.services.recommendation_service import RecommendationService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and evaluate CineMatch recommendation models.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for Precision@K, Recall@K, NDCG@K, MAP@K.")
    parser.add_argument("--max-queries", type=int, default=200, help="Number of evaluation queries to benchmark.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional dataset sample size.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    t0 = time.perf_counter()

    loader = MovieDataLoader(
        dataset_path=config.dataset_path,
        ratings_path=config.ratings_path,
        sample_size=args.sample_size or config.sample_size,
        random_state=config.random_state,
    )
    loaded = loader.load()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    movie_service = MovieService(loaded.movies_df, tmdb_image_base_url=config.tmdb_image_base_url)
    rec_service = RecommendationService(dataset=loaded, config=config, movie_service=movie_service)
    fit_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    eval_metrics = evaluate_catalog_models(
        movies_df=loaded.movies_df,
        k=args.k,
        max_queries=args.max_queries,
        random_state=config.random_state,
    )
    eval_time = time.perf_counter() - t2

    report = {
        "timing_seconds": {
            "data_load_and_cleaning": round(load_time, 3),
            "model_fitting": round(fit_time, 3),
            "quantitative_evaluation": round(eval_time, 3),
        },
        "health": rec_service.get_health_status(),
        "evaluation": {name: metrics.to_dict() for name, metrics in eval_metrics.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
