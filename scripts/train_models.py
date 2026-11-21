"""CLI script to fit all recommendation models, measure memory usage, and run reproducible quantitative benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cinematch import __version__  # noqa: E402
from cinematch.config import AppConfig  # noqa: E402
from cinematch.data.loader import MovieDataLoader  # noqa: E402
from cinematch.models.evaluation import (  # noqa: E402
    evaluate_catalog_models,
    evaluate_hybrid_weight_sensitivity,
)
from cinematch.models.hybrid import HybridWeights  # noqa: E402
from cinematch.services.movie_service import MovieService  # noqa: E402
from cinematch.services.recommendation_service import RecommendationService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and evaluate CineMatch recommendation models.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for Precision@K, Recall@K, NDCG@K, MAP@K.")
    parser.add_argument("--max-queries", type=int, default=200, help="Number of evaluation queries to benchmark.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional dataset sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for reproducibility.")
    parser.add_argument("--sensitivity", action="store_true", help="Run hybrid weight sensitivity analysis.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to export benchmark JSON.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to export benchmark CSV.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    t0 = time.perf_counter()

    loader = MovieDataLoader(
        dataset_path=config.dataset_path,
        ratings_path=config.ratings_path,
        sample_size=args.sample_size or config.sample_size,
        random_state=args.seed,
    )
    loaded = loader.load()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    movie_service = MovieService(loaded.movies_df, tmdb_image_base_url=config.tmdb_image_base_url)
    rec_service = RecommendationService(dataset=loaded, config=config, movie_service=movie_service)
    fit_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    configured_weights = HybridWeights(
        content=config.hybrid_weight_content,
        demographic=config.hybrid_weight_demographic,
        collaborative=config.hybrid_weight_collaborative,
    )
    eval_metrics = evaluate_catalog_models(
        movies_df=loaded.movies_df,
        k=args.k,
        max_queries=args.max_queries,
        random_state=args.seed,
        ratings_df=loaded.ratings_df,
        hybrid_weights=configured_weights,
    )
    eval_time = time.perf_counter() - t2

    sensitivity_results = []
    if args.sensitivity:
        sensitivity_results = evaluate_hybrid_weight_sensitivity(
            movies_df=loaded.movies_df,
            k=args.k,
            max_queries=args.max_queries,
            random_state=args.seed,
        )

    from typing import Any
    report: dict[str, Any] = {
        "experiment_metadata": {
            "version": __version__,
            "dataset_path": str(config.dataset_path),
            "ratings_path": str(config.ratings_path) if config.ratings_path else None,
            "catalog_size": len(loaded.movies_df),
            "evaluated_queries": args.max_queries,
            "k": args.k,
            "random_seed": args.seed,
            "hybrid_weights_configured": {
                "content": config.hybrid_weight_content,
                "demographic": config.hybrid_weight_demographic,
                "collaborative": config.hybrid_weight_collaborative,
            },
            "collaborative_evaluated": "Collaborative Filtering" in eval_metrics,
        },
        "timing_seconds": {
            "data_load_and_cleaning": round(load_time, 3),
            "model_fitting": round(fit_time, 3),
            "quantitative_evaluation": round(eval_time, 3),
        },
        "health": rec_service.get_health_status(),
        "evaluation": {name: metrics.to_dict() for name, metrics in eval_metrics.items()},
    }
    if sensitivity_results:
        report["hybrid_weight_sensitivity"] = sensitivity_results

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [m.to_dict() for m in eval_metrics.values()]
        if rows:
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
