"""CLI script to fit all recommendation models, calibrate weights, and run reproducible benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cinematch import __version__  # noqa: E402
from cinematch.config import AppConfig  # noqa: E402
from cinematch.data.loader import MovieDataLoader  # noqa: E402
from cinematch.models.collaborative import CollaborativeRecommender  # noqa: E402
from cinematch.models.content_based import ContentBasedRecommender, ContentFeatureWeights  # noqa: E402
from cinematch.models.demographic import DemographicRecommender  # noqa: E402
from cinematch.models.evaluation import (  # noqa: E402
    build_relevance_oracle,
    calibrate_hybrid_weights,
    compute_paired_bootstrap_comparison,
    evaluate_catalog_models,
    evaluate_cold_start_stratification,
    evaluate_hybrid_weight_sensitivity,
    evaluate_user_level_collaborative_holdout,
    split_oracle_queries,
    split_ratings_train_val_test,
)
from cinematch.models.hybrid import HybridRecommender  # noqa: E402
from cinematch.services.movie_service import MovieService  # noqa: E402
from cinematch.services.recommendation_service import RecommendationService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and evaluate CineMatch recommendation models.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for ranking metrics.")
    parser.add_argument("--val-queries", type=int, default=100, help="Validation queries for calibration.")
    parser.add_argument("--max-queries", type=int, default=200, help="Held-out test queries for evaluation.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional dataset sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for reproducibility.")
    parser.add_argument("--ratings-path", type=str, default=None, help="Optional ratings CSV override.")
    parser.add_argument("--sensitivity", action="store_true", help="Run hybrid sensitivity on validation set.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to export benchmark JSON.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to export benchmark CSV.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    t0 = time.perf_counter()

    effective_ratings_path = Path(args.ratings_path) if args.ratings_path else config.ratings_path
    loader = MovieDataLoader(
        dataset_path=config.dataset_path,
        ratings_path=effective_ratings_path,
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
    oracle = build_relevance_oracle(loaded.movies_df, min_group_size=3, require_genre_overlap=False)
    val_queries, test_queries = split_oracle_queries(
        oracle, val_size=args.val_queries, test_size=args.max_queries, random_state=args.seed
    )

    # Strictly calibrate weights on the validation set ONLY
    eval_weights = ContentFeatureWeights(overview=2, genres=3, tagline=2, collection=0, production=1, language=1)
    demo_eval = DemographicRecommender(quantile=0.90).fit(loaded.movies_df)
    cb_eval = ContentBasedRecommender(weights=eval_weights).fit(loaded.movies_df)
    cf_eval: CollaborativeRecommender | None = None
    train_ratings = None
    test_ratings = None
    if loaded.ratings_df is not None and not loaded.ratings_df.empty:
        train_ratings, _val_ratings, test_ratings = split_ratings_train_val_test(
            loaded.ratings_df, random_state=args.seed
        )
        fitted_cf = CollaborativeRecommender(random_state=args.seed).fit(
            loaded.movies_df, ratings_df=train_ratings
        )
        if fitted_cf.is_available:
            cf_eval = fitted_cf

    calibrated_weights, calibration_history = calibrate_hybrid_weights(
        movies_df=loaded.movies_df,
        val_queries=val_queries,
        content_model=cb_eval,
        demographic_model=demo_eval,
        collab_model=cf_eval,
        k=args.k,
    )

    # Evaluate strictly on held-out test_queries
    eval_metrics = evaluate_catalog_models(
        movies_df=loaded.movies_df,
        k=args.k,
        max_queries=args.max_queries,
        random_state=args.seed,
        ratings_df=loaded.ratings_df,
        hybrid_weights=calibrated_weights,
        queries_override=test_queries,
        require_genre_overlap=False,
    )

    # Paired bootstrap on test_queries (Hybrid vs Content-Based)
    hyb_test = HybridRecommender(
        content_model=cb_eval,
        demographic_model=demo_eval,
        collaborative_model=cf_eval,
        weights=calibrated_weights,
    ).fit(loaded.movies_df)
    cb_test_pairs = [
        ([int(m) for m in cb_eval.recommend(identifier=q, top_n=args.k)["movie_id"].tolist()], rel)
        for q, rel in test_queries
    ]
    hy_test_pairs = [
        ([item.movie_id for item in hyb_test.recommend(identifier=q, top_n=args.k)], rel)
        for q, rel in test_queries
    ]
    bootstrap_meta = compute_paired_bootstrap_comparison(
        baseline_pairs=cb_test_pairs,
        candidate_pairs=hy_test_pairs,
        k=args.k,
        n_resamples=5000,
        random_state=args.seed,
        content_model=cb_eval,
    )

    cold_start_report = evaluate_cold_start_stratification(
        movies_df=loaded.movies_df,
        test_queries=test_queries,
        k=args.k,
        random_state=args.seed,
        ratings_df=loaded.ratings_df,
        hybrid_weights=calibrated_weights,
    )

    user_holdout_report: dict[str, Any] = {}
    if train_ratings is not None and test_ratings is not None and not test_ratings.empty:
        user_holdout_report = evaluate_user_level_collaborative_holdout(
            movies_df=loaded.movies_df,
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            k=args.k,
            max_users=args.max_queries,
            random_state=args.seed,
            hybrid_weights=calibrated_weights,
        )

    sensitivity_results: list[dict[str, Any]] = []
    if args.sensitivity:
        sensitivity_results = evaluate_hybrid_weight_sensitivity(
            movies_df=loaded.movies_df,
            k=args.k,
            max_queries=len(val_queries),
            random_state=args.seed,
            queries_override=val_queries,
        )
    eval_time = time.perf_counter() - t2

    report: dict[str, Any] = {
        "experiment_metadata": {
            "version": __version__,
            "evaluation_protocol": "v2.1.0_strict_train_val_test_holdout",
            "metadata_relevance_type": "franchise_collection_holdout_without_genre_overlap",
            "dataset_path": str(config.dataset_path),
            "ratings_path": str(effective_ratings_path) if effective_ratings_path else None,
            "catalog_size": len(loaded.movies_df),
            "oracle_total_queries": len(oracle),
            "validation_queries": len(val_queries),
            "test_queries": len(test_queries),
            "val_test_overlap": len({q for q, _ in val_queries} & {q for q, _ in test_queries}),
            "k": args.k,
            "random_seed": args.seed,
            "calibrated_hybrid_weights": {
                "content": calibrated_weights.content,
                "demographic": calibrated_weights.demographic,
                "collaborative": calibrated_weights.collaborative,
            },
            "collaborative_evaluated": "Collaborative Filtering" in eval_metrics,
        },
        "timing_seconds": {
            "data_load_and_cleaning": round(load_time, 3),
            "model_fitting": round(fit_time, 3),
            "quantitative_evaluation": round(eval_time, 3),
        },
        "health": rec_service.get_health_status(),
        "validation_calibration_history": calibration_history,
        "evaluation": {name: metrics.to_dict() for name, metrics in eval_metrics.items()},
        "paired_bootstrap_hybrid_vs_cb_test": bootstrap_meta,
        "cold_start_stratification": cold_start_report,
    }
    if user_holdout_report:
        report["observed_user_preferences_holdout"] = user_holdout_report
    if sensitivity_results:
        report["validation_weight_sensitivity"] = sensitivity_results

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
