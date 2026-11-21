"""Unit tests for quantitative evaluation metrics, oracle criteria, configuration, WSGI, and services."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import pandas as pd
import pytest

from cinematch.config import AppConfig
from cinematch.data.loader import LoadedDataset
from cinematch.data.preprocessing import preprocess_movies_dataframe
from cinematch.models.content_based import ContentBasedRecommender
from cinematch.models.evaluation import (
    average_precision_at_k,
    build_item_self_information_map,
    build_relevance_oracle,
    calibrate_hybrid_weights,
    catalog_coverage,
    compute_paired_bootstrap_comparison,
    evaluate_catalog_models,
    evaluate_cold_start_stratification,
    evaluate_hybrid_weight_sensitivity,
    evaluate_user_level_collaborative_holdout,
    hit_rate_at_k,
    intra_list_similarity,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
    recommendation_diversity,
    split_oracle_queries,
    split_ratings_train_val_test,
)
from cinematch.services.movie_service import MovieService
from cinematch.utils.exceptions import ConfigurationError, MovieNotFoundError
from cinematch.utils.logging import StructuredJsonFormatter
from recommender import MovieRecommenderSystem


def test_ranking_evaluation_metrics_mathematics() -> None:
    recommended = [10, 20, 30, 40, 50]
    relevant = {10, 30, 60}

    assert math.isclose(precision_at_k(recommended, relevant, k=5), 2 / 5)
    assert math.isclose(recall_at_k(recommended, relevant, k=5), 2 / 3)
    assert math.isclose(average_precision_at_k(recommended, relevant, k=5), 5 / 9)

    expected_idcg = 1.0 + (1.0 / math.log2(3)) + 0.5
    assert math.isclose(ndcg_at_k(recommended, relevant, k=5), 1.5 / expected_idcg)

    # Edge cases k=0 or empty relevant set
    assert precision_at_k(recommended, relevant, k=0) == 0.0
    assert recall_at_k(recommended, set(), k=5) == 0.0
    assert average_precision_at_k(recommended, set(), k=5) == 0.0
    assert ndcg_at_k(recommended, set(), k=5) == 0.0
    assert catalog_coverage([[10]], catalog_size=0) == 0.0

    lists = [[10, 20], [20, 30]]
    assert math.isclose(catalog_coverage(lists, catalog_size=10), 3 / 10)


def test_relevance_oracle_enforces_both_collection_and_genre_intersection(
    raw_movies_dataframe: pd.DataFrame,
) -> None:
    # Add a movie in 'Toy Story Collection' that has a completely disjoint genre ('Horror')
    mismatch_row = pd.DataFrame(
        [
            {
                "id": "99999",
                "imdb_id": "tt9999999",
                "title": "Toy Story Horror Spin-Off",
                "overview": "A scary doll story.",
                "tagline": "",
                "genres": "[{'id': 27, 'name': 'Horror'}]",
                "belongs_to_collection": "{'id': 10194, 'name': 'Toy Story Collection'}",
                "production_companies": "[]",
                "release_date": "2020-01-01",
                "vote_average": 5.0,
                "vote_count": 50,
                "popularity": 1.0,
                "runtime": 90.0,
                "original_language": "en",
                "poster_path": None,
            }
        ]
    )
    df_with_mismatch, _ = preprocess_movies_dataframe(
        pd.concat([raw_movies_dataframe, mismatch_row], ignore_index=True)
    )
    oracle = dict(build_relevance_oracle(df_with_mismatch, min_group_size=3, require_genre_overlap=True))
    # For Toy Story (862, genres: Animation, Comedy, Family), 99999 (Horror) must NOT be in relevant set
    assert 862 in oracle
    assert 863 in oracle[862]
    assert 10193 in oracle[862]
    assert 99999 not in oracle[862]

    # When require_genre_overlap=False (default v2.1.0), oracle uses only collection_name
    # so genres_list has zero overlap with oracle construction.
    strict_oracle = dict(build_relevance_oracle(df_with_mismatch, min_group_size=3))
    assert 99999 in strict_oracle[862]


def test_diversity_and_catalog_evaluation_pipeline_with_and_without_ratings(
    cleaned_dataset: LoadedDataset,
    synthetic_ratings_dataframe: pd.DataFrame,
) -> None:
    content = ContentBasedRecommender().fit(cleaned_dataset.movies_df)
    rec_lists = [[862, 863, 10193], [949, 500, 680]]

    ils = intra_list_similarity(rec_lists, content)
    div = recommendation_diversity(rec_lists, content)
    assert 0.0 <= ils <= 1.0
    assert math.isclose(ils + div, 1.0, rel_tol=1e-6)

    # Without ratings_df -> 3 models evaluated
    eval_summary = evaluate_catalog_models(cleaned_dataset.movies_df, k=3, max_queries=10)
    assert set(eval_summary.keys()) == {
        "Demographic Filtering",
        "Content-Based Filtering",
        "Hybrid Recommendation",
    }
    assert eval_summary["Content-Based Filtering"].precision_at_k > 0.0
    assert eval_summary["Hybrid Recommendation"].precision_at_k > 0.0

    # With real ratings_df -> all 4 models evaluated (including Collaborative Filtering)
    eval_with_collab = evaluate_catalog_models(
        cleaned_dataset.movies_df,
        k=3,
        max_queries=10,
        ratings_df=synthetic_ratings_dataframe,
    )
    assert "Collaborative Filtering" in eval_with_collab
    assert eval_with_collab["Collaborative Filtering"].precision_at_k > 0.0

    # Sensitivity analysis
    sens = evaluate_hybrid_weight_sensitivity(
        cleaned_dataset.movies_df,
        weight_configs=[(0.85, 0.15), (0.70, 0.30)],
        k=3,
        max_queries=10,
    )
    assert len(sens) == 2
    assert sens[0]["weight_content"] == 0.85


def test_app_config_validation_rules_and_structured_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="DEBUG mode cannot be enabled"):
        AppConfig(env="production", debug=True).validate()

    with pytest.raises(ConfigurationError, match="demographic_quantile"):
        AppConfig(demographic_quantile=1.5).validate()

    with pytest.raises(ConfigurationError, match="non-negative"):
        AppConfig(hybrid_weight_content=-0.5).validate()

    with pytest.raises(ConfigurationError, match="Gunicorn workers"):
        AppConfig(gunicorn_workers=0).validate()

    monkeypatch.setenv("CINEMATCH_ENV", "development")
    monkeypatch.setenv("CINEMATCH_DEBUG", "true")
    monkeypatch.setenv("CINEMATCH_SAMPLE_SIZE", "500")
    monkeypatch.setenv("CINEMATCH_LOG_JSON", "true")
    monkeypatch.setenv("GUNICORN_WORKERS", "4")
    monkeypatch.setenv("GUNICORN_TIMEOUT", "90")
    cfg = AppConfig.from_env()
    assert cfg.env == "development"
    assert cfg.debug is True
    assert cfg.sample_size == 500
    assert cfg.log_json is True
    assert cfg.gunicorn_workers == 4
    assert cfg.gunicorn_timeout == 90

    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="cinematch.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Structured test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert '"message": "Structured test message"' in formatted


def test_movie_service_search_and_details(cleaned_dataset: LoadedDataset) -> None:
    service = MovieService(cleaned_dataset.movies_df)
    assert service.catalog_size == len(cleaned_dataset.movies_df)

    details = service.get_movie_by_id(862)
    assert details["title"] == "Toy Story"
    assert details["collection_name"] == "Toy Story Collection"
    assert details["poster_url"] == "https://image.tmdb.org/t/p/w500/rhIRbceoE9lR4veEXuwCC2wARtG.jpg"

    with pytest.raises(MovieNotFoundError):
        service.get_movie_by_id(999999)

    genres = service.get_genres()
    assert any(g["name"] == "Animation" for g in genres)

    search_res = service.search_movies(query="Heat", page=1, page_size=10)
    assert search_res["total"] == 2
    assert search_res["items"][0]["movie_id"] == 949
    assert search_res["items"][1]["movie_id"] == 20776


def test_legacy_movie_recommender_system_adapter(
    tmp_path: Path,
    raw_movies_dataframe: pd.DataFrame,
) -> None:
    csv_path = tmp_path / "movies_metadata.csv"
    raw_movies_dataframe.to_csv(csv_path, index=False)

    legacy = MovieRecommenderSystem(csv_path)
    demo_df = legacy.get_demographic_recommendations(top_n=3)
    assert len(demo_df) == 3

    legacy.compute_content_similarity()
    recs = legacy.get_content_recommendations("Toy Story", top_n=2)
    assert len(recs) == 2
    assert "Toy Story" not in recs.tolist()

    heat_recs = legacy.get_content_recommendations("Heat", top_n=2)
    assert len(heat_recs) == 2


def test_dependency_manifest_consistency_between_pyproject_and_requirements() -> None:
    import tomllib

    repo_root = Path(__file__).resolve().parents[2]
    pyproject_data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    expected_deps = set(pyproject_data["project"]["dependencies"]) | set(
        pyproject_data["project"]["optional-dependencies"]["dev"]
    )
    req_lines = {
        line.strip()
        for line in (repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert expected_deps == req_lines


def test_strict_val_test_split_temporal_ratings_and_bootstrap_metrics(
    cleaned_dataset: LoadedDataset,
    synthetic_ratings_dataframe: pd.DataFrame,
) -> None:
    # 1. HitRate@K and Novelty@K
    assert hit_rate_at_k([10, 20], {20}, k=2) == 1.0
    assert hit_rate_at_k([10, 20], {99}, k=2) == 0.0
    nov_map = build_item_self_information_map(cleaned_dataset.movies_df)
    assert len(nov_map) == len(cleaned_dataset.movies_df)
    nov = novelty_at_k([[862, 777]], nov_map, k=2)
    assert 0.0 <= nov <= 1.0

    # 2. Disjoint validation/test split of oracle queries
    oracle = build_relevance_oracle(cleaned_dataset.movies_df, min_group_size=3)
    val_q, test_q = split_oracle_queries(oracle, val_size=3, test_size=3, random_state=42)
    assert len(val_q) > 0 and len(test_q) > 0
    assert {q for q, _ in val_q}.isdisjoint({q for q, _ in test_q})

    # 3. Calibration strictly on validation queries & paired bootstrap on test queries
    from cinematch.models.content_based import ContentFeatureWeights
    from cinematch.models.demographic import DemographicRecommender

    demo = DemographicRecommender(quantile=0.5).fit(cleaned_dataset.movies_df)
    cb = ContentBasedRecommender(weights=ContentFeatureWeights(collection=0)).fit(cleaned_dataset.movies_df)
    best_w, cal_hist = calibrate_hybrid_weights(
        cleaned_dataset.movies_df, val_q, cb, demo, k=3
    )
    assert len(cal_hist) > 0
    assert best_w.content > 0.0

    cs_report = evaluate_cold_start_stratification(cleaned_dataset.movies_df, test_q, k=3)
    assert "cold_start" in cs_report and "warm" in cs_report

    pairs_a = [([863, 10193, 949], rel) for _, rel in test_q]
    pairs_b = [([863, 949, 500], rel) for _, rel in test_q]
    boot = compute_paired_bootstrap_comparison(pairs_a, pairs_b, k=3, n_resamples=200, content_model=cb)
    assert "ndcg_at_k" in boot and "ci_95_low" in boot["ndcg_at_k"]

    # 4. Temporal ratings split & user-level holdout evaluation
    dense_ratings = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 862, "rating": 5.0, "timestamp": 100},
            {"user_id": 1, "movie_id": 863, "rating": 4.5, "timestamp": 200},
            {"user_id": 1, "movie_id": 949, "rating": 1.5, "timestamp": 300},
            {"user_id": 1, "movie_id": 500, "rating": 2.0, "timestamp": 400},
            {"user_id": 1, "movie_id": 10193, "rating": 5.0, "timestamp": 500},
            {"user_id": 2, "movie_id": 862, "rating": 4.8, "timestamp": 110},
            {"user_id": 2, "movie_id": 863, "rating": 5.0, "timestamp": 210},
            {"user_id": 2, "movie_id": 949, "rating": 1.0, "timestamp": 310},
            {"user_id": 2, "movie_id": 500, "rating": 1.5, "timestamp": 410},
            {"user_id": 2, "movie_id": 10193, "rating": 4.5, "timestamp": 510},
            {"user_id": 3, "movie_id": 949, "rating": 5.0, "timestamp": 120},
            {"user_id": 3, "movie_id": 500, "rating": 4.5, "timestamp": 220},
            {"user_id": 3, "movie_id": 862, "rating": 1.5, "timestamp": 320},
            {"user_id": 3, "movie_id": 863, "rating": 2.0, "timestamp": 420},
            {"user_id": 3, "movie_id": 680, "rating": 5.0, "timestamp": 520},
            {"user_id": 4, "movie_id": 949, "rating": 4.5, "timestamp": 130},
            {"user_id": 4, "movie_id": 500, "rating": 5.0, "timestamp": 230},
            {"user_id": 4, "movie_id": 862, "rating": 1.0, "timestamp": 330},
            {"user_id": 4, "movie_id": 863, "rating": 1.5, "timestamp": 430},
            {"user_id": 4, "movie_id": 680, "rating": 4.8, "timestamp": 530},
        ]
    )
    tr, va, te = split_ratings_train_val_test(dense_ratings, val_ratio=0.2, test_ratio=0.2)
    assert not tr.empty and not va.empty and not te.empty
    holdout = evaluate_user_level_collaborative_holdout(
        cleaned_dataset.movies_df, tr, te, k=3, max_users=10
    )
    assert holdout["evaluation_type"] == "observed_user_interactions_holdout"
    assert holdout["evaluated_users"] == 4
    assert "Collaborative Filtering" in holdout["metrics"]
    assert "paired_bootstrap_cf_vs_cb" in holdout

    # Cold-start stratification with ratings
    cs_with_ratings = evaluate_cold_start_stratification(
        cleaned_dataset.movies_df, test_q, k=3, ratings_df=dense_ratings
    )
    assert cs_with_ratings["criterion"] == "collaborative_coverage"
