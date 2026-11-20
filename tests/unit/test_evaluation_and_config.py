"""Unit tests for quantitative evaluation metrics, configuration, services, and legacy adapter."""

from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import pytest

from cinematch.config import AppConfig
from cinematch.data.loader import LoadedDataset
from cinematch.models.content_based import ContentBasedRecommender
from cinematch.models.evaluation import (
    average_precision_at_k,
    catalog_coverage,
    evaluate_catalog_models,
    intra_list_similarity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recommendation_diversity,
)
from cinematch.services.movie_service import MovieService
from cinematch.utils.exceptions import ConfigurationError, MovieNotFoundError
from recommender import MovieRecommenderSystem


def test_ranking_evaluation_metrics_mathematics() -> None:
    recommended = [10, 20, 30, 40, 50]
    relevant = {10, 30, 60}

    # Top-5 hits: 10 (pos 1) and 30 (pos 3) -> 2 hits
    assert math.isclose(precision_at_k(recommended, relevant, k=5), 2 / 5)
    assert math.isclose(recall_at_k(recommended, relevant, k=5), 2 / 3)

    # AP@5: (1/1 + 2/3) / min(3, 5) = (5/3) / 3 = 5/9
    assert math.isclose(average_precision_at_k(recommended, relevant, k=5), 5 / 9)

    # NDCG@5: DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG for 3 relevant items = 1/log2(2) + 1/log2(3) + 1/log2(4)
    expected_idcg = 1.0 + (1.0 / math.log2(3)) + 0.5
    assert math.isclose(ndcg_at_k(recommended, relevant, k=5), 1.5 / expected_idcg)

    # Coverage
    lists = [[10, 20], [20, 30]]
    assert math.isclose(catalog_coverage(lists, catalog_size=10), 3 / 10)


def test_diversity_and_catalog_evaluation_pipeline(cleaned_dataset: LoadedDataset) -> None:
    content = ContentBasedRecommender().fit(cleaned_dataset.movies_df)
    rec_lists = [[862, 863, 10193], [949, 500, 680]]

    ils = intra_list_similarity(rec_lists, content)
    div = recommendation_diversity(rec_lists, content)
    assert 0.0 <= ils <= 1.0
    assert math.isclose(ils + div, 1.0, rel_tol=1e-6)

    eval_summary = evaluate_catalog_models(cleaned_dataset.movies_df, k=3, max_queries=10)
    assert "Demographic Filtering" in eval_summary
    assert "Content-Based Filtering" in eval_summary
    assert "Hybrid Recommendation" in eval_summary
    assert eval_summary["Content-Based Filtering"].precision_at_k > 0.0


def test_app_config_validation_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="DEBUG mode cannot be enabled"):
        AppConfig(env="production", debug=True).validate()

    with pytest.raises(ConfigurationError, match="demographic_quantile"):
        AppConfig(demographic_quantile=1.5).validate()

    with pytest.raises(ConfigurationError, match="non-negative"):
        AppConfig(hybrid_weight_content=-0.5).validate()

    monkeypatch.setenv("CINEMATCH_ENV", "development")
    monkeypatch.setenv("CINEMATCH_DEBUG", "true")
    monkeypatch.setenv("CINEMATCH_SAMPLE_SIZE", "500")
    monkeypatch.setenv("CINEMATCH_LOG_JSON", "true")
    cfg = AppConfig.from_env()
    assert cfg.env == "development"
    assert cfg.debug is True
    assert cfg.sample_size == 500
    assert cfg.log_json is True


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

    # Search for duplicate title 'Heat' returns both entries ranked by vote_count
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

    # Duplicate title 'Heat' works without crashing
    heat_recs = legacy.get_content_recommendations("Heat", top_n=2)
    assert len(heat_recs) == 2
