"""Unit tests for Demographic, Content-Based, Collaborative, and Hybrid recommendation models."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cinematch.data.loader import LoadedDataset
from cinematch.models.collaborative import CollaborativeRecommender
from cinematch.models.content_based import ContentBasedRecommender
from cinematch.models.demographic import DemographicRecommender, compute_weighted_rating
from cinematch.models.hybrid import HybridRecommender, HybridWeights
from cinematch.utils.exceptions import (
    DataValidationError,
    InvalidParameterError,
    ModelNotTrainedError,
    MovieNotFoundError,
)


def test_weighted_rating_mathematical_formula() -> None:
    v, r, m, c = 1000.0, 8.0, 1000.0, 6.0
    expected = (1000.0 / 2000.0) * 8.0 + (1000.0 / 2000.0) * 6.0
    assert math.isclose(float(compute_weighted_rating(v, r, m, c)), expected, rel_tol=1e-9)

    assert math.isclose(float(compute_weighted_rating(0.0, 8.0, 0.0, 6.5)), 6.5, rel_tol=1e-9)

    with pytest.raises(InvalidParameterError):
        compute_weighted_rating(10.0, 7.0, -5.0, 6.0)


def test_demographic_recommender_fitting_and_filtering(cleaned_dataset: LoadedDataset) -> None:
    model = DemographicRecommender(quantile=0.50)
    with pytest.raises(ModelNotTrainedError):
        model.recommend(top_n=3)

    model.fit(cleaned_dataset.movies_df)
    assert model.is_fitted
    assert model.parameters.total_count == len(cleaned_dataset.movies_df)

    top_df = model.recommend(top_n=3)
    assert len(top_df) == 3
    assert int(top_df.iloc[0]["movie_id"]) == 680

    anim_df = model.recommend(top_n=5, genre="Animation")
    assert all("Animation" in g_list for g_list in anim_df["genres_list"])

    with pytest.raises(DataValidationError):
        DemographicRecommender().fit(pd.DataFrame())


def test_content_based_recommender_uses_sparse_matrix_and_handles_duplicate_titles(
    cleaned_dataset: LoadedDataset,
) -> None:
    model = ContentBasedRecommender()
    model.fit(cleaned_dataset.movies_df)

    assert isinstance(model.tfidf_matrix, csr_matrix)
    assert not hasattr(model, "cosine_sim")
    assert model.memory_bytes > 0

    recs = model.recommend("Toy Story", top_n=3)
    rec_ids = recs["movie_id"].tolist()
    assert 862 not in rec_ids
    assert 863 in rec_ids[:2]
    assert 10193 in rec_ids[:2]

    heat_idx = model.resolve_movie_index("Heat")
    resolved_id = int(cleaned_dataset.movies_df.iloc[heat_idx]["movie_id"])
    assert resolved_id == 949

    heat_recs = model.recommend("Heat", top_n=3)
    assert 949 not in heat_recs["movie_id"].tolist()

    no_synopsis_recs = model.recommend(777, top_n=3)
    assert len(no_synopsis_recs) == 3

    with pytest.raises(MovieNotFoundError):
        model.recommend("NonExistentMovieTitleXYZ123")


def test_collaborative_recommender_standby_and_real_interactions(
    cleaned_dataset: LoadedDataset,
    synthetic_ratings_dataframe: pd.DataFrame,
) -> None:
    # 1. Cold-start standby when no ratings exist
    collab_standby = CollaborativeRecommender().fit(cleaned_dataset.movies_df, ratings_df=None)
    assert collab_standby.is_fitted
    assert not collab_standby.is_available
    assert not collab_standby.has_item(862)
    assert not collab_standby.has_user(1)
    assert collab_standby.predict_catalog_scores(862) is None

    # 2. Fitted with real user-item interaction matrix
    collab_active = CollaborativeRecommender(n_components=4, min_interactions=2).fit(
        cleaned_dataset.movies_df, ratings_df=synthetic_ratings_dataframe
    )
    assert collab_active.is_fitted
    assert collab_active.is_available
    assert collab_active.has_item(862)
    assert not collab_active.has_item(777)
    assert collab_active.has_user(1)
    assert not collab_active.has_user(9999)

    scores_862 = collab_active.predict_catalog_scores(862)
    assert scores_862 is not None
    assert len(scores_862) == len(cleaned_dataset.movies_df)

    # Verify non-positive cosine similarities are clamped to 0.0 (no artificial 0.5 baseline!)
    # Crime movies (949, 500, 680) were disliked by Toy Story fans (users 1..3) and loved by users 5..7
    crime_idx = cleaned_dataset.movies_df.index[cleaned_dataset.movies_df["movie_id"] == 949][0]
    assert scores_862[crime_idx] == 0.0

    # Unrated movie (777) must also have 0.0 score
    unrated_idx = cleaned_dataset.movies_df.index[cleaned_dataset.movies_df["movie_id"] == 777][0]
    assert scores_862[unrated_idx] == 0.0

    recs_df = collab_active.recommend(862, cleaned_dataset.movies_df, top_n=2)
    rec_ids = recs_df["movie_id"].tolist()
    assert 862 not in rec_ids
    assert set(rec_ids) == {863, 10193}


def test_collaborative_deduplication_distinct_users_and_timestamp_policy(
    cleaned_dataset: LoadedDataset,
) -> None:
    # Single user (user_id=1) rates movies 862 and 863 five times each.
    # Because min_interactions=2 requires 2 DISTINCT users, model must stay in standby!
    duplicate_single_user_logs = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 862, "rating": 5.0, "timestamp": 100},
            {"user_id": 1, "movie_id": 862, "rating": 4.0, "timestamp": 200},
            {"user_id": 1, "movie_id": 862, "rating": 5.0, "timestamp": 300},
            {"user_id": 1, "movie_id": 863, "rating": 4.5, "timestamp": 100},
            {"user_id": 1, "movie_id": 863, "rating": 5.0, "timestamp": 200},
        ]
    )
    collab_single = CollaborativeRecommender(min_interactions=2).fit(
        cleaned_dataset.movies_df, ratings_df=duplicate_single_user_logs
    )
    assert not collab_single.is_available
    assert collab_single.covered_movies_count == 0

    # Now add a second distinct user (user_id=2), plus duplicate ratings for user 1 where the
    # earlier timestamp had rating=1.0 and the newer timestamp (999) has rating=5.0.
    multi_user_with_timestamps = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 862, "rating": 1.0, "timestamp": 100},
            {"user_id": 1, "movie_id": 862, "rating": 5.0, "timestamp": 999},  # Latest must win
            {"user_id": 1, "movie_id": 863, "rating": 4.8, "timestamp": 500},
            {"user_id": 1, "movie_id": 949, "rating": 1.5, "timestamp": 600},
            {"user_id": 2, "movie_id": 862, "rating": 4.9, "timestamp": 200},
            {"user_id": 2, "movie_id": 863, "rating": 5.0, "timestamp": 300},
            {"user_id": 2, "movie_id": 949, "rating": 1.2, "timestamp": 400},
        ]
    )
    collab_dedup = CollaborativeRecommender(n_components=2, min_interactions=2).fit(
        cleaned_dataset.movies_df, ratings_df=multi_user_with_timestamps
    )
    assert collab_dedup.is_available
    assert collab_dedup.get_distinct_user_count(862) == 2
    assert collab_dedup.distinct_users_count == 2

    scores = collab_dedup.predict_catalog_scores(862)
    assert scores is not None
    idx_863 = cleaned_dataset.movies_df.index[cleaned_dataset.movies_df["movie_id"] == 863][0]
    idx_949 = cleaned_dataset.movies_df.index[cleaned_dataset.movies_df["movie_id"] == 949][0]
    # Positive correlation between 862 and 863 (> 0.8), negative correlation with 949 clamped to 0.0
    assert scores[idx_863] > 0.8
    assert scores[idx_949] == 0.0
    assert np.all(scores >= 0.0)


def test_hybrid_recommender_weight_adjustment_gating_and_explanations(
    cleaned_dataset: LoadedDataset,
    synthetic_ratings_dataframe: pd.DataFrame,
) -> None:
    demo = DemographicRecommender(quantile=0.5).fit(cleaned_dataset.movies_df)
    content = ContentBasedRecommender().fit(cleaned_dataset.movies_df)
    collab = CollaborativeRecommender(n_components=4, min_interactions=2).fit(
        cleaned_dataset.movies_df, ratings_df=synthetic_ratings_dataframe
    )

    hybrid = HybridRecommender(
        content_model=content,
        demographic_model=demo,
        collaborative_model=collab,
        weights=HybridWeights(content=0.60, demographic=0.20, collaborative=0.20),
    ).fit(cleaned_dataset.movies_df, ratings_df=synthetic_ratings_dataframe)

    # Movie 862 has collaborative interactions -> all 3 weights active
    items_with_collab = hybrid.recommend(862, top_n=3)
    assert len(items_with_collab) == 3
    first = items_with_collab[0]
    assert first.active_weights["collaborative"] > 0.0
    assert first.individual_scores["collaborative"] is not None
    assert "Recomendado porque" in first.explanation
    assert "Toy Story Collection" in first.explanation
    # Top-2 recommendations for Toy Story must be Toy Story 2 & 3, not Pulp Fiction
    top_2_ids = {items_with_collab[0].movie_id, items_with_collab[1].movie_id}
    assert top_2_ids == {863, 10193}

    # Movie 777 has no collaborative interactions -> collaborative weight redistributed to 0.0
    items_cold_start = hybrid.recommend(777, top_n=3)
    assert len(items_cold_start) == 3
    cold_first = items_cold_start[0]
    assert cold_first.active_weights["collaborative"] == 0.0
    assert cold_first.individual_scores["collaborative"] is None
    assert math.isclose(
        cold_first.active_weights["content"] + cold_first.active_weights["demographic"],
        1.0,
        rel_tol=1e-3,
    )
