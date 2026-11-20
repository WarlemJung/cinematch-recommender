"""Unit tests for Demographic, Content-Based, Collaborative, and Hybrid recommendation models."""

from __future__ import annotations

import math
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
    # WR = (v / (v + m)) * R + (m / (v + m)) * C
    v, r, m, c = 1000.0, 8.0, 1000.0, 6.0
    expected = (1000.0 / 2000.0) * 8.0 + (1000.0 / 2000.0) * 6.0
    assert math.isclose(float(compute_weighted_rating(v, r, m, c)), expected, rel_tol=1e-9)

    # Zero denominator edge case
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
    # Pulp Fiction has highest rating (8.3) and votes (8670), so it must rank #1
    assert int(top_df.iloc[0]["movie_id"]) == 680

    # Filter by genre Animation
    anim_df = model.recommend(top_n=5, genre="Animation")
    assert all("Animation" in g_list for g_list in anim_df["genres_list"])

    # Empty dataset error
    with pytest.raises(DataValidationError):
        DemographicRecommender().fit(pd.DataFrame())


def test_content_based_recommender_uses_sparse_matrix_and_handles_duplicate_titles(
    cleaned_dataset: LoadedDataset,
) -> None:
    model = ContentBasedRecommender()
    model.fit(cleaned_dataset.movies_df)

    # Verify sparse CSR matrix and absence of dense quadratic cosine_sim matrix
    assert isinstance(model.tfidf_matrix, csr_matrix)
    assert not hasattr(model, "cosine_sim")
    assert model.memory_bytes > 0

    # Query for 'Toy Story' (862) should return Toy Story 2 & Toy Story 3 at the top and exclude Toy Story itself
    recs = model.recommend("Toy Story", top_n=3)
    rec_ids = recs["movie_id"].tolist()
    assert 862 not in rec_ids
    assert 863 in rec_ids[:2]
    assert 10193 in rec_ids[:2]

    # Duplicate title 'Heat': two entries exist (1995 with 1886 votes, 1972 with 18 votes).
    # Must deterministically resolve to movie_id 949 (1995) without crashing!
    heat_idx = model.resolve_movie_index("Heat")
    resolved_id = int(cleaned_dataset.movies_df.iloc[heat_idx]["movie_id"])
    assert resolved_id == 949

    heat_recs = model.recommend("Heat", top_n=3)
    assert 949 not in heat_recs["movie_id"].tolist()

    # Movie without synopsis (movie_id 777) should still work without crashing
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
    assert not collab_standby.is_available
    assert not collab_standby.has_item(862)
    assert collab_standby.predict_catalog_scores(862) is None

    # 2. Fitted with real user-item interaction matrix
    collab_active = CollaborativeRecommender(n_components=4, min_interactions=2).fit(
        cleaned_dataset.movies_df, ratings_df=synthetic_ratings_dataframe
    )
    assert collab_active.is_available
    assert collab_active.has_item(862)
    assert not collab_active.has_item(777)  # Unrated movie is in cold-start

    scores_862 = collab_active.predict_catalog_scores(862)
    assert scores_862 is not None
    assert len(scores_862) == len(cleaned_dataset.movies_df)

    recs_df = collab_active.recommend(862, cleaned_dataset.movies_df, top_n=2)
    rec_ids = recs_df["movie_id"].tolist()
    assert 862 not in rec_ids
    # Toy Story 2 (863) and Toy Story 3 (10193) are co-liked by users 1..3
    assert set(rec_ids) == {863, 10193}


def test_hybrid_recommender_weight_adjustment_and_explanations(
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
