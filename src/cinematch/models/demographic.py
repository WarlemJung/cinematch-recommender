"""Demographic Filtering model based on the IMDB Weighted Rating formula."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from cinematch.utils.exceptions import (
    DataValidationError,
    InvalidParameterError,
    ModelNotTrainedError,
)


def compute_weighted_rating(
    vote_count: float | np.ndarray | pd.Series,
    vote_average: float | np.ndarray | pd.Series,
    min_votes_m: float,
    mean_vote_c: float,
) -> float | np.ndarray | pd.Series:
    """
    Calculate the IMDB Weighted Rating (WR):

        WR = (v / (v + m)) * R + (m / (v + m)) * C

    Where:
        R = average rating for the movie (vote_average)
        v = number of votes for the movie (vote_count)
        m = minimum votes required to be listed
        C = the mean vote across the whole catalog
    """
    if min_votes_m < 0:
        raise InvalidParameterError("Minimum vote threshold 'm' must be non-negative.")

    denominator = vote_count + min_votes_m
    if isinstance(denominator, (int, float, np.number)):
        if denominator == 0:
            return float(mean_vote_c)
        return float((vote_count / denominator) * vote_average + (min_votes_m / denominator) * mean_vote_c)

    safe_denom = np.where(denominator == 0, 1.0, denominator)
    wr = (vote_count / safe_denom) * vote_average + (min_votes_m / safe_denom) * mean_vote_c
    return np.where(denominator == 0, mean_vote_c, wr)


@dataclass(frozen=True)
class DemographicParameters:
    """Fitted parameters for the Demographic Recommender."""

    mean_vote_c: float
    min_votes_m: float
    quantile: float
    qualifying_count: int
    total_count: int


class DemographicRecommender:
    """
    Recommends universally acclaimed and popular movies using IMDB's Bayesian Weighted Rating.
    """

    def __init__(self, quantile: float = 0.90, min_votes_override: int | None = None) -> None:
        if not (0.0 <= quantile <= 1.0):
            raise InvalidParameterError("quantile must be between 0.0 and 1.0.")
        if min_votes_override is not None and min_votes_override < 0:
            raise InvalidParameterError("min_votes_override must be non-negative.")

        self.quantile = quantile
        self.min_votes_override = min_votes_override
        self._df: pd.DataFrame | None = None
        self._params: DemographicParameters | None = None
        self._all_scores: np.ndarray | None = None
        self._normalized_scores: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._df is not None and self._params is not None

    @property
    def parameters(self) -> DemographicParameters:
        if self._params is None:
            raise ModelNotTrainedError("DemographicRecommender")
        return self._params

    def fit(self, movies_df: pd.DataFrame) -> DemographicRecommender:
        """Compute global mean C, vote threshold m, and weighted ratings for the catalog."""
        if movies_df is None or movies_df.empty:
            raise DataValidationError("Cannot fit DemographicRecommender on an empty DataFrame.")

        required = {"movie_id", "title", "vote_average", "vote_count"}
        if not required.issubset(set(movies_df.columns)):
            raise DataValidationError(f"DemographicRecommender requires columns: {sorted(required)}")

        v = pd.to_numeric(movies_df["vote_count"], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(movies_df["vote_average"], errors="coerce").to_numpy(dtype=float)

        if np.isnan(v).any() or np.isnan(r).any() or (v < 0).any():
            raise DataValidationError("Dataset contains NaN or negative values in vote_count/vote_average.")

        mean_c = float(np.mean(r))
        if self.min_votes_override is not None:
            min_m = float(self.min_votes_override)
        else:
            min_m = float(np.quantile(v, self.quantile))

        scores = np.asarray(compute_weighted_rating(v, r, min_m, mean_c), dtype=float)

        min_s = float(np.min(scores))
        max_s = float(np.max(scores))
        if max_s > min_s:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.full_like(scores, 0.5)

        df_copy = movies_df.copy().reset_index(drop=True)
        df_copy["score"] = np.round(scores, 4)
        df_copy["normalized_demographic_score"] = np.round(norm_scores, 4)
        df_copy["qualifies_demographic"] = v >= min_m

        self._df = df_copy
        self._all_scores = scores
        self._normalized_scores = norm_scores
        self._params = DemographicParameters(
            mean_vote_c=round(mean_c, 4),
            min_votes_m=round(min_m, 2),
            quantile=self.quantile,
            qualifying_count=int(np.sum(v >= min_m)),
            total_count=len(df_copy),
        )
        return self

    def get_normalized_scores(self) -> np.ndarray:
        """Return array of [0, 1] normalized weighted ratings aligned with catalog row indices."""
        if not self.is_fitted or self._normalized_scores is None:
            raise ModelNotTrainedError("DemographicRecommender")
        return self._normalized_scores

    def get_raw_scores(self) -> np.ndarray:
        """Return array of raw weighted ratings aligned with catalog row indices."""
        if not self.is_fitted or self._all_scores is None:
            raise ModelNotTrainedError("DemographicRecommender")
        return self._all_scores

    def recommend(
        self,
        top_n: int = 12,
        genre: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_rating: float | None = None,
        strict_qualification: bool = True,
    ) -> pd.DataFrame:
        """Return top_n movies ranked by Weighted Rating, with optional filtering."""
        if not self.is_fitted or self._df is None:
            raise ModelNotTrainedError("DemographicRecommender")
        if top_n < 1:
            raise InvalidParameterError("top_n must be at least 1.")

        candidates = self._df
        if strict_qualification:
            qualified = candidates.loc[candidates["qualifies_demographic"]]
            if len(qualified) >= min(top_n, len(candidates)):
                candidates = qualified

        if genre:
            genre_lower = genre.strip().lower()
            candidates = candidates.loc[
                candidates["genres_list"].apply(
                    lambda g_list: any(g.lower() == genre_lower for g in g_list)
                )
            ]

        if min_year is not None and "release_year" in candidates.columns:
            candidates = candidates.loc[candidates["release_year"] >= min_year]

        if max_year is not None and "release_year" in candidates.columns:
            candidates = candidates.loc[
                (candidates["release_year"] > 0) & (candidates["release_year"] <= max_year)
            ]

        if min_rating is not None:
            candidates = candidates.loc[candidates["vote_average"] >= min_rating]

        ranked = candidates.sort_values(
            by=["score", "vote_count", "popularity"],
            ascending=[False, False, False],
        )
        return ranked.head(top_n).copy()

    def to_records(self, top_n: int = 12, **filters: Any) -> list[dict[str, Any]]:
        """Return demographic recommendations as a list of dictionaries."""
        df_top = self.recommend(top_n=top_n, **filters)
        return df_top.to_dict(orient="records")
