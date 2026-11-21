"""Collaborative Filtering model using sparse Item-Based / SVD latent factor representations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from cinematch.utils.exceptions import DataValidationError, ModelNotTrainedError
from cinematch.utils.logging import get_logger

logger = get_logger("models.collaborative")


class CollaborativeRecommender:
    """
    Isolated Item-Based / Matrix Factorization Collaborative Filtering component.

    Design & Integrity Principles:
    1. NEVER invents or synthesizes fake user ratings.
    2. Fits exclusively on real user-item interaction logs (`user_id`, `movie_id`, `rating`).
    3. Deduplicates multiple ratings by the same (user_id, movie_id):
       - When a valid numeric `timestamp` column is provided, retains the most recent rating.
       - When no `timestamp` is present, retains the latest record in sequence order (`keep="last"`).
       - Counts item qualification strictly by distinct users (`nunique`), never raw duplicate rows.
    4. Cosine Similarity Calibration:
       - Clamps non-positive cosine similarities at zero: `max(0.0, cos(f_q, f_i))`.
       - Zero or negative correlation never generates artificial positive preference evidence.
    5. Explicitly addresses the Cold-Start Problem:
       - If no interaction dataset is available, or if a movie has fewer than `min_interactions`
         distinct user ratings, `has_item(movie_id)` returns `False` and `predict_catalog_scores`
         returns `None`, signaling the Hybrid Recommender to redistribute weights cleanly.
    """

    def __init__(self, n_components: int = 32, min_interactions: int = 2, random_state: int = 42) -> None:
        self.n_components = n_components
        self.min_interactions = min_interactions
        self.random_state = random_state

        self._is_fitted: bool = False
        self._is_available: bool = False
        self._catalog_movie_ids: list[int] = []
        self._catalog_id_to_idx: dict[int, int] = {}
        self._rated_movie_to_factor_idx: dict[int, int] = {}
        self._known_user_ids: set[int] = set()
        self._item_factors: np.ndarray | None = None
        self._interaction_counts: dict[int, int] = {}

    @property
    def is_fitted(self) -> bool:
        """True once `fit()` has been executed (including cold-start standby initialization)."""
        return self._is_fitted

    @property
    def is_available(self) -> bool:
        """True only when fitted with sufficient real user-item rating interactions."""
        return self._is_available and self._item_factors is not None

    @property
    def covered_movies_count(self) -> int:
        """Number of distinct catalog movies meeting the minimum distinct user interaction threshold."""
        return len(self._rated_movie_to_factor_idx)

    @property
    def distinct_users_count(self) -> int:
        """Number of distinct users in the active interaction matrix."""
        return len(self._known_user_ids)

    def has_item(self, movie_id: int) -> bool:
        """Return True if the movie has collaborative embeddings derived from real interactions."""
        return self.is_available and int(movie_id) in self._rated_movie_to_factor_idx

    def has_user(self, user_id: int) -> bool:
        """Return True if the user ID exists in the fitted interaction history."""
        return self.is_available and int(user_id) in self._known_user_ids

    def get_distinct_user_count(self, movie_id: int) -> int:
        """Return the number of distinct users who rated a given movie_id."""
        return self._interaction_counts.get(int(movie_id), 0)

    def _validate_and_deduplicate_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate user_id/userId, movie_id/movieId, and rating columns, and deduplicate
        multiple ratings from the same user for the same movie.
        """
        cols = set(ratings_df.columns)
        if {"user_id", "movie_id", "rating"}.issubset(cols):
            u_col, m_col, r_col = "user_id", "movie_id", "rating"
        elif {"userId", "movieId", "rating"}.issubset(cols):
            u_col, m_col, r_col = "userId", "movieId", "rating"
        else:
            raise DataValidationError(
                "ratings_df must contain columns ('user_id', 'movie_id', 'rating') "
                "or ('userId', 'movieId', 'rating')."
            )

        u_series = pd.to_numeric(ratings_df[u_col], errors="coerce")
        m_series = pd.to_numeric(ratings_df[m_col], errors="coerce")
        r_series = pd.to_numeric(ratings_df[r_col], errors="coerce")

        valid_mask = (
            u_series.notna()
            & m_series.notna()
            & r_series.notna()
            & (u_series > 0)
            & (m_series > 0)
            & (r_series >= 0.5)
            & (r_series <= 5.0)
        )

        cleaned = pd.DataFrame(
            {
                "user_id": u_series.loc[valid_mask].astype(int),
                "movie_id": m_series.loc[valid_mask].astype(int),
                "rating": r_series.loc[valid_mask].astype(float),
            }
        )

        if "timestamp" in ratings_df.columns:
            ts_series = pd.to_numeric(ratings_df.loc[valid_mask, "timestamp"], errors="coerce")
            if ts_series.notna().any():
                cleaned["timestamp"] = ts_series.fillna(-1).astype(int)
                cleaned = cleaned.sort_values(by="timestamp", ascending=True, kind="mergesort")

        # Deduplicate: keep only the latest rating per (user_id, movie_id) pair
        cleaned = cleaned.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
        if "timestamp" in cleaned.columns:
            cleaned = cleaned.drop(columns=["timestamp"])

        return cleaned.reset_index(drop=True)

    def fit(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame | None = None,
    ) -> CollaborativeRecommender:
        """
        Fit low-rank item embeddings from a real user-item ratings DataFrame.
        If `ratings_df` is None or empty, the component enters standby (cold-start) mode cleanly.
        """
        if movies_df is None or movies_df.empty:
            raise DataValidationError("Cannot initialize CollaborativeRecommender with empty catalog.")

        self._catalog_movie_ids = [int(mid) for mid in movies_df["movie_id"].tolist()]
        self._catalog_id_to_idx = {mid: idx for idx, mid in enumerate(self._catalog_movie_ids)}
        self._rated_movie_to_factor_idx = {}
        self._known_user_ids = set()
        self._item_factors = None
        self._interaction_counts = {}
        self._is_available = False
        self._is_fitted = True

        if ratings_df is None or ratings_df.empty:
            logger.info(
                "CollaborativeRecommender initialized in cold-start standby mode (no user ratings provided)."
            )
            return self

        dedup_ratings = self._validate_and_deduplicate_ratings(ratings_df)
        valid_ratings = dedup_ratings.loc[dedup_ratings["movie_id"].isin(self._catalog_id_to_idx)].copy()
        if valid_ratings.empty:
            logger.info("No overlapping ratings found for catalog movies; staying in standby mode.")
            return self

        # Count DISTINCT users per movie (never duplicate rows)
        distinct_user_counts = valid_ratings.groupby("movie_id")["user_id"].nunique()
        qualifying_ids = sorted(
            int(mid) for mid, cnt in distinct_user_counts.items() if int(cnt) >= self.min_interactions
        )
        if len(qualifying_ids) < 2:
            logger.info(
                "Fewer than 2 movies meet min_interactions=%d distinct users; staying in standby mode.",
                self.min_interactions,
            )
            return self

        filtered = valid_ratings.loc[valid_ratings["movie_id"].isin(qualifying_ids)].copy()
        unique_users = sorted(int(u) for u in filtered["user_id"].unique())
        if len(unique_users) < 2:
            logger.info("Fewer than 2 distinct users in qualifying interactions; staying in standby mode.")
            return self

        # Mean-center ratings per user to remove individual user rating bias (Adjusted Cosine)
        user_means = filtered.groupby("user_id")["rating"].transform("mean")
        filtered["centered_rating"] = filtered["rating"] - user_means
        # If a user gave constant ratings across their items, map relative to neutral midpoint (2.75)
        zero_mask = np.isclose(filtered["centered_rating"].to_numpy(), 0.0)
        if zero_mask.any():
            filtered.loc[zero_mask, "centered_rating"] = (filtered.loc[zero_mask, "rating"] - 2.75) / 2.25

        user_to_col = {u: c for c, u in enumerate(unique_users)}
        item_to_row = {mid: r for r, mid in enumerate(qualifying_ids)}

        rows = [item_to_row[int(m)] for m in filtered["movie_id"]]
        cols = [user_to_col[int(u)] for u in filtered["user_id"]]
        vals = filtered["centered_rating"].to_numpy(dtype=np.float32)

        item_user_matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(qualifying_ids), len(unique_users)),
            dtype=np.float32,
        )

        max_k = min(self.n_components, item_user_matrix.shape[0] - 1, item_user_matrix.shape[1] - 1)
        if max_k >= 2:
            svd = TruncatedSVD(n_components=max_k, random_state=self.random_state)
            factors = svd.fit_transform(item_user_matrix)
        else:
            factors = item_user_matrix.toarray()

        normalized_factors = normalize(factors, norm="l2", axis=1)

        self._rated_movie_to_factor_idx = item_to_row
        self._known_user_ids = set(unique_users)
        self._item_factors = np.asarray(normalized_factors, dtype=np.float32)
        self._interaction_counts = {int(mid): int(distinct_user_counts[mid]) for mid in qualifying_ids}
        self._is_available = True

        logger.info(
            "CollaborativeRecommender fitted on %d deduplicated interactions across %d movies and %d distinct users.",
            len(filtered),
            len(qualifying_ids),
            len(unique_users),
        )
        return self

    def predict_catalog_scores(self, movie_id: int) -> np.ndarray | None:
        """
        Return a 1D array of positive collaborative similarity scores in [0, 1] aligned with the full catalog,
        or None if Collaborative Filtering is unavailable or `movie_id` is in a cold-start state.

        Non-positive cosine similarities (<= 0.0) are truncated to 0.0 so zero or negative correlation
        is never treated as positive recommendation evidence.
        """
        if not self.has_item(movie_id) or self._item_factors is None:
            return None

        factor_idx = self._rated_movie_to_factor_idx[int(movie_id)]
        query_factor = self._item_factors[factor_idx]
        raw_sims = self._item_factors @ query_factor
        # Truncate non-positive similarities at 0.0 (no artificial positive score for cos <= 0)
        positive_sims = np.clip(raw_sims, 0.0, 1.0)

        catalog_scores = np.zeros(len(self._catalog_movie_ids), dtype=float)
        for rated_mid, r_idx in self._rated_movie_to_factor_idx.items():
            cat_idx = self._catalog_id_to_idx[rated_mid]
            catalog_scores[cat_idx] = float(positive_sims[r_idx])

        return catalog_scores

    def recommend(self, movie_id: int, movies_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
        """Return top_n collaborative recommendations with strictly positive correlation for a movie."""
        if not self._is_available:
            raise ModelNotTrainedError("CollaborativeRecommender (no interaction data available)")

        scores = self.predict_catalog_scores(movie_id)
        if scores is None:
            return pd.DataFrame(columns=movies_df.columns)

        target_cat_idx = self._catalog_id_to_idx.get(int(movie_id))
        if target_cat_idx is not None:
            scores[target_cat_idx] = -1.0

        order = np.argsort(-scores)
        top_indices = [int(i) for i in order[:top_n] if scores[i] > 0.0]

        result = movies_df.iloc[top_indices].copy()
        result["collaborative_score"] = np.round(scores[top_indices], 4)
        return result.reset_index(drop=True)
