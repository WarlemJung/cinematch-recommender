"""Sparse Content-Based Filtering model with weighted metadata features and on-demand similarity."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from cinematch.data.preprocessing import normalize_title
from cinematch.utils.exceptions import (
    DataValidationError,
    InvalidParameterError,
    ModelNotTrainedError,
    MovieNotFoundError,
)
from cinematch.utils.logging import get_logger

logger = get_logger("models.content_based")

_TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9]+")


def _slugify_token(prefix: str, value: str) -> str:
    """Convert a structured metadata entity (e.g. 'Science Fiction') into a single weighted token."""
    slug = _TOKEN_CLEAN_RE.sub("_", value.strip().lower()).strip("_")
    return f"{prefix}_{slug}" if slug else ""


@dataclass(frozen=True)
class ContentFeatureWeights:
    """
    Configurable feature repetition weights for TF-IDF document construction.

    Structured attributes are converted into namespace-prefixed tokens (e.g., `genre_sci_fi`,
    `collection_toy_story`, `studio_pixar`) before repetition so they never collide with
    free-text words from the synopsis (`overview`) or `tagline`.
    """

    overview: int = 2
    genres: int = 3
    tagline: int = 2
    collection: int = 2
    production: int = 1
    language: int = 1


class ContentBasedRecommender:
    """
    Scalable Content-Based Recommender using sparse TF-IDF representations and on-demand
    cosine similarity (`linear_kernel`).

    Computational Complexity:
    - Training Time: O(N * L_avg), where N is the catalog size and L_avg is average tokens per movie.
    - Memory Footprint: O(nnz), storing only non-zero entries of the sparse CSR TF-IDF matrix
      (~25-35 MB for 45,000 movies, compared to 16.5 GB for a dense N x N matrix).
    - Query Time: O(nnz_query * d_avg + N log(top_n)), computing the dot product of a single
      1 x V sparse row against the N x V sparse matrix on demand (< 15 ms for 45,000 movies).
    """

    def __init__(
        self,
        weights: ContentFeatureWeights | None = None,
        max_features: int = 25000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.weights = weights or ContentFeatureWeights()
        self.max_features = max_features
        self.ngram_range = ngram_range

        self._df: pd.DataFrame | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: csr_matrix | None = None
        self._id_to_idx: dict[int, int] = {}
        self._norm_title_to_indices: dict[str, list[int]] = {}
        self._exact_title_to_indices: dict[str, list[int]] = {}

    @property
    def is_fitted(self) -> bool:
        return self._df is not None and self._tfidf_matrix is not None

    @property
    def tfidf_matrix(self) -> csr_matrix:
        if self._tfidf_matrix is None:
            raise ModelNotTrainedError("ContentBasedRecommender")
        return self._tfidf_matrix

    @property
    def memory_bytes(self) -> int:
        """Return exact memory consumed by the sparse TF-IDF matrix in bytes."""
        if self._tfidf_matrix is None:
            return 0
        return int(
            self._tfidf_matrix.data.nbytes
            + self._tfidf_matrix.indptr.nbytes
            + self._tfidf_matrix.indices.nbytes
        )

    def _build_weighted_document(self, row: pd.Series) -> str:
        """Construct a structured, weighted text representation for a single movie row."""
        parts: list[str] = []

        overview = str(row.get("overview", "") or "").strip()
        if overview and self.weights.overview > 0:
            parts.extend([overview] * self.weights.overview)

        tagline = str(row.get("tagline", "") or "").strip()
        if tagline and self.weights.tagline > 0:
            parts.extend([tagline] * self.weights.tagline)

        genres_list = row.get("genres_list")
        if isinstance(genres_list, list) and self.weights.genres > 0:
            genre_tokens = [_slugify_token("genre", g) for g in genres_list if g]
            genre_str = " ".join(t for t in genre_tokens if t)
            if genre_str:
                parts.extend([genre_str] * self.weights.genres)

        collection_name = str(row.get("collection_name", "") or "").strip()
        if collection_name and self.weights.collection > 0:
            col_token = _slugify_token("collection", collection_name)
            if col_token:
                parts.extend([col_token] * self.weights.collection)

        production_list = row.get("production_list")
        if isinstance(production_list, list) and self.weights.production > 0:
            prod_tokens = [_slugify_token("studio", p) for p in production_list if p]
            prod_str = " ".join(t for t in prod_tokens if t)
            if prod_str:
                parts.extend([prod_str] * self.weights.production)

        lang = str(row.get("original_language", "") or "").strip()
        if lang and self.weights.language > 0:
            lang_token = _slugify_token("lang", lang)
            if lang_token:
                parts.extend([lang_token] * self.weights.language)

        # Fallback for movies without synopsis or metadata: use normalized title tokens
        if not parts:
            title_fallback = normalize_title(str(row.get("title", "")))
            return title_fallback if title_fallback else "unknown_movie"

        return " ".join(parts)

    def fit(self, movies_df: pd.DataFrame) -> ContentBasedRecommender:
        """Build the sparse TF-IDF matrix and lookup indices without materializing a dense similarity matrix."""
        if movies_df is None or movies_df.empty:
            raise DataValidationError("Cannot fit ContentBasedRecommender on an empty DataFrame.")

        required = {"movie_id", "title", "overview"}
        if not required.issubset(set(movies_df.columns)):
            raise DataValidationError(f"ContentBasedRecommender requires columns: {sorted(required)}")

        df_copy = movies_df.copy().reset_index(drop=True)
        if "normalized_title" not in df_copy.columns:
            df_copy["normalized_title"] = df_copy["title"].apply(normalize_title)

        records = df_copy.to_dict(orient="records")
        corpus = [self._build_weighted_document(row) for row in records]  # type: ignore[arg-type]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            dtype=np.float32,
        )
        matrix = csr_matrix(vectorizer.fit_transform(corpus))

        id_to_idx: dict[int, int] = {}
        norm_title_map: dict[str, list[int]] = {}
        exact_title_map: dict[str, list[int]] = {}

        vote_counts = (
            pd.to_numeric(df_copy["vote_count"], errors="coerce").fillna(0).to_numpy()
            if "vote_count" in df_copy.columns
            else np.zeros(len(df_copy))
        )

        for row_idx, row in enumerate(records):
            m_id = int(row["movie_id"])
            id_to_idx[m_id] = row_idx

            norm_t = str(row["normalized_title"])
            norm_title_map.setdefault(norm_t, []).append(row_idx)

            exact_t = str(row["title"]).strip().lower()
            exact_title_map.setdefault(exact_t, []).append(row_idx)

        for key, idx_list in norm_title_map.items():
            if len(idx_list) > 1:
                norm_title_map[key] = sorted(idx_list, key=lambda i: float(vote_counts[i]), reverse=True)

        for key, idx_list in exact_title_map.items():
            if len(idx_list) > 1:
                exact_title_map[key] = sorted(idx_list, key=lambda i: float(vote_counts[i]), reverse=True)

        self._df = df_copy
        self._vectorizer = vectorizer
        self._tfidf_matrix = matrix
        self._id_to_idx = id_to_idx
        self._norm_title_to_indices = norm_title_map
        self._exact_title_to_indices = exact_title_map

        logger.info(
            "ContentBasedRecommender fitted: shape=%s, nnz=%d, memory=%.2f MB",
            matrix.shape,
            matrix.nnz,
            self.memory_bytes / (1024 * 1024),
        )
        return self

    def resolve_movie_index(self, identifier: int | str) -> int:
        """
        Resolve a movie_id (int) or movie title (str) to its unique catalog row index.
        Handles duplicate titles deterministically by selecting the version with highest vote_count.
        """
        if not self.is_fitted or self._df is None:
            raise ModelNotTrainedError("ContentBasedRecommender")

        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.strip().isdigit()):
            movie_id = int(identifier)
            if movie_id in self._id_to_idx:
                return self._id_to_idx[movie_id]
            if isinstance(identifier, int):
                raise MovieNotFoundError(identifier)

        if isinstance(identifier, str):
            raw_query = identifier.strip()
            if not raw_query:
                raise InvalidParameterError("Movie title query cannot be empty.")

            lowered = raw_query.lower()
            if lowered in self._exact_title_to_indices:
                return self._exact_title_to_indices[lowered][0]

            norm_q = normalize_title(raw_query)
            if norm_q and norm_q in self._norm_title_to_indices:
                return self._norm_title_to_indices[norm_q][0]

            # Substring fallback matching most popular movie containing normalized query
            if norm_q:
                matches = self._df.loc[
                    self._df["normalized_title"].str.contains(re.escape(norm_q), na=False)
                ]
                if not matches.empty:
                    best_match = matches.sort_values(by="vote_count", ascending=False).iloc[0]
                    return self._id_to_idx[int(best_match["movie_id"])]

        raise MovieNotFoundError(identifier)

    def compute_similarity_for_index(self, row_idx: int) -> np.ndarray:
        """
        Compute the 1D cosine similarity vector between a single movie row and all catalog items on demand.
        Complexity: O(nnz_query * d_avg) time and O(N) auxiliary memory.
        """
        if not self.is_fitted or self._tfidf_matrix is None:
            raise ModelNotTrainedError("ContentBasedRecommender")

        if row_idx < 0 or row_idx >= self._tfidf_matrix.shape[0]:
            raise IndexError(f"Row index {row_idx} is out of bounds.")

        query_vec = self._tfidf_matrix[row_idx : row_idx + 1]
        sim_vector = linear_kernel(query_vec, self._tfidf_matrix).ravel()
        return np.clip(np.asarray(sim_vector, dtype=float), 0.0, 1.0)

    def recommend(
        self,
        identifier: int | str,
        top_n: int = 12,
    ) -> pd.DataFrame:
        """
        Return top_n most similar movies for a given movie_id or title, excluding the movie itself.
        """
        if not self.is_fitted or self._df is None:
            raise ModelNotTrainedError("ContentBasedRecommender")
        if top_n < 1:
            raise InvalidParameterError("top_n must be at least 1.")

        target_idx = self.resolve_movie_index(identifier)
        sim_scores = self.compute_similarity_for_index(target_idx)

        # Exclude target movie itself
        sim_scores[target_idx] = -1.0

        # Rank by similarity descending, tie-breaking by vote_count descending
        vote_counts = (
            self._df["vote_count"].to_numpy(dtype=float)
            if "vote_count" in self._df.columns
            else np.zeros(len(self._df))
        )
        order = np.lexsort((-vote_counts, -sim_scores))
        top_indices = [int(i) for i in order[:top_n] if sim_scores[i] >= 0.0]

        result_df = self._df.iloc[top_indices].copy()
        result_df["similarity_score"] = np.round(sim_scores[top_indices], 4)
        return result_df.reset_index(drop=True)

    def to_records(self, identifier: int | str, top_n: int = 12) -> list[dict[str, Any]]:
        """Return content-based recommendations as a list of metadata dictionaries."""
        return self.recommend(identifier=identifier, top_n=top_n).to_dict(orient="records")
