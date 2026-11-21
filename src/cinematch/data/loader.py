"""Dataset loader for movie metadata and optional user ratings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from cinematch.data.preprocessing import preprocess_movies_dataframe
from cinematch.data.validation import DataQualityReport
from cinematch.utils.exceptions import DatasetNotFoundError, DataValidationError
from cinematch.utils.logging import get_logger

logger = get_logger("data.loader")


@dataclass(frozen=True)
class LoadedDataset:
    """Container holding the cleaned movies DataFrame, quality report, and optional ratings."""

    movies_df: pd.DataFrame
    quality_report: DataQualityReport
    ratings_df: pd.DataFrame | None = None


class MovieDataLoader:
    """Loads and validates movie metadata and optional interaction datasets from disk."""

    def __init__(
        self,
        dataset_path: Path | str,
        ratings_path: Path | str | None = None,
        sample_size: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.ratings_path = Path(ratings_path) if ratings_path else None
        self.sample_size = sample_size
        self.random_state = random_state

    def load(self) -> LoadedDataset:
        """Read CSV files, validate schema and permissions, run preprocessing, and return LoadedDataset."""
        try:
            exists = self.dataset_path.exists()
        except PermissionError as exc:
            raise DataValidationError(
                f"Permission denied when accessing dataset path '{self.dataset_path}'."
            ) from exc

        if not exists:
            raise DatasetNotFoundError(str(self.dataset_path))

        if self.dataset_path.is_dir():
            raise DataValidationError(
                f"Dataset path '{self.dataset_path}' is a directory, not a CSV file. "
                "Verify that data/movies_metadata.csv exists on the host before mounting."
            )

        try:
            file_size = self.dataset_path.stat().st_size
        except PermissionError as exc:
            raise DataValidationError(
                f"Permission denied when inspecting dataset file '{self.dataset_path}'."
            ) from exc

        if file_size == 0:
            raise DataValidationError(f"Dataset file '{self.dataset_path}' is empty (0 bytes).")

        logger.info("Loading movie metadata from '%s'", self.dataset_path)
        try:
            raw_df = pd.read_csv(self.dataset_path, low_memory=False)
        except PermissionError as exc:
            raise DataValidationError(
                f"Permission denied when reading dataset file '{self.dataset_path}'."
            ) from exc
        except pd.errors.EmptyDataError as exc:
            raise DataValidationError(
                f"Dataset file '{self.dataset_path}' contains no parseable CSV data."
            ) from exc
        except Exception as exc:
            raise DataValidationError(f"Failed to parse CSV file '{self.dataset_path}': {exc}") from exc

        movies_df, report = preprocess_movies_dataframe(
            raw_df,
            sample_size=self.sample_size,
            random_state=self.random_state,
        )
        logger.info(
            "Dataset preprocessed: %d valid movies (%d active after sampling, %d genres).",
            report.valid_rows,
            report.sampled_rows,
            report.unique_genres_count,
        )

        ratings_df = self._load_optional_ratings(set(movies_df["movie_id"].tolist()))
        return LoadedDataset(
            movies_df=movies_df,
            quality_report=report,
            ratings_df=ratings_df,
        )

    def _load_optional_ratings(self, valid_movie_ids: set[int]) -> pd.DataFrame | None:
        """
        Load and deduplicate real user ratings if a valid ratings CSV is configured and exists.

        Deduplication policy for multiple ratings by the same (user_id, movie_id):
        - If a valid numeric `timestamp` column is present, records are ordered by `timestamp`
          ascending and the most recent rating (`keep="last"`) is retained.
        - Otherwise, the latest entry in file order (`keep="last"`) is retained.
        """
        if self.ratings_path is None or not self.ratings_path.exists() or self.ratings_path.is_dir():
            return None

        logger.info("Loading optional user ratings from '%s'", self.ratings_path)
        try:
            raw_ratings = pd.read_csv(self.ratings_path, low_memory=False)
        except Exception as exc:
            logger.warning("Could not read ratings file '%s': %s", self.ratings_path, exc)
            return None

        if {"userId", "movieId", "rating"}.issubset(set(raw_ratings.columns)):
            u_col, m_col, r_col = "userId", "movieId", "rating"
        elif {"user_id", "movie_id", "rating"}.issubset(set(raw_ratings.columns)):
            u_col, m_col, r_col = "user_id", "movie_id", "rating"
        else:
            logger.warning(
                "Ratings file '%s' does not contain required columns (userId/user_id, movieId/movie_id, rating).",
                self.ratings_path,
            )
            return None

        user_id = pd.to_numeric(raw_ratings[u_col], errors="coerce")
        movie_id = pd.to_numeric(raw_ratings[m_col], errors="coerce")
        rating = pd.to_numeric(raw_ratings[r_col], errors="coerce")

        valid_mask = (
            user_id.notna()
            & movie_id.notna()
            & (user_id > 0)
            & (movie_id > 0)
            & rating.notna()
            & (rating >= 0.5)
            & (rating <= 5.0)
            & movie_id.isin(valid_movie_ids)
        )
        cleaned = pd.DataFrame(
            {
                "user_id": user_id.loc[valid_mask].astype(int),
                "movie_id": movie_id.loc[valid_mask].astype(int),
                "rating": rating.loc[valid_mask].astype(float),
            }
        )
        if "timestamp" in raw_ratings.columns:
            ts = pd.to_numeric(raw_ratings.loc[valid_mask, "timestamp"], errors="coerce")
            if ts.notna().any():
                cleaned["timestamp"] = ts.fillna(-1).astype(int)
                cleaned = cleaned.sort_values(by="timestamp", ascending=True, kind="mergesort")

        if cleaned.empty:
            logger.info("No overlapping user ratings found for active catalog movie IDs.")
            return None

        before_dedup = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
        if "timestamp" in cleaned.columns:
            cleaned = cleaned.drop(columns=["timestamp"])

        logger.info(
            "Loaded %d unique user-movie interactions (removed %d duplicate rating logs).",
            len(cleaned),
            before_dedup - len(cleaned),
        )
        return cleaned.reset_index(drop=True)
