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
        """Read CSV files, validate schema, run preprocessing, and return LoadedDataset."""
        if not self.dataset_path.exists() or not self.dataset_path.is_file():
            raise DatasetNotFoundError(str(self.dataset_path))

        logger.info("Loading movie metadata from '%s'", self.dataset_path)
        try:
            raw_df = pd.read_csv(self.dataset_path, low_memory=False)
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
        """Load real user ratings if a valid ratings CSV is configured and exists."""
        if self.ratings_path is None or not self.ratings_path.exists():
            return None

        logger.info("Loading optional user ratings from '%s'", self.ratings_path)
        try:
            raw_ratings = pd.read_csv(self.ratings_path, low_memory=False)
        except Exception as exc:
            logger.warning("Could not read ratings file '%s': %s", self.ratings_path, exc)
            return None

        required_cols = {"userId", "movieId", "rating"}
        if not required_cols.issubset(set(raw_ratings.columns)):
            logger.warning("Ratings file '%s' does not contain required columns %s", self.ratings_path, required_cols)
            return None

        user_id = pd.to_numeric(raw_ratings["userId"], errors="coerce")
        movie_id = pd.to_numeric(raw_ratings["movieId"], errors="coerce")
        rating = pd.to_numeric(raw_ratings["rating"], errors="coerce")

        valid_mask = (
            user_id.notna()
            & movie_id.notna()
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
        if cleaned.empty:
            logger.info("No overlapping user ratings found for active catalog movie IDs.")
            return None

        logger.info("Loaded %d verified user-movie rating interactions.", len(cleaned))
        return cleaned.reset_index(drop=True)
