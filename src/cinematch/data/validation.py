"""Dataset validation and data quality reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd

from cinematch.utils.exceptions import DataValidationError

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "title",
        "overview",
        "vote_average",
        "vote_count",
        "release_date",
    }
)

OPTIONAL_METADATA_COLUMNS: tuple[str, ...] = (
    "id",
    "imdb_id",
    "genres",
    "tagline",
    "poster_path",
    "popularity",
    "belongs_to_collection",
    "production_companies",
    "original_language",
    "runtime",
)


@dataclass(frozen=True)
class DataQualityReport:
    """Comprehensive audit metrics produced during dataset validation and cleaning."""

    total_raw_rows: int
    valid_rows: int
    sampled_rows: int
    dropped_corrupted_id_rows: int
    dropped_missing_title_rows: int
    dropped_invalid_rating_rows: int
    dropped_duplicate_id_rows: int
    duplicate_title_count: int
    missing_overview_count: int
    invalid_release_date_count: int
    unique_genres_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return report as a serializable dictionary."""
        return asdict(self)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Verify that the dataframe exists, is non-empty, and has all required columns."""
    if df is None or df.empty:
        raise DataValidationError("The provided dataset is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sorted_missing = ", ".join(sorted(missing))
        raise DataValidationError(f"Dataset is missing required columns: {sorted_missing}")
