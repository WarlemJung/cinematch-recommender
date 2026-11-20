"""Data loading, validation, and preprocessing layer."""

from cinematch.data.loader import LoadedDataset, MovieDataLoader
from cinematch.data.preprocessing import (
    normalize_title,
    parse_collection_name,
    parse_metadata_list,
    preprocess_movies_dataframe,
    sanitize_poster_path,
)
from cinematch.data.validation import DataQualityReport, REQUIRED_COLUMNS, validate_required_columns

__all__ = [
    "DataQualityReport",
    "LoadedDataset",
    "MovieDataLoader",
    "REQUIRED_COLUMNS",
    "normalize_title",
    "parse_collection_name",
    "parse_metadata_list",
    "preprocess_movies_dataframe",
    "sanitize_poster_path",
    "validate_required_columns",
]
