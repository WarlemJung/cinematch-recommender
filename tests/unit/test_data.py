"""Unit tests for data loading, schema validation, cleaning, and reproducible sampling."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from cinematch.data.loader import MovieDataLoader
from cinematch.data.preprocessing import (
    normalize_title,
    parse_collection_name,
    parse_metadata_list,
    preprocess_movies_dataframe,
    sanitize_poster_path,
)
from cinematch.data.validation import validate_required_columns
from cinematch.utils.exceptions import DatasetNotFoundError, DataValidationError


def test_validate_required_columns_rejects_empty_and_missing_cols() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        validate_required_columns(pd.DataFrame())

    incomplete_df = pd.DataFrame([{"title": "Inception", "overview": "Dreams"}])
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_required_columns(incomplete_df)


def test_normalize_title_handles_accents_and_punctuation() -> None:
    assert normalize_title("  Le Fabuleux Destin d'Amélie Poulain! ") == "le fabuleux destin d amelie poulain"
    assert normalize_title("Spider-Man: No Way Home (2021)") == "spider man no way home 2021"
    assert normalize_title(None) == ""  # type: ignore[arg-type]


def test_parse_metadata_and_collection_helpers() -> None:
    raw_genres = "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
    assert parse_metadata_list(raw_genres) == ["Action", "Adventure"]
    assert parse_metadata_list("invalid-json") == []
    assert parse_metadata_list([{"name": "Sci-Fi"}, "Thriller"]) == ["Sci-Fi", "Thriller"]

    raw_col = "{'id': 10, 'name': 'Star Wars Collection'}"
    assert parse_collection_name(raw_col) == "Star Wars Collection"
    assert parse_collection_name({"name": "Toy Story"}) == "Toy Story"
    assert parse_collection_name("malformed") == ""


def test_sanitize_poster_path_blocks_unsafe_paths() -> None:
    assert sanitize_poster_path("/rhIRbceoE9lR4veEXuwCC2wARtG.jpg") == "/rhIRbceoE9lR4veEXuwCC2wARtG.jpg"
    assert sanitize_poster_path("https://evil.example.com/xss.svg") is None
    assert sanitize_poster_path("javascript:alert(1)") is None
    assert sanitize_poster_path("/bad_extension.exe") is None
    assert sanitize_poster_path(None) is None


def test_preprocess_movies_dataframe_filters_corrupted_rows_and_deduplicates(
    raw_movies_dataframe: pd.DataFrame,
) -> None:
    dirty_rows = pd.DataFrame(
        [
            {
                "id": "1997-08-20",
                "title": "Corrupted Row",
                "overview": "Invalid id",
                "release_date": "1997-08-20",
                "vote_average": 5.0,
                "vote_count": 10,
            },
            {
                "id": "9991",
                "title": "   ",
                "overview": "No title movie",
                "release_date": "2001-01-01",
                "vote_average": 6.0,
                "vote_count": 10,
            },
            {
                "id": "9992",
                "title": "Invalid Rating Movie",
                "overview": "Score > 10",
                "release_date": "2001-01-01",
                "vote_average": 15.0,
                "vote_count": 10,
            },
            {
                "id": "862",
                "title": "Toy Story Duplicate Lower Votes",
                "overview": "Duplicate entry",
                "release_date": "1995-10-30",
                "vote_average": 7.7,
                "vote_count": 100,
            },
        ]
    )
    combined = pd.concat([raw_movies_dataframe, dirty_rows], ignore_index=True)
    cleaned_df, report = preprocess_movies_dataframe(combined)

    assert report.total_raw_rows == len(raw_movies_dataframe) + 4
    assert report.dropped_corrupted_id_rows == 1
    assert report.dropped_missing_title_rows == 1
    assert report.dropped_invalid_rating_rows == 1
    assert report.dropped_duplicate_id_rows == 1
    assert report.valid_rows == len(raw_movies_dataframe)
    assert report.missing_overview_count == 1
    assert report.invalid_release_date_count == 1
    assert report.duplicate_title_count == 2

    toy_story = cleaned_df.loc[cleaned_df["movie_id"] == 862].iloc[0]
    assert toy_story["vote_count"] == 5415
    assert toy_story["title"] == "Toy Story"


def test_reproducible_random_sampling(raw_movies_dataframe: pd.DataFrame) -> None:
    df_sample_a, report_a = preprocess_movies_dataframe(
        raw_movies_dataframe, sample_size=4, random_state=123
    )
    df_sample_b, _ = preprocess_movies_dataframe(
        raw_movies_dataframe, sample_size=4, random_state=123
    )
    df_sample_c, _ = preprocess_movies_dataframe(
        raw_movies_dataframe, sample_size=4, random_state=999
    )

    assert len(df_sample_a) == 4
    assert report_a.sampled_rows == 4
    assert df_sample_a["movie_id"].tolist() == df_sample_b["movie_id"].tolist()
    assert df_sample_a["movie_id"].tolist() != df_sample_c["movie_id"].tolist()


def test_movie_data_loader_from_disk_and_missing_file(
    tmp_path: Path,
    raw_movies_dataframe: pd.DataFrame,
    synthetic_ratings_dataframe: pd.DataFrame,
) -> None:
    missing_loader = MovieDataLoader(dataset_path=tmp_path / "nonexistent.csv")
    with pytest.raises(DatasetNotFoundError):
        missing_loader.load()

    csv_path = tmp_path / "movies.csv"
    raw_movies_dataframe.to_csv(csv_path, index=False)

    ratings_csv = tmp_path / "ratings.csv"
    ratings_renamed = synthetic_ratings_dataframe.rename(
        columns={"user_id": "userId", "movie_id": "movieId"}
    )
    ratings_renamed.to_csv(ratings_csv, index=False)

    loader = MovieDataLoader(dataset_path=csv_path, ratings_path=ratings_csv)
    loaded = loader.load()

    assert len(loaded.movies_df) == len(raw_movies_dataframe)
    assert loaded.ratings_df is not None
    assert len(loaded.ratings_df) == len(synthetic_ratings_dataframe)
