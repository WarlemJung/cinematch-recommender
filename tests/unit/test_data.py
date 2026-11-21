"""Unit tests for data loading, schema validation, cleaning, permissions, and reproducible sampling."""

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

    # Include duplicate user-movie ratings with timestamps to verify loader deduplication
    ratings_with_dups = pd.concat(
        [
            synthetic_ratings_dataframe.assign(timestamp=1000),
            pd.DataFrame([{"user_id": 1, "movie_id": 862, "rating": 2.5, "timestamp": 500}]),
        ],
        ignore_index=True,
    )
    ratings_csv = tmp_path / "ratings.csv"
    ratings_renamed = ratings_with_dups.rename(
        columns={"user_id": "userId", "movie_id": "movieId"}
    )
    ratings_renamed.to_csv(ratings_csv, index=False)

    loader = MovieDataLoader(dataset_path=csv_path, ratings_path=ratings_csv)
    loaded = loader.load()

    assert len(loaded.movies_df) == len(raw_movies_dataframe)
    assert loaded.ratings_df is not None
    # Duplicate (user=1, movie=862) should be deduplicated keeping timestamp=1000 (rating=5.0)
    assert len(loaded.ratings_df) == len(synthetic_ratings_dataframe)
    u1_m862 = loaded.ratings_df.loc[
        (loaded.ratings_df["user_id"] == 1) & (loaded.ratings_df["movie_id"] == 862)
    ]
    assert len(u1_m862) == 1
    assert float(u1_m862.iloc[0]["rating"]) == 5.0


def test_movie_data_loader_edge_cases_empty_directory_corrupted_and_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Path is a directory (e.g., broken Docker volume bind mount)
    dir_loader = MovieDataLoader(dataset_path=tmp_path)
    with pytest.raises(DataValidationError, match="is a directory"):
        dir_loader.load()

    # 2. Empty file (0 bytes)
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="empty"):
        MovieDataLoader(dataset_path=empty_file).load()

    # 3. Invalid CSV missing required columns
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="missing required columns"):
        MovieDataLoader(dataset_path=bad_csv).load()

    # 4. PermissionError when reading file
    def raise_perm_error(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise PermissionError("Access denied")

    monkeypatch.setattr(pd, "read_csv", raise_perm_error)
    with pytest.raises(DataValidationError, match="Permission denied"):
        MovieDataLoader(dataset_path=bad_csv).load()


def test_prepare_movielens_ratings_mapping_dedup_and_validation(
    tmp_path: Path,
    raw_movies_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zipfile
    from cinematch.data.movielens import prepare_movielens_ratings

    movies_csv = tmp_path / "movies_metadata.csv"
    raw_movies_dataframe.to_csv(movies_csv, index=False)

    ml_dir = tmp_path / "ml_dir"
    ml_dir.mkdir()
    pd.DataFrame(
        [
            {"movieId": 1, "imdbId": "0114709", "tmdbId": 862},
            {"movieId": 2, "imdbId": "0120363", "tmdbId": 863},
            {"movieId": 999, "imdbId": "0000000", "tmdbId": None},
        ]
    ).to_csv(ml_dir / "links.csv", index=False)
    pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 3.0, "timestamp": 100},
            {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 200},
            {"userId": 2, "movieId": 2, "rating": 4.5, "timestamp": 150},
            {"userId": 2, "movieId": 999, "rating": 4.0, "timestamp": 160},
        ]
    ).to_csv(ml_dir / "ratings.csv", index=False)

    out_csv = tmp_path / "out" / "ratings.csv"
    report = prepare_movielens_ratings(
        movies_metadata_path=movies_csv,
        output_ratings_path=out_csv,
        source_dir=ml_dir,
        download_if_missing=False,
    )
    assert report.deduplicated_rows == 2
    assert report.distinct_users == 2
    assert report.distinct_movies == 2
    saved_df = pd.read_csv(out_csv)
    assert float(saved_df.loc[saved_df["userId"] == 1, "rating"].iloc[0]) == 5.0

    # Also validate zip ingestion path
    zip_path = tmp_path / "ml.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(ml_dir / "links.csv", arcname="ml-latest-small/links.csv")
        zf.write(ml_dir / "ratings.csv", arcname="ml-latest-small/ratings.csv")
    out_zip_csv = tmp_path / "out_zip" / "ratings.csv"
    zip_report = prepare_movielens_ratings(
        movies_metadata_path=movies_csv,
        output_ratings_path=out_zip_csv,
        source_zip=zip_path,
        download_if_missing=False,
    )
    assert zip_report.deduplicated_rows == 2

    # Error handling for missing files / download failure
    with pytest.raises(DataValidationError, match="must contain both"):
        prepare_movielens_ratings(movies_csv, out_zip_csv, source_dir=tmp_path, download_if_missing=False)
    with pytest.raises(DataValidationError, match="does not exist"):
        prepare_movielens_ratings(movies_csv, out_zip_csv, source_zip=tmp_path / "missing.zip", download_if_missing=False)
    with pytest.raises(DataValidationError, match="No local MovieLens source"):
        prepare_movielens_ratings(movies_csv, out_zip_csv, download_if_missing=False)

    # Mock download_movielens_small via monkeypatch for download_if_missing=True branch
    from cinematch.data import movielens as ml_mod

    zip_bytes = zip_path.read_bytes()

    class _FakeResp:
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

        def read(self) -> bytes:
            return zip_bytes

    monkeypatch.setattr(ml_mod.urllib.request, "urlopen", lambda *_a, **_k: _FakeResp())
    dl_report = prepare_movielens_ratings(
        movies_csv, tmp_path / "dl" / "ratings.csv", download_if_missing=True
    )
    assert dl_report.source == "GroupLens ml-latest-small.zip"
    assert dl_report.deduplicated_rows == 2
