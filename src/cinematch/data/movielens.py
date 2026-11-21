"""Optional GroupLens MovieLens dataset preparation and TMDB ID mapping utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
from pathlib import Path
import ssl
from typing import Any
import urllib.error
import urllib.request
import zipfile
import pandas as pd

from cinematch.data.loader import MovieDataLoader
from cinematch.utils.exceptions import DataValidationError
from cinematch.utils.logging import get_logger

logger = get_logger("data.movielens")

MOVIELENS_SMALL_URLS: tuple[str, ...] = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
)


@dataclass(frozen=True)
class MovieLensPreparationReport:
    """Audit metrics produced when mapping MovieLens ratings to the CineMatch catalog."""

    source: str
    output_path: str
    raw_ratings_rows: int
    mapped_tmdb_rows: int
    catalog_matched_rows: int
    deduplicated_rows: int
    distinct_users: int
    distinct_movies: int
    sparsity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_csv_from_zip(zip_bytes: bytes, filename_suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        matches = [name for name in zf.namelist() if name.endswith(filename_suffix)]
        if not matches:
            raise DataValidationError(f"Archive does not contain '{filename_suffix}'.")
        with zf.open(matches[0]) as fh:
            return pd.read_csv(fh, low_memory=False)


def download_movielens_small(timeout_seconds: int = 30) -> bytes:
    """Download the official GroupLens ml-latest-small.zip archive via HTTPS or HTTP fallback."""
    ssl_ctx: ssl.SSLContext | None = None
    try:
        import certifi  # type: ignore[import-untyped]

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_ctx = ssl.create_default_context()

    last_exc: Exception | None = None
    for url in MOVIELENS_SMALL_URLS:
        try:
            logger.info("Downloading MovieLens ml-latest-small from '%s'", url)
            req = urllib.request.Request(url, headers={"User-Agent": "CineMatch-MLOps/2.1.0"})
            with urllib.request.urlopen(req, timeout=timeout_seconds, context=ssl_ctx) as resp:
                data: bytes = resp.read()
                if data:
                    return data
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Failed downloading MovieLens from '%s': %s", url, exc)
            last_exc = exc
    raise DataValidationError(
        f"Could not download MovieLens ml-latest-small dataset: {last_exc}"
    )


def prepare_movielens_ratings(
    movies_metadata_path: Path | str,
    output_ratings_path: Path | str,
    source_zip: Path | str | None = None,
    source_dir: Path | str | None = None,
    download_if_missing: bool = True,
) -> MovieLensPreparationReport:
    """
    Map MovieLens ratings (`userId`, `movieId`, `rating`, `timestamp`) to CineMatch TMDB `movie_id`
    via `links.csv`, filter to valid catalog movies, deduplicate by `(user_id, movie_id)`, and save CSV.
    """
    loaded = MovieDataLoader(dataset_path=movies_metadata_path).load()
    valid_catalog_ids = set(loaded.movies_df["movie_id"].astype(int).tolist())

    source_desc = ""
    if source_dir is not None:
        src_path = Path(source_dir)
        ratings_file = src_path / "ratings.csv"
        links_file = src_path / "links.csv"
        if not ratings_file.exists() or not links_file.exists():
            raise DataValidationError(
                f"Directory '{src_path}' must contain both 'ratings.csv' and 'links.csv'."
            )
        raw_ratings = pd.read_csv(ratings_file, low_memory=False)
        links_df = pd.read_csv(links_file, low_memory=False)
        source_desc = str(src_path)
    elif source_zip is not None:
        zip_path = Path(source_zip)
        if not zip_path.exists():
            raise DataValidationError(f"MovieLens archive '{zip_path}' does not exist.")
        zip_bytes = zip_path.read_bytes()
        raw_ratings = _read_csv_from_zip(zip_bytes, "/ratings.csv")
        links_df = _read_csv_from_zip(zip_bytes, "/links.csv")
        source_desc = str(zip_path)
    elif download_if_missing:
        zip_bytes = download_movielens_small()
        raw_ratings = _read_csv_from_zip(zip_bytes, "/ratings.csv")
        links_df = _read_csv_from_zip(zip_bytes, "/links.csv")
        source_desc = "GroupLens ml-latest-small.zip"
    else:
        raise DataValidationError("No local MovieLens source provided and download_if_missing=False.")

    required_ratings = {"userId", "movieId", "rating"}
    required_links = {"movieId", "tmdbId"}
    if not required_ratings.issubset(raw_ratings.columns):
        raise DataValidationError("MovieLens ratings.csv missing required columns.")
    if not required_links.issubset(links_df.columns):
        raise DataValidationError("MovieLens links.csv missing required columns.")

    raw_count = len(raw_ratings)

    links_clean = links_df[["movieId", "tmdbId"]].copy()
    links_clean["movieId"] = pd.to_numeric(links_clean["movieId"], errors="coerce")
    links_clean["tmdbId"] = pd.to_numeric(links_clean["tmdbId"], errors="coerce")
    links_clean = links_clean.dropna(subset=["movieId", "tmdbId"])
    links_clean = links_clean.loc[(links_clean["movieId"] > 0) & (links_clean["tmdbId"] > 0)]
    links_clean["movieId"] = links_clean["movieId"].astype(int)
    links_clean["tmdbId"] = links_clean["tmdbId"].astype(int)
    links_clean = links_clean.drop_duplicates(subset=["movieId"], keep="first")

    merged = raw_ratings.merge(links_clean, on="movieId", how="inner")
    mapped_count = len(merged)

    merged["user_id"] = pd.to_numeric(merged["userId"], errors="coerce")
    merged["movie_id"] = merged["tmdbId"]
    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")

    valid_mask = (
        merged["user_id"].notna()
        & (merged["user_id"] > 0)
        & merged["rating"].notna()
        & (merged["rating"] >= 0.5)
        & (merged["rating"] <= 5.0)
        & merged["movie_id"].isin(valid_catalog_ids)
    )
    filtered = merged.loc[valid_mask].copy()
    catalog_matched_count = len(filtered)

    cols = ["user_id", "movie_id", "rating"]
    filtered["user_id"] = filtered["user_id"].astype(int)
    filtered["movie_id"] = filtered["movie_id"].astype(int)
    filtered["rating"] = filtered["rating"].astype(float)

    if "timestamp" in filtered.columns:
        ts = pd.to_numeric(filtered["timestamp"], errors="coerce")
        if ts.notna().any():
            filtered["timestamp"] = ts.fillna(0).astype(int)
            filtered = filtered.sort_values(by="timestamp", ascending=True, kind="mergesort")
            cols.append("timestamp")

    dedup = filtered[cols].drop_duplicates(subset=["user_id", "movie_id"], keep="last").reset_index(drop=True)

    out_path = Path(output_ratings_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Export standard MovieLens headers (`userId,movieId,rating[,timestamp]`) with TMDB IDs
    export_df = dedup.rename(columns={"user_id": "userId", "movie_id": "movieId"})
    export_df.to_csv(out_path, index=False)

    n_users = int(dedup["user_id"].nunique()) if not dedup.empty else 0
    n_movies = int(dedup["movie_id"].nunique()) if not dedup.empty else 0
    sparsity = 1.0 - (len(dedup) / (n_users * n_movies)) if (n_users > 0 and n_movies > 0) else 1.0

    report = MovieLensPreparationReport(
        source=source_desc,
        output_path=str(out_path),
        raw_ratings_rows=raw_count,
        mapped_tmdb_rows=mapped_count,
        catalog_matched_rows=catalog_matched_count,
        deduplicated_rows=len(dedup),
        distinct_users=n_users,
        distinct_movies=n_movies,
        sparsity=round(sparsity, 6),
    )
    logger.info(
        "Prepared MovieLens ratings: %d rows across %d users and %d catalog movies saved to '%s'.",
        report.deduplicated_rows,
        report.distinct_users,
        report.distinct_movies,
        report.output_path,
    )
    return report
