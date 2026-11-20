"""Data preprocessing, cleaning, feature extraction, and reproducible sampling."""

from __future__ import annotations

import ast
import re
import unicodedata
from typing import Any
import pandas as pd

from cinematch.data.validation import DataQualityReport, validate_required_columns

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Normalize a movie title for case-insensitive and punctuation-resilient matching."""
    if not isinstance(title, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", title)
    ascii_text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    lowered = ascii_text.lower().strip()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def parse_metadata_list(raw_val: Any, key: str = "name", max_items: int = 5) -> list[str]:
    """Safely extract string values (such as genre or studio names) from serialized JSON/AST strings."""
    if isinstance(raw_val, list):
        items = raw_val
    elif isinstance(raw_val, str):
        stripped = raw_val.strip()
        if not stripped or stripped == "[]":
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return []
        if not isinstance(parsed, list):
            return []
        items = parsed
    else:
        return []

    extracted: list[str] = []
    for entry in items:
        if isinstance(entry, dict) and key in entry and isinstance(entry[key], str):
            val = entry[key].strip()
            if val:
                extracted.append(val)
        elif isinstance(entry, str) and entry.strip():
            extracted.append(entry.strip())
        if len(extracted) >= max_items:
            break
    return extracted


def parse_collection_name(raw_val: Any) -> str:
    """Safely extract collection/franchise name from serialized dict string."""
    if isinstance(raw_val, dict):
        name = raw_val.get("name")
        return str(name).strip() if name else ""
    if isinstance(raw_val, str):
        stripped = raw_val.strip()
        if not stripped or not stripped.startswith("{"):
            return ""
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict) and parsed.get("name"):
                return str(parsed["name"]).strip()
        except (ValueError, SyntaxError):
            return ""
    return ""


def sanitize_poster_path(raw_val: Any) -> str | None:
    """Validate that a TMDB poster path is a safe relative path starting with '/' and ending with an image ext."""
    if not isinstance(raw_val, str):
        return None
    cleaned = raw_val.strip()
    if not cleaned:
        return None
    if re.match(r"^/[A-Za-z0-9_\-]+\.(jpg|jpeg|png|webp)$", cleaned, flags=re.IGNORECASE):
        return cleaned
    return None


def preprocess_movies_dataframe(
    raw_df: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Validate, clean, deduplicate, enrich, and optionally sample the raw movies dataset.

    Preserves unique movie IDs, removes corrupted rows, extracts structured metadata,
    and generates an accurate DataQualityReport.
    """
    validate_required_columns(raw_df)
    total_raw_rows = len(raw_df)
    df = raw_df.copy()

    # 1. Assign or validate movie_id
    if "id" in df.columns:
        numeric_id = pd.to_numeric(df["id"], errors="coerce")
        corrupted_id_mask = numeric_id.isna() | (numeric_id <= 0)
        dropped_corrupted_id_rows = int(corrupted_id_mask.sum())
        df = df.loc[~corrupted_id_mask].copy()
        df["movie_id"] = numeric_id.loc[~corrupted_id_mask].astype(int)
    else:
        dropped_corrupted_id_rows = 0
        df["movie_id"] = range(1, len(df) + 1)

    # 2. Validate title (drop missing or blank titles)
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    missing_title_mask = df["title"] == ""
    dropped_missing_title_rows = int(missing_title_mask.sum())
    df = df.loc[~missing_title_mask].copy()

    # 3. Validate numeric vote_average and vote_count
    vote_avg = pd.to_numeric(df["vote_average"], errors="coerce")
    vote_cnt = pd.to_numeric(df["vote_count"], errors="coerce")
    invalid_rating_mask = (
        vote_avg.isna()
        | vote_cnt.isna()
        | (vote_avg < 0.0)
        | (vote_avg > 10.0)
        | (vote_cnt < 0)
    )
    dropped_invalid_rating_rows = int(invalid_rating_mask.sum())
    df = df.loc[~invalid_rating_mask].copy()
    df["vote_average"] = vote_avg.loc[~invalid_rating_mask].astype(float).round(2)
    df["vote_count"] = vote_cnt.loc[~invalid_rating_mask].astype(int)

    # 4. Deduplicate by movie_id (keep the entry with highest vote_count)
    df = df.sort_values(by=["movie_id", "vote_count"], ascending=[True, False])
    dup_id_mask = df.duplicated(subset=["movie_id"], keep="first")
    dropped_duplicate_id_rows = int(dup_id_mask.sum())
    df = df.loc[~dup_id_mask].copy()

    # Restore original row ordering before any sampling
    df = df.sort_index()

    # 5. Clean overview and tagline
    raw_overviews = df["overview"].fillna("").astype(str).str.strip()
    missing_overview_count = int((raw_overviews == "").sum())
    df["overview"] = raw_overviews

    if "tagline" in df.columns:
        df["tagline"] = df["tagline"].fillna("").astype(str).str.strip()
    else:
        df["tagline"] = ""

    # 6. Validate and extract release_date and release_year
    parsed_dates = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")
    invalid_release_date_count = int(parsed_dates.isna().sum())
    df["release_date"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna("")
    df["release_year"] = parsed_dates.dt.year.fillna(0).astype(int)

    # 7. Parse genres, collection, production companies, popularity, poster_path
    if "genres" in df.columns:
        df["genres_list"] = df["genres"].apply(lambda x: parse_metadata_list(x, key="name", max_items=6))
    else:
        df["genres_list"] = [[] for _ in range(len(df))]

    if "belongs_to_collection" in df.columns:
        df["collection_name"] = df["belongs_to_collection"].apply(parse_collection_name)
    else:
        df["collection_name"] = ""

    if "production_companies" in df.columns:
        df["production_list"] = df["production_companies"].apply(
            lambda x: parse_metadata_list(x, key="name", max_items=3)
        )
    else:
        df["production_list"] = [[] for _ in range(len(df))]

    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["popularity"] = 0.0

    if "runtime" in df.columns:
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["runtime"] = 0.0

    if "original_language" in df.columns:
        df["original_language"] = df["original_language"].fillna("en").astype(str).str.strip()
    else:
        df["original_language"] = "en"

    if "imdb_id" in df.columns:
        df["imdb_id"] = df["imdb_id"].fillna("").astype(str).str.strip()
    else:
        df["imdb_id"] = ""

    if "poster_path" in df.columns:
        df["poster_path"] = df["poster_path"].apply(sanitize_poster_path)
    else:
        df["poster_path"] = None

    # 8. Title normalization and duplicate title metrics
    df["normalized_title"] = df["title"].apply(normalize_title)
    duplicate_title_count = int(df["normalized_title"].duplicated(keep=False).sum())

    valid_rows = len(df)

    # 9. Reproducible sampling if sample_size is specified and smaller than valid_rows
    if sample_size is not None and 0 < sample_size < valid_rows:
        df = df.sample(n=sample_size, random_state=random_state).sort_index()

    df = df.reset_index(drop=True)
    sampled_rows = len(df)

    all_genres: set[str] = set()
    for g_list in df["genres_list"]:
        all_genres.update(g_list)

    report = DataQualityReport(
        total_raw_rows=total_raw_rows,
        valid_rows=valid_rows,
        sampled_rows=sampled_rows,
        dropped_corrupted_id_rows=dropped_corrupted_id_rows,
        dropped_missing_title_rows=dropped_missing_title_rows,
        dropped_invalid_rating_rows=dropped_invalid_rating_rows,
        dropped_duplicate_id_rows=dropped_duplicate_id_rows,
        duplicate_title_count=duplicate_title_count,
        missing_overview_count=missing_overview_count,
        invalid_release_date_count=invalid_release_date_count,
        unique_genres_count=len(all_genres),
    )

    return df, report
