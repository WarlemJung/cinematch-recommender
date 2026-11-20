"""Movie catalog querying, search, filtering, and metadata enrichment service."""

from __future__ import annotations

import math
import re
from typing import Any
import pandas as pd

from cinematch.data.preprocessing import normalize_title
from cinematch.utils.exceptions import InvalidParameterError, MovieNotFoundError


class MovieService:
    """Provides read and search operations over the validated movie catalog."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        tmdb_image_base_url: str = "https://image.tmdb.org/t/p/w500",
    ) -> None:
        self._df = movies_df.copy().reset_index(drop=True)
        self._tmdb_image_base_url = tmdb_image_base_url.rstrip("/")
        self._id_to_idx: dict[int, int] = {
            int(mid): idx for idx, mid in enumerate(self._df["movie_id"].tolist())
        }

    @property
    def catalog_size(self) -> int:
        return len(self._df)

    def build_poster_url(self, poster_path: Any) -> str | None:
        """Construct a validated HTTPS poster URL if a valid poster_path is available."""
        if not isinstance(poster_path, str) or not poster_path.startswith("/"):
            return None
        return f"{self._tmdb_image_base_url}{poster_path}"

    def serialize_movie(self, row: pd.Series | dict[str, Any], include_details: bool = False) -> dict[str, Any]:
        """Convert a DataFrame row or dict into a clean, serializable movie dictionary."""
        year_val = int(row.get("release_year", 0) or 0)
        poster_path = row.get("poster_path")
        payload: dict[str, Any] = {
            "movie_id": int(row["movie_id"]),
            "title": str(row["title"]),
            "release_year": year_val if year_val > 0 else None,
            "release_date": str(row.get("release_date", "") or ""),
            "vote_average": round(float(row.get("vote_average", 0.0)), 2),
            "vote_count": int(row.get("vote_count", 0)),
            "popularity": round(float(row.get("popularity", 0.0)), 2),
            "genres": list(row.get("genres_list") or []),
            "overview": str(row.get("overview", "") or ""),
            "poster_path": poster_path if isinstance(poster_path, str) else None,
            "poster_url": self.build_poster_url(poster_path),
        }
        if "score" in row and pd.notna(row["score"]):
            payload["score"] = round(float(row["score"]), 4)
        if "similarity_score" in row and pd.notna(row["similarity_score"]):
            payload["similarity_score"] = round(float(row["similarity_score"]), 4)

        if include_details:
            payload.update(
                {
                    "tagline": str(row.get("tagline", "") or ""),
                    "collection_name": str(row.get("collection_name", "") or ""),
                    "production_companies": list(row.get("production_list") or []),
                    "original_language": str(row.get("original_language", "en") or "en"),
                    "runtime": round(float(row.get("runtime", 0.0) or 0.0), 1),
                    "imdb_id": str(row.get("imdb_id", "") or ""),
                }
            )
        return payload

    def get_movie_by_id(self, movie_id: int) -> dict[str, Any]:
        """Retrieve full details for a single movie by its unique integer ID."""
        idx = self._id_to_idx.get(int(movie_id))
        if idx is None:
            raise MovieNotFoundError(movie_id)
        row = self._df.iloc[idx]
        return self.serialize_movie(row, include_details=True)

    def get_genres(self) -> list[dict[str, Any]]:
        """Return a sorted list of all available genres and their movie counts."""
        counts: dict[str, int] = {}
        for g_list in self._df["genres_list"]:
            if isinstance(g_list, list):
                for genre in g_list:
                    if genre:
                        counts[genre] = counts.get(genre, 0) + 1
        return [
            {"name": name, "count": count}
            for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ]

    def search_movies(
        self,
        query: str = "",
        genre: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_rating: float | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """
        Search and filter the movie catalog with relevance ranking and pagination.
        Handles duplicate titles gracefully by exposing release_year, movie_id, and vote_count.
        """
        if page < 1:
            raise InvalidParameterError("page must be >= 1.")
        if page_size < 1:
            raise InvalidParameterError("page_size must be >= 1.")

        candidates = self._df.copy()

        if genre:
            g_low = genre.strip().lower()
            candidates = candidates.loc[
                candidates["genres_list"].apply(
                    lambda gl: any(str(g).lower() == g_low for g in gl)
                )
            ]

        if min_year is not None:
            candidates = candidates.loc[candidates["release_year"] >= min_year]

        if max_year is not None:
            candidates = candidates.loc[
                (candidates["release_year"] > 0) & (candidates["release_year"] <= max_year)
            ]

        if min_rating is not None:
            candidates = candidates.loc[candidates["vote_average"] >= min_rating]

        norm_q = normalize_title(query) if query else ""
        if norm_q:
            escaped_q = re.escape(norm_q)
            exact_mask = candidates["normalized_title"] == norm_q
            prefix_mask = candidates["normalized_title"].str.startswith(norm_q, na=False)
            contains_mask = candidates["normalized_title"].str.contains(escaped_q, na=False)

            candidates = candidates.loc[contains_mask].copy()
            if not candidates.empty:
                candidates["_match_tier"] = 2
                candidates.loc[prefix_mask.loc[candidates.index], "_match_tier"] = 1
                candidates.loc[exact_mask.loc[candidates.index], "_match_tier"] = 0
                candidates = candidates.sort_values(
                    by=["_match_tier", "vote_count", "popularity"],
                    ascending=[True, False, False],
                )
        else:
            candidates = candidates.sort_values(
                by=["vote_count", "vote_average", "popularity"],
                ascending=[False, False, False],
            )

        total = len(candidates)
        total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 0
        start = (page - 1) * page_size
        end = start + page_size
        page_slice = candidates.iloc[start:end]

        items = [self.serialize_movie(row, include_details=False) for _, row in page_slice.iterrows()]
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }
