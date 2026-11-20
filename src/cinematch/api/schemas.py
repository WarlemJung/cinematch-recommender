"""Pydantic schemas for request parameter validation and standardized API responses."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrendingQueryParams(BaseModel):
    """Validated query parameters for GET /api/trending."""

    model_config = ConfigDict(extra="ignore")

    top_n: int = Field(default=12, ge=1, le=200)
    page: int = Field(default=1, ge=1, le=10000)
    page_size: int | None = Field(default=None, ge=1, le=100)
    genre: str | None = Field(default=None, max_length=64)
    min_year: int | None = Field(default=None, ge=1880, le=2100)
    max_year: int | None = Field(default=None, ge=1880, le=2100)
    min_rating: float | None = Field(default=None, ge=0.0, le=10.0)

    @field_validator("genre")
    @classmethod
    def strip_genre(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None


class SearchQueryParams(BaseModel):
    """Validated query parameters for GET /api/movies/search."""

    model_config = ConfigDict(extra="ignore")

    q: str = Field(default="", max_length=200)
    page: int = Field(default=1, ge=1, le=10000)
    page_size: int = Field(default=20, ge=1, le=100)
    genre: str | None = Field(default=None, max_length=64)
    min_year: int | None = Field(default=None, ge=1880, le=2100)
    max_year: int | None = Field(default=None, ge=1880, le=2100)
    min_rating: float | None = Field(default=None, ge=0.0, le=10.0)

    @field_validator("q")
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip()

    @field_validator("genre")
    @classmethod
    def strip_genre(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None


class RecommendationQueryParams(BaseModel):
    """Validated query parameters for recommendation endpoints."""

    model_config = ConfigDict(extra="ignore")

    top_n: int = Field(default=12, ge=1, le=50)
    genre: str | None = Field(default=None, max_length=64)
    min_year: int | None = Field(default=None, ge=1880, le=2100)
    max_year: int | None = Field(default=None, ge=1880, le=2100)
    min_rating: float | None = Field(default=None, ge=0.0, le=10.0)

    @field_validator("genre")
    @classmethod
    def strip_genre(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None


class ErrorResponseSchema(BaseModel):
    """Standardized error response schema."""

    error: str
    error_code: str
    status_code: int
    details: list[dict[str, Any]] | None = None
