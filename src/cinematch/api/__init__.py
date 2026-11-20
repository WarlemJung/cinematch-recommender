"""REST API package for CineMatch."""

from cinematch.api.errors import register_error_handlers
from cinematch.api.routes import create_api_blueprint
from cinematch.api.schemas import (
    ErrorResponseSchema,
    RecommendationQueryParams,
    SearchQueryParams,
    TrendingQueryParams,
)

__all__ = [
    "ErrorResponseSchema",
    "RecommendationQueryParams",
    "SearchQueryParams",
    "TrendingQueryParams",
    "create_api_blueprint",
    "register_error_handlers",
]
