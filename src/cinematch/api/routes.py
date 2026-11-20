"""REST API routes and web interface controller for CineMatch."""

from __future__ import annotations

from typing import Any
from flask import Blueprint, Response, current_app, jsonify, render_template, request

from cinematch.api.schemas import (
    RecommendationQueryParams,
    SearchQueryParams,
    TrendingQueryParams,
)
from cinematch.services.recommendation_service import RecommendationService
from cinematch.utils.exceptions import CineMatchError, InvalidParameterError


def _get_rec_service() -> RecommendationService:
    """Retrieve the active RecommendationService from the Flask app context."""
    service: RecommendationService | None = current_app.config.get("RECOMMENDATION_SERVICE")
    if service is None:
        raise CineMatchError(
            "Recommendation system is not initialized. Verify dataset availability.",
            status_code=503,
            error_code="SYSTEM_NOT_INITIALIZED",
        )
    return service


def _extract_query_dict() -> dict[str, Any]:
    """Extract non-empty query arguments from the current Flask request."""
    return {k: v for k, v in request.args.items() if v is not None and v.strip() != ""}


def create_api_blueprint() -> Blueprint:
    """Create the main Blueprint serving both the web interface and REST API endpoints."""
    bp = Blueprint("cinematch", __name__)

    @bp.route("/", methods=["GET"])
    def index() -> str:
        return render_template("index.html")

    @bp.route("/api/health", methods=["GET"])
    def health_check() -> tuple[Response, int]:
        service: RecommendationService | None = current_app.config.get("RECOMMENDATION_SERVICE")
        if service is None:
            return (
                jsonify(
                    {
                        "status": "degraded",
                        "service": "CineMatch Hybrid Recommendation Engine",
                        "error": "Dataset or recommendation service is not initialized.",
                    }
                ),
                503,
            )
        return jsonify(service.get_health_status()), 200

    @bp.route("/api/trending", methods=["GET"])
    def get_trending() -> tuple[Response, int]:
        service = _get_rec_service()
        params = TrendingQueryParams.model_validate(_extract_query_dict())
        if params.min_year and params.max_year and params.min_year > params.max_year:
            raise InvalidParameterError("min_year cannot be greater than max_year.")

        result = service.get_trending(
            top_n=params.top_n,
            page=params.page,
            page_size=params.page_size,
            genre=params.genre,
            min_year=params.min_year,
            max_year=params.max_year,
            min_rating=params.min_rating,
        )
        return jsonify(result), 200

    @bp.route("/api/genres", methods=["GET"])
    def get_genres() -> tuple[Response, int]:
        service = _get_rec_service()
        genres = service.movie_service.get_genres()
        return jsonify({"items": genres, "total": len(genres)}), 200

    @bp.route("/api/movies/search", methods=["GET"])
    def search_movies() -> tuple[Response, int]:
        service = _get_rec_service()
        params = SearchQueryParams.model_validate(_extract_query_dict())
        if params.min_year and params.max_year and params.min_year > params.max_year:
            raise InvalidParameterError("min_year cannot be greater than max_year.")

        result = service.movie_service.search_movies(
            query=params.q,
            genre=params.genre,
            min_year=params.min_year,
            max_year=params.max_year,
            min_rating=params.min_rating,
            page=params.page,
            page_size=params.page_size,
        )
        return jsonify(result), 200

    @bp.route("/api/movies/<int:movie_id>", methods=["GET"])
    def get_movie_details(movie_id: int) -> tuple[Response, int]:
        service = _get_rec_service()
        movie = service.movie_service.get_movie_by_id(movie_id)
        return jsonify(movie), 200

    @bp.route("/api/recommend/<int:movie_id>", methods=["GET"])
    def recommend_by_id(movie_id: int) -> tuple[Response, int]:
        service = _get_rec_service()
        params = RecommendationQueryParams.model_validate(_extract_query_dict())
        result = service.get_content_recommendations(
            identifier=movie_id,
            top_n=params.top_n,
        )
        return jsonify(result), 200

    @bp.route("/api/recommend", methods=["GET"])
    def recommend_by_query() -> tuple[Response, int]:
        """
        Backward-compatible content recommendation endpoint supporting `?title=...` or `?movie_id=...`.
        """
        service = _get_rec_service()
        raw_args = _extract_query_dict()
        movie_id_arg = raw_args.get("movie_id")
        title_arg = raw_args.get("title")

        identifier: int | str
        if movie_id_arg and movie_id_arg.isdigit():
            identifier = int(movie_id_arg)
        elif title_arg:
            identifier = title_arg
        else:
            raise InvalidParameterError("Query parameter 'title' or 'movie_id' is required.")

        params = RecommendationQueryParams.model_validate(raw_args)
        result = service.get_content_recommendations(
            identifier=identifier,
            top_n=params.top_n,
        )
        return jsonify(result), 200

    @bp.route("/api/recommendations/hybrid/<int:movie_id>", methods=["GET"])
    def recommend_hybrid_by_id(movie_id: int) -> tuple[Response, int]:
        service = _get_rec_service()
        params = RecommendationQueryParams.model_validate(_extract_query_dict())
        if params.min_year and params.max_year and params.min_year > params.max_year:
            raise InvalidParameterError("min_year cannot be greater than max_year.")

        result = service.get_hybrid_recommendations(
            identifier=movie_id,
            top_n=params.top_n,
            genre=params.genre,
            min_year=params.min_year,
            max_year=params.max_year,
            min_rating=params.min_rating,
        )
        return jsonify(result), 200

    @bp.route("/api/recommendations/hybrid", methods=["GET"])
    def recommend_hybrid_by_query() -> tuple[Response, int]:
        """Hybrid recommendation endpoint supporting query by `?title=...` or `?movie_id=...`."""
        service = _get_rec_service()
        raw_args = _extract_query_dict()
        movie_id_arg = raw_args.get("movie_id")
        title_arg = raw_args.get("title")

        identifier: int | str
        if movie_id_arg and movie_id_arg.isdigit():
            identifier = int(movie_id_arg)
        elif title_arg:
            identifier = title_arg
        else:
            raise InvalidParameterError("Query parameter 'title' or 'movie_id' is required.")

        params = RecommendationQueryParams.model_validate(raw_args)
        if params.min_year and params.max_year and params.min_year > params.max_year:
            raise InvalidParameterError("min_year cannot be greater than max_year.")

        result = service.get_hybrid_recommendations(
            identifier=identifier,
            top_n=params.top_n,
            genre=params.genre,
            min_year=params.min_year,
            max_year=params.max_year,
            min_rating=params.min_rating,
        )
        return jsonify(result), 200

    return bp
