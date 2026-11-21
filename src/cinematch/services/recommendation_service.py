"""Orchestration service for Demographic, Content-Based, Collaborative, and Hybrid recommendations."""

from __future__ import annotations

import math
from typing import Any

from cinematch import __version__
from cinematch.config import AppConfig
from cinematch.data.loader import LoadedDataset
from cinematch.models.collaborative import CollaborativeRecommender
from cinematch.models.content_based import ContentBasedRecommender, ContentFeatureWeights
from cinematch.models.demographic import DemographicRecommender
from cinematch.models.hybrid import HybridRecommender, HybridWeights
from cinematch.services.movie_service import MovieService
from cinematch.utils.exceptions import InvalidParameterError


class RecommendationService:
    """High-level facade coordinating all recommendation engines and catalog metadata."""

    def __init__(
        self,
        dataset: LoadedDataset,
        config: AppConfig,
        movie_service: MovieService,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.movie_service = movie_service

        self.demographic_model = DemographicRecommender(
            quantile=config.demographic_quantile,
            min_votes_override=config.demographic_min_votes,
        ).fit(dataset.movies_df)

        feature_weights = ContentFeatureWeights(
            overview=config.weight_overview,
            genres=config.weight_genres,
            tagline=config.weight_tagline,
            collection=config.weight_collection,
            production=config.weight_production,
            language=1,
        )
        self.content_model = ContentBasedRecommender(
            weights=feature_weights,
            max_features=config.tfidf_max_features,
        ).fit(dataset.movies_df)

        self.collaborative_model = CollaborativeRecommender(
            random_state=config.random_state,
        ).fit(dataset.movies_df, ratings_df=dataset.ratings_df)

        hybrid_weights = HybridWeights(
            content=config.hybrid_weight_content,
            demographic=config.hybrid_weight_demographic,
            collaborative=config.hybrid_weight_collaborative,
        )
        self.hybrid_model = HybridRecommender(
            content_model=self.content_model,
            demographic_model=self.demographic_model,
            collaborative_model=self.collaborative_model,
            weights=hybrid_weights,
        ).fit(dataset.movies_df, ratings_df=dataset.ratings_df)

    def get_trending(
        self,
        top_n: int = 12,
        page: int = 1,
        page_size: int | None = None,
        genre: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_rating: float | None = None,
    ) -> dict[str, Any]:
        """Return paginated top movies ranked by IMDB Weighted Rating."""
        if top_n < 1 or top_n > self.config.max_top_n * 4:
            raise InvalidParameterError(f"top_n must be between 1 and {self.config.max_top_n * 4}.")
        if page < 1:
            raise InvalidParameterError("page must be >= 1.")

        effective_page_size = page_size if page_size is not None else top_n
        if effective_page_size < 1 or effective_page_size > self.config.max_page_size:
            raise InvalidParameterError(f"page_size must be between 1 and {self.config.max_page_size}.")

        fetch_limit = max(top_n, page * effective_page_size)
        df_ranked = self.demographic_model.recommend(
            top_n=fetch_limit,
            genre=genre,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
        )

        total = len(df_ranked)
        total_pages = max(1, math.ceil(total / effective_page_size)) if total > 0 else 0
        start = (page - 1) * effective_page_size
        end = start + effective_page_size
        page_df = df_ranked.iloc[start:end]

        items = [self.movie_service.serialize_movie(row) for _, row in page_df.iterrows()]
        params = self.demographic_model.parameters
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": effective_page_size,
            "total_pages": total_pages,
            "parameters": {
                "mean_vote_c": params.mean_vote_c,
                "min_votes_m": params.min_votes_m,
                "quantile": params.quantile,
                "qualifying_count": params.qualifying_count,
            },
        }

    def get_content_recommendations(
        self,
        identifier: int | str,
        top_n: int = 12,
    ) -> dict[str, Any]:
        """Return content-based recommendations for a given movie_id or title."""
        if top_n < 1 or top_n > self.config.max_top_n:
            raise InvalidParameterError(f"top_n must be between 1 and {self.config.max_top_n}.")

        target_idx = self.content_model.resolve_movie_index(identifier)
        source_row = self.dataset.movies_df.iloc[target_idx]
        source_movie = self.movie_service.serialize_movie(source_row, include_details=True)

        recs_df = self.content_model.recommend(identifier=int(source_movie["movie_id"]), top_n=top_n)
        items = [self.movie_service.serialize_movie(row) for _, row in recs_df.iterrows()]

        return {
            "source_movie": source_movie,
            "algorithm": "content_based_tfidf",
            "total": len(items),
            "items": items,
        }

    def get_hybrid_recommendations(
        self,
        identifier: int | str,
        top_n: int = 12,
        genre: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_rating: float | None = None,
    ) -> dict[str, Any]:
        """Return hybrid recommendations with individual component scores and explanations."""
        if top_n < 1 or top_n > self.config.max_top_n:
            raise InvalidParameterError(f"top_n must be between 1 and {self.config.max_top_n}.")

        target_idx = self.content_model.resolve_movie_index(identifier)
        source_row = self.dataset.movies_df.iloc[target_idx]
        source_movie = self.movie_service.serialize_movie(source_row, include_details=True)

        hybrid_items = self.hybrid_model.recommend(
            identifier=int(source_movie["movie_id"]),
            top_n=top_n,
            genre=genre,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
        )

        serialized_items: list[dict[str, Any]] = []
        for item in hybrid_items:
            record = item.to_dict()
            record["poster_url"] = self.movie_service.build_poster_url(item.poster_path)
            serialized_items.append(record)

        active_weights = (
            hybrid_items[0].active_weights
            if hybrid_items
            else self.hybrid_model.weights.normalize(
                include_collaborative=self.collaborative_model.has_item(int(source_movie["movie_id"]))
            )
        )

        return {
            "source_movie": source_movie,
            "algorithm": "hybrid_weighted_ensemble",
            "collaborative_active": self.collaborative_model.has_item(int(source_movie["movie_id"])),
            "active_weights": active_weights,
            "total": len(serialized_items),
            "items": serialized_items,
        }

    def get_health_status(self) -> dict[str, Any]:
        """Return operational health check and model memory/data quality telemetry."""
        tfidf_shape = self.content_model.tfidf_matrix.shape
        return {
            "status": "healthy",
            "service": "CineMatch Hybrid Recommendation Engine",
            "version": __version__,
            "catalog_size": self.movie_service.catalog_size,
            "models": {
                "demographic_fitted": self.demographic_model.is_fitted,
                "content_based_fitted": self.content_model.is_fitted,
                "collaborative_available": self.collaborative_model.is_available,
                "collaborative_covered_movies": self.collaborative_model.covered_movies_count,
                "collaborative_distinct_users": self.collaborative_model.distinct_users_count,
                "hybrid_fitted": self.hybrid_model.is_fitted,
                "tfidf_matrix_shape": [int(tfidf_shape[0]), int(tfidf_shape[1])],
                "tfidf_sparse_memory_mb": round(self.content_model.memory_bytes / (1024 * 1024), 3),
            },
            "data_quality": self.dataset.quality_report.to_dict(),
        }
