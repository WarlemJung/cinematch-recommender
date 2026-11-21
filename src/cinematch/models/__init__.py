"""Machine Learning recommendation models and evaluation metrics."""

from cinematch.models.collaborative import CollaborativeRecommender
from cinematch.models.content_based import ContentBasedRecommender, ContentFeatureWeights
from cinematch.models.demographic import (
    DemographicParameters,
    DemographicRecommender,
    compute_weighted_rating,
)
from cinematch.models.evaluation import (
    ModelEvaluationMetrics,
    average_precision_at_k,
    build_relevance_oracle,
    catalog_coverage,
    evaluate_catalog_models,
    evaluate_hybrid_weight_sensitivity,
    intra_list_similarity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recommendation_diversity,
)
from cinematch.models.hybrid import (
    HybridRecommendationItem,
    HybridRecommender,
    HybridWeights,
)

__all__ = [
    "CollaborativeRecommender",
    "ContentBasedRecommender",
    "ContentFeatureWeights",
    "DemographicParameters",
    "DemographicRecommender",
    "HybridRecommendationItem",
    "HybridRecommender",
    "HybridWeights",
    "ModelEvaluationMetrics",
    "average_precision_at_k",
    "build_relevance_oracle",
    "catalog_coverage",
    "compute_weighted_rating",
    "evaluate_catalog_models",
    "evaluate_hybrid_weight_sensitivity",
    "intra_list_similarity",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "recommendation_diversity",
]
