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
    build_item_self_information_map,
    build_relevance_oracle,
    calibrate_hybrid_weights,
    catalog_coverage,
    compute_paired_bootstrap_comparison,
    evaluate_catalog_models,
    evaluate_cold_start_stratification,
    evaluate_hybrid_weight_sensitivity,
    evaluate_user_level_collaborative_holdout,
    hit_rate_at_k,
    intra_list_similarity,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
    recommendation_diversity,
    split_oracle_queries,
    split_ratings_train_val_test,
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
    "build_item_self_information_map",
    "build_relevance_oracle",
    "calibrate_hybrid_weights",
    "catalog_coverage",
    "compute_paired_bootstrap_comparison",
    "compute_weighted_rating",
    "evaluate_catalog_models",
    "evaluate_cold_start_stratification",
    "evaluate_hybrid_weight_sensitivity",
    "evaluate_user_level_collaborative_holdout",
    "hit_rate_at_k",
    "intra_list_similarity",
    "ndcg_at_k",
    "novelty_at_k",
    "precision_at_k",
    "recall_at_k",
    "recommendation_diversity",
    "split_oracle_queries",
    "split_ratings_train_val_test",
]
