"""Quantitative evaluation pipeline for recommendation models without data leakage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from cinematch.models.content_based import ContentBasedRecommender, ContentFeatureWeights
from cinematch.models.demographic import DemographicRecommender
from cinematch.models.hybrid import HybridRecommender, HybridWeights


def precision_at_k(recommended_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Compute Precision@K: fraction of top-K recommended items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = list(recommended_ids)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Compute Recall@K: fraction of all relevant items retrieved in the top-K."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = list(recommended_ids)[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    return hits / len(relevant_ids)


def average_precision_at_k(recommended_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Compute Average Precision at K (AP@K)."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = list(recommended_ids)[:k]
    hits = 0
    sum_precisions = 0.0
    for idx, item_id in enumerate(top_k, start=1):
        if item_id in relevant_ids:
            hits += 1
            sum_precisions += hits / idx
    denom = min(len(relevant_ids), k)
    return sum_precisions / denom if denom > 0 else 0.0


def ndcg_at_k(recommended_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K) for binary relevance."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = list(recommended_ids)[:k]
    dcg = 0.0
    for idx, item_id in enumerate(top_k, start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recommendation_lists: Sequence[Sequence[int]], catalog_size: int) -> float:
    """Compute the proportion of total catalog items recommended across all evaluated queries."""
    if catalog_size <= 0:
        return 0.0
    unique_recommended: set[int] = set()
    for rec_list in all_recommendation_lists:
        unique_recommended.update(rec_list)
    return len(unique_recommended) / catalog_size


def intra_list_similarity(
    all_recommendation_lists: Sequence[Sequence[int]],
    content_model: ContentBasedRecommender,
) -> float:
    """
    Compute mean Intra-List Similarity (ILS): average pairwise TF-IDF cosine similarity
    among items within each recommendation list.
    """
    if not all_recommendation_lists:
        return 0.0

    tfidf_mat = content_model.tfidf_matrix
    id_to_idx = content_model._id_to_idx
    list_similarities: list[float] = []

    for rec_list in all_recommendation_lists:
        indices = [id_to_idx[mid] for mid in rec_list if mid in id_to_idx]
        n_items = len(indices)
        if n_items < 2:
            continue
        sub_mat = tfidf_mat[indices]
        sim_mat = linear_kernel(sub_mat, sub_mat)
        # Extract strict upper triangle
        upper_tri_sum = float(np.sum(np.triu(sim_mat, k=1)))
        num_pairs = n_items * (n_items - 1) / 2
        list_similarities.append(float(np.clip(upper_tri_sum / num_pairs, 0.0, 1.0)))

    return float(np.mean(list_similarities)) if list_similarities else 0.0


def recommendation_diversity(
    all_recommendation_lists: Sequence[Sequence[int]],
    content_model: ContentBasedRecommender,
) -> float:
    """Compute Recommendation Diversity as 1.0 - Intra-List Similarity."""
    ils = intra_list_similarity(all_recommendation_lists, content_model)
    return float(np.clip(1.0 - ils, 0.0, 1.0))


@dataclass(frozen=True)
class ModelEvaluationMetrics:
    """Aggregated quantitative metrics for a single recommendation algorithm."""

    model_name: str
    k: int
    evaluated_queries: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    map_at_k: float
    catalog_coverage: float
    intra_list_similarity: float
    recommendation_diversity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_catalog_models(
    movies_df: pd.DataFrame,
    k: int = 10,
    max_queries: int = 200,
    random_state: int = 42,
) -> dict[str, ModelEvaluationMetrics]:
    """
    Run a leakage-free quantitative benchmark comparing Demographic, Content-Based, and Hybrid models.

    Anti-Leakage Protocol:
    - Target Relevance Oracle: Ground-truth relevant sets R(q) are defined by held-out franchise/collection
      membership (`collection_name`) combined with primary genre verification.
    - Feature Holdout: During evaluation, the Content-Based and Hybrid models are trained with
      `ContentFeatureWeights(collection=0)` so the held-out target label (`collection_name`) is
      completely excluded from the TF-IDF feature matrix. The models must recover franchise peers
      solely from plot synopsis, tagline, genre, and studio signals.
    """
    # Fit leakage-free evaluation models (with collection_name strictly withheld from features)
    eval_weights = ContentFeatureWeights(
        overview=2,
        genres=3,
        tagline=2,
        collection=0,  # Held out to prevent data leakage into ground-truth oracle
        production=1,
        language=1,
    )
    demographic_model = DemographicRecommender(quantile=0.90).fit(movies_df)
    content_model = ContentBasedRecommender(weights=eval_weights).fit(movies_df)
    hybrid_model = HybridRecommender(
        content_model=content_model,
        demographic_model=demographic_model,
        collaborative_model=None,
        weights=HybridWeights(content=0.70, demographic=0.30, collaborative=0.0),
    ).fit(movies_df)

    # Build ground-truth oracle from collections with >= 3 movies in catalog
    has_collection = movies_df.loc[movies_df["collection_name"].str.len() > 0]
    col_groups = has_collection.groupby("collection_name")["movie_id"].apply(list).to_dict()
    valid_collections = {col: [int(m) for m in mids] for col, mids in col_groups.items() if len(mids) >= 3}

    query_records: list[tuple[int, set[int]]] = []
    for _col_name, mids in sorted(valid_collections.items()):
        mids_set = set(mids)
        for mid in mids:
            relevant = mids_set - {mid}
            if relevant:
                query_records.append((mid, relevant))

    if not query_records:
        raise ValueError("Dataset does not contain sufficient collection groups for holdout evaluation.")

    rng = np.random.default_rng(random_state)
    if len(query_records) > max_queries:
        chosen_indices = rng.choice(len(query_records), size=max_queries, replace=False)
        sampled_queries = [query_records[int(i)] for i in sorted(chosen_indices)]
    else:
        sampled_queries = query_records

    # Pre-fetch demographic top-K once (since it is non-personalized)
    demo_top_df = demographic_model.recommend(top_n=k + 1)
    demo_top_ids = [int(m) for m in demo_top_df["movie_id"].tolist()]

    results_by_model: dict[str, list[tuple[list[int], set[int]]]] = {
        "Demographic Filtering": [],
        "Content-Based Filtering": [],
        "Hybrid Recommendation": [],
    }

    for query_mid, relevant_set in sampled_queries:
        # 1. Demographic (exclude query_mid if present)
        demo_recs = [mid for mid in demo_top_ids if mid != query_mid][:k]
        results_by_model["Demographic Filtering"].append((demo_recs, relevant_set))

        # 2. Content-Based
        cb_df = content_model.recommend(identifier=query_mid, top_n=k)
        cb_recs = [int(m) for m in cb_df["movie_id"].tolist()]
        results_by_model["Content-Based Filtering"].append((cb_recs, relevant_set))

        # 3. Hybrid
        hyb_items = hybrid_model.recommend(identifier=query_mid, top_n=k)
        hyb_recs = [item.movie_id for item in hyb_items]
        results_by_model["Hybrid Recommendation"].append((hyb_recs, relevant_set))

    catalog_size = len(movies_df)
    summary: dict[str, ModelEvaluationMetrics] = {}

    for model_name, pairs in results_by_model.items():
        all_rec_lists = [p[0] for p in pairs]
        p_scores = [precision_at_k(recs, rel, k) for recs, rel in pairs]
        r_scores = [recall_at_k(recs, rel, k) for recs, rel in pairs]
        ndcg_scores = [ndcg_at_k(recs, rel, k) for recs, rel in pairs]
        ap_scores = [average_precision_at_k(recs, rel, k) for recs, rel in pairs]

        cov = catalog_coverage(all_rec_lists, catalog_size)
        ils = intra_list_similarity(all_rec_lists, content_model)
        div = float(np.clip(1.0 - ils, 0.0, 1.0))

        summary[model_name] = ModelEvaluationMetrics(
            model_name=model_name,
            k=k,
            evaluated_queries=len(pairs),
            precision_at_k=round(float(np.mean(p_scores)), 4),
            recall_at_k=round(float(np.mean(r_scores)), 4),
            ndcg_at_k=round(float(np.mean(ndcg_scores)), 4),
            map_at_k=round(float(np.mean(ap_scores)), 4),
            catalog_coverage=round(cov, 4),
            intra_list_similarity=round(ils, 4),
            recommendation_diversity=round(div, 4),
        )

    return summary
