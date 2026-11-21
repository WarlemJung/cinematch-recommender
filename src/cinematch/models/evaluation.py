"""Quantitative evaluation pipeline for recommendation models without data leakage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from cinematch.models.collaborative import CollaborativeRecommender
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
        upper_tri_sum = float(np.sum(np.triu(sim_mat, k=1)))
        num_pairs = n_items * (n_items - 1) / 2
        list_similarities.append(float(np.clip(upper_tri_sum / num_pairs, 0.0, 1.0)))

    return float(np.mean(list_similarities)) if list_similarities else 0.0


def recommendation_diversity(
    all_recommendation_lists: Sequence[Sequence[int]],
    content_model: ContentBasedRecommender,
) -> float:
    """Compute Intra-List Diversity (Recommendation Diversity) as 1.0 - Intra-List Similarity."""
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


def build_relevance_oracle(
    movies_df: pd.DataFrame,
    min_group_size: int = 3,
) -> list[tuple[int, set[int]]]:
    """
    Build the ground-truth relevance oracle R(q) without data leakage.

    Relevance Criteria (matching documented specification):
    A candidate movie m != q is relevant to query q (m in R(q)) if and only if:
    1. Both q and m belong to the same non-empty franchise/collection (`collection_name`),
       where the collection has at least `min_group_size` movies in the catalog; AND
    2. Movie m shares at least one genre with query q (`genres_list` intersection is non-empty,
       or both have empty genre lists).
    """
    has_collection = movies_df.loc[movies_df["collection_name"].astype(str).str.strip().str.len() > 0]
    col_groups = has_collection.groupby("collection_name")["movie_id"].apply(list).to_dict()
    valid_collections = {
        col: [int(m) for m in mids]
        for col, mids in col_groups.items()
        if len(mids) >= min_group_size
    }

    genre_map: dict[int, set[str]] = {}
    for row in has_collection.to_dict(orient="records"):
        mid = int(row["movie_id"])
        g_list = row.get("genres_list")
        genre_map[mid] = {str(g).strip().lower() for g in g_list if g} if isinstance(g_list, list) else set()

    query_records: list[tuple[int, set[int]]] = []
    for _col_name, mids in sorted(valid_collections.items()):
        for q_mid in mids:
            q_genres = genre_map.get(q_mid, set())
            relevant: set[int] = set()
            for cand_mid in mids:
                if cand_mid == q_mid:
                    continue
                cand_genres = genre_map.get(cand_mid, set())
                if not q_genres or not cand_genres or (q_genres & cand_genres):
                    relevant.add(cand_mid)
            if relevant:
                query_records.append((q_mid, relevant))

    return query_records


def _compute_metrics_for_pairs(
    model_name: str,
    pairs: list[tuple[list[int], set[int]]],
    k: int,
    catalog_size: int,
    content_model: ContentBasedRecommender,
) -> ModelEvaluationMetrics:
    all_rec_lists = [p[0] for p in pairs]
    p_scores = [precision_at_k(recs, rel, k) for recs, rel in pairs]
    r_scores = [recall_at_k(recs, rel, k) for recs, rel in pairs]
    ndcg_scores = [ndcg_at_k(recs, rel, k) for recs, rel in pairs]
    ap_scores = [average_precision_at_k(recs, rel, k) for recs, rel in pairs]

    cov = catalog_coverage(all_rec_lists, catalog_size)
    ils = intra_list_similarity(all_rec_lists, content_model)
    div = float(np.clip(1.0 - ils, 0.0, 1.0))

    return ModelEvaluationMetrics(
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


def evaluate_catalog_models(
    movies_df: pd.DataFrame,
    k: int = 10,
    max_queries: int = 200,
    random_state: int = 42,
    ratings_df: pd.DataFrame | None = None,
    hybrid_weights: HybridWeights | None = None,
) -> dict[str, ModelEvaluationMetrics]:
    """
    Run a leakage-free quantitative benchmark comparing Demographic, Content-Based,
    Collaborative (when real user ratings exist), and Hybrid models.

    Anti-Leakage Protocol:
    - Target Relevance Oracle: Ground-truth relevant sets R(q) are defined by held-out
      franchise/collection membership (`collection_name`) combined with genre intersection
      (`genres_list`).
    - Feature Holdout: During evaluation, the Content-Based and Hybrid models are trained with
      `ContentFeatureWeights(collection=0)` so the held-out target label (`collection_name`) is
      completely excluded from the TF-IDF feature matrix.
    - Query Self-Exclusion: The queried movie q is strictly excluded from all recommendation lists.
    """
    eval_weights = ContentFeatureWeights(
        overview=2,
        genres=3,
        tagline=2,
        collection=0,  # Strictly held out to prevent data leakage into ground-truth oracle
        production=1,
        language=1,
    )
    demographic_model = DemographicRecommender(quantile=0.90).fit(movies_df)
    content_model = ContentBasedRecommender(weights=eval_weights).fit(movies_df)

    collab_model: CollaborativeRecommender | None = None
    if ratings_df is not None and not ratings_df.empty:
        fitted_collab = CollaborativeRecommender(random_state=random_state).fit(
            movies_df, ratings_df=ratings_df
        )
        if fitted_collab.is_available:
            collab_model = fitted_collab

    effective_hybrid_weights = hybrid_weights or HybridWeights(
        content=0.75,
        demographic=0.15,
        collaborative=0.10 if collab_model is not None else 0.0,
    )
    hybrid_model = HybridRecommender(
        content_model=content_model,
        demographic_model=demographic_model,
        collaborative_model=collab_model,
        weights=effective_hybrid_weights,
    ).fit(movies_df, ratings_df=ratings_df)

    query_records = build_relevance_oracle(movies_df, min_group_size=3)
    if not query_records:
        raise ValueError("Dataset does not contain sufficient collection groups for holdout evaluation.")

    rng = np.random.default_rng(random_state)
    if len(query_records) > max_queries:
        chosen_indices = rng.choice(len(query_records), size=max_queries, replace=False)
        sampled_queries = [query_records[int(i)] for i in sorted(chosen_indices)]
    else:
        sampled_queries = query_records

    demo_top_df = demographic_model.recommend(top_n=k + 1)
    demo_top_ids = [int(m) for m in demo_top_df["movie_id"].tolist()]

    results_by_model: dict[str, list[tuple[list[int], set[int]]]] = {
        "Demographic Filtering": [],
        "Content-Based Filtering": [],
        "Hybrid Recommendation": [],
    }
    if collab_model is not None:
        results_by_model["Collaborative Filtering"] = []

    for query_mid, relevant_set in sampled_queries:
        demo_recs = [mid for mid in demo_top_ids if mid != query_mid][:k]
        results_by_model["Demographic Filtering"].append((demo_recs, relevant_set))

        cb_df = content_model.recommend(identifier=query_mid, top_n=k)
        cb_recs = [int(m) for m in cb_df["movie_id"].tolist()]
        results_by_model["Content-Based Filtering"].append((cb_recs, relevant_set))

        if collab_model is not None and collab_model.has_item(query_mid):
            cf_df = collab_model.recommend(movie_id=query_mid, movies_df=movies_df, top_n=k)
            cf_recs = [int(m) for m in cf_df["movie_id"].tolist()]
            results_by_model["Collaborative Filtering"].append((cf_recs, relevant_set))

        hyb_items = hybrid_model.recommend(identifier=query_mid, top_n=k)
        hyb_recs = [item.movie_id for item in hyb_items]
        results_by_model["Hybrid Recommendation"].append((hyb_recs, relevant_set))

    catalog_size = len(movies_df)
    summary: dict[str, ModelEvaluationMetrics] = {}

    for model_name, pairs in results_by_model.items():
        if not pairs:
            continue
        summary[model_name] = _compute_metrics_for_pairs(
            model_name=model_name,
            pairs=pairs,
            k=k,
            catalog_size=catalog_size,
            content_model=content_model,
        )

    return summary


def evaluate_hybrid_weight_sensitivity(
    movies_df: pd.DataFrame,
    weight_configs: Sequence[tuple[float, float]] | None = None,
    k: int = 10,
    max_queries: int = 200,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Evaluate sensitivity of the Hybrid Recommender across multiple (w_content, w_demographic)
    weight ratios on the exact same leakage-free query set.
    """
    configs = list(weight_configs) if weight_configs is not None else [
        (0.90, 0.10),
        (0.85, 0.15),
        (0.75, 0.25),
        (0.65, 0.35),
        (0.50, 0.50),
    ]

    eval_weights = ContentFeatureWeights(
        overview=2,
        genres=3,
        tagline=2,
        collection=0,
        production=1,
        language=1,
    )
    demographic_model = DemographicRecommender(quantile=0.90).fit(movies_df)
    content_model = ContentBasedRecommender(weights=eval_weights).fit(movies_df)

    query_records = build_relevance_oracle(movies_df, min_group_size=3)
    if not query_records:
        return []

    rng = np.random.default_rng(random_state)
    if len(query_records) > max_queries:
        chosen_indices = rng.choice(len(query_records), size=max_queries, replace=False)
        sampled_queries = [query_records[int(i)] for i in sorted(chosen_indices)]
    else:
        sampled_queries = query_records

    catalog_size = len(movies_df)
    sensitivity_rows: list[dict[str, Any]] = []

    for w_c, w_d in configs:
        hybrid = HybridRecommender(
            content_model=content_model,
            demographic_model=demographic_model,
            collaborative_model=None,
            weights=HybridWeights(content=w_c, demographic=w_d, collaborative=0.0),
        ).fit(movies_df)

        pairs: list[tuple[list[int], set[int]]] = []
        for query_mid, relevant_set in sampled_queries:
            items = hybrid.recommend(identifier=query_mid, top_n=k)
            pairs.append(([item.movie_id for item in items], relevant_set))

        metrics = _compute_metrics_for_pairs(
            model_name=f"Hybrid ({w_c:.2f}/{w_d:.2f})",
            pairs=pairs,
            k=k,
            catalog_size=catalog_size,
            content_model=content_model,
        )
        row = metrics.to_dict()
        row["weight_content"] = round(w_c, 2)
        row["weight_demographic"] = round(w_d, 2)
        sensitivity_rows.append(row)

    return sensitivity_rows
