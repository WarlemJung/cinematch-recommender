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
    """Compute Precision@K: |L_K(q) ∩ R(q)| / K."""
    if k <= 0:
        return 0.0
    top_k = list(recommended_ids)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Compute Recall@K: |L_K(q) ∩ R(q)| / |R(q)|."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = list(recommended_ids)[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(recommended_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Compute HitRate@K: 1.0 if |L_K(q) ∩ R(q)| >= 1 else 0.0."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = list(recommended_ids)[:k]
    return 1.0 if any(item_id in relevant_ids for item_id in top_k) else 0.0


def average_precision_at_k(recommended_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Compute Average Precision at K (AP@K) with denominator min(|R(q)|, K)."""
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
    """Compute Catalog Coverage: |∪_q L_K(q)| / N."""
    if catalog_size <= 0:
        return 0.0
    unique_recommended: set[int] = set()
    for rec_list in all_recommendation_lists:
        unique_recommended.update(rec_list)
    return len(unique_recommended) / catalog_size


def build_item_self_information_map(movies_df: pd.DataFrame) -> dict[int, float]:
    """
    Compute normalized self-information novelty score in [0, 1] for each catalog item:

        p(i) = (vote_count_i + 1) / sum_j(vote_count_j + 1)
        Novelty(i) = -log2(p(i)) / log2(sum_j(vote_count_j + 1))
    """
    if movies_df.empty:
        return {}
    counts = pd.to_numeric(movies_df["vote_count"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    shifted = counts + 1.0
    total = float(np.sum(shifted))
    if total <= 1.0:
        return {int(mid): 0.0 for mid in movies_df["movie_id"].tolist()}
    denom = math.log2(total)
    probs = shifted / total
    scores = np.clip(-np.log2(probs) / denom, 0.0, 1.0)
    return {int(mid): float(sc) for mid, sc in zip(movies_df["movie_id"].tolist(), scores, strict=True)}


def novelty_at_k(
    all_recommendation_lists: Sequence[Sequence[int]],
    item_novelty_map: dict[int, float],
    k: int,
) -> float:
    """Compute mean normalized Novelty@K across recommendation lists."""
    if k <= 0 or not all_recommendation_lists or not item_novelty_map:
        return 0.0
    list_novelties: list[float] = []
    for rec_list in all_recommendation_lists:
        top_k = list(rec_list)[:k]
        vals = [item_novelty_map[mid] for mid in top_k if mid in item_novelty_map]
        if vals:
            list_novelties.append(float(np.mean(vals)))
    return float(np.mean(list_novelties)) if list_novelties else 0.0


def intra_list_similarity(
    all_recommendation_lists: Sequence[Sequence[int]],
    content_model: ContentBasedRecommender,
) -> float:
    """
    Compute mean Intra-List Similarity (ILS@K): average pairwise TF-IDF cosine similarity
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
    """Compute Intra-List Diversity (Diversity@K) as 1.0 - Intra-List Similarity."""
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
    hit_rate_at_k: float
    catalog_coverage: float
    intra_list_similarity: float
    recommendation_diversity: float
    novelty_at_k: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_relevance_oracle(
    movies_df: pd.DataFrame,
    min_group_size: int = 3,
    require_genre_overlap: bool = False,
) -> list[tuple[int, set[int]]]:
    """
    Build the metadata-based ground-truth relevance oracle R_meta(q) without data leakage.

    By default (`require_genre_overlap=False`, v2.1.0 strict anti-circularity protocol),
    a candidate movie m != q is relevant to query q iff both belong to the same non-empty
    franchise/collection (`collection_name`) with at least `min_group_size` movies, while
    `collection_name` is strictly held out (`collection=0`) from TF-IDF training features.
    """
    has_collection = movies_df.loc[movies_df["collection_name"].astype(str).str.strip().str.len() > 0]
    col_groups = has_collection.groupby("collection_name")["movie_id"].apply(list).to_dict()
    valid_collections = {
        col: [int(m) for m in mids]
        for col, mids in col_groups.items()
        if len(mids) >= min_group_size
    }

    genre_map: dict[int, set[str]] = {}
    if require_genre_overlap:
        for row in has_collection.to_dict(orient="records"):
            mid = int(row["movie_id"])
            g_list = row.get("genres_list")
            genre_map[mid] = {str(g).strip().lower() for g in g_list if g} if isinstance(g_list, list) else set()

    query_records: list[tuple[int, set[int]]] = []
    for _col_name, mids in sorted(valid_collections.items()):
        for q_mid in mids:
            if not require_genre_overlap:
                relevant = {cand_mid for cand_mid in mids if cand_mid != q_mid}
            else:
                q_genres = genre_map.get(q_mid, set())
                relevant = set()
                for cand_mid in mids:
                    if cand_mid == q_mid:
                        continue
                    cand_genres = genre_map.get(cand_mid, set())
                    if not q_genres or not cand_genres or (q_genres & cand_genres):
                        relevant.add(cand_mid)
            if relevant:
                query_records.append((q_mid, relevant))

    return query_records


def split_oracle_queries(
    query_records: Sequence[tuple[int, set[int]]],
    val_size: int = 100,
    test_size: int = 200,
    random_state: int = 42,
) -> tuple[list[tuple[int, set[int]]], list[tuple[int, set[int]]]]:
    """
    Partition eligible oracle queries into strictly disjoint validation (calibration)
    and test (final evaluation) subsets with zero overlap.
    """
    total = len(query_records)
    if total == 0:
        return [], []

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(total)

    if total >= val_size + test_size:
        val_idx = sorted(int(i) for i in perm[:val_size])
        test_idx = sorted(int(i) for i in perm[val_size : val_size + test_size])
    else:
        n_val = max(1, int(round(total * (val_size / max(1, val_size + test_size)))))
        n_val = min(n_val, max(1, total - 1)) if total > 1 else 0
        val_idx = sorted(int(i) for i in perm[:n_val])
        test_idx = sorted(int(i) for i in perm[n_val:])

    val_queries = [query_records[i] for i in val_idx]
    test_queries = [query_records[i] for i in test_idx]
    return val_queries, test_queries


def split_ratings_train_val_test(
    ratings_df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split user ratings into strictly disjoint train, validation, and test DataFrames.

    Uses chronological per-user temporal splitting when a `timestamp` column exists;
    otherwise uses deterministic seeded shuffling per user.
    """
    if ratings_df is None or ratings_df.empty:
        empty = pd.DataFrame(columns=["user_id", "movie_id", "rating"])
        return empty.copy(), empty.copy(), empty.copy()

    df = ratings_df.copy()
    has_ts = "timestamp" in df.columns and pd.to_numeric(df["timestamp"], errors="coerce").notna().any()
    rng = np.random.default_rng(random_state)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _uid, user_group in df.groupby("user_id", sort=True):
        if has_ts:
            ordered = user_group.sort_values(by="timestamp", ascending=True, kind="mergesort")
        else:
            perm = rng.permutation(len(user_group))
            ordered = user_group.iloc[perm]

        n = len(ordered)
        if n < 3:
            train_parts.append(ordered)
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 5 else 0
        if n_test + n_val >= n:
            n_val = 0
            n_test = 1
        n_train = n - n_val - n_test

        train_parts.append(ordered.iloc[:n_train])
        if n_val > 0:
            val_parts.append(ordered.iloc[n_train : n_train + n_val])
        test_parts.append(ordered.iloc[n_train + n_val :])

    cols = list(df.columns)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=cols)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=cols)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=cols)
    return train_df, val_df, test_df


def _compute_metrics_for_pairs(
    model_name: str,
    pairs: list[tuple[list[int], set[int]]],
    k: int,
    catalog_size: int,
    content_model: ContentBasedRecommender,
    novelty_map: dict[int, float],
) -> ModelEvaluationMetrics:
    all_rec_lists = [p[0] for p in pairs]
    p_scores = [precision_at_k(recs, rel, k) for recs, rel in pairs]
    r_scores = [recall_at_k(recs, rel, k) for recs, rel in pairs]
    ndcg_scores = [ndcg_at_k(recs, rel, k) for recs, rel in pairs]
    ap_scores = [average_precision_at_k(recs, rel, k) for recs, rel in pairs]
    hr_scores = [hit_rate_at_k(recs, rel, k) for recs, rel in pairs]

    cov = catalog_coverage(all_rec_lists, catalog_size)
    ils = intra_list_similarity(all_rec_lists, content_model)
    div = float(np.clip(1.0 - ils, 0.0, 1.0))
    nov = novelty_at_k(all_rec_lists, novelty_map, k)

    return ModelEvaluationMetrics(
        model_name=model_name,
        k=k,
        evaluated_queries=len(pairs),
        precision_at_k=round(float(np.mean(p_scores)), 4),
        recall_at_k=round(float(np.mean(r_scores)), 4),
        ndcg_at_k=round(float(np.mean(ndcg_scores)), 4),
        map_at_k=round(float(np.mean(ap_scores)), 4),
        hit_rate_at_k=round(float(np.mean(hr_scores)), 4),
        catalog_coverage=round(cov, 4),
        intra_list_similarity=round(ils, 4),
        recommendation_diversity=round(div, 4),
        novelty_at_k=round(nov, 4),
    )


def calibrate_hybrid_weights(
    movies_df: pd.DataFrame,
    val_queries: Sequence[tuple[int, set[int]]],
    content_model: ContentBasedRecommender,
    demographic_model: DemographicRecommender,
    collab_model: CollaborativeRecommender | None = None,
    candidate_weights: Sequence[tuple[float, float, float]] | None = None,
    k: int = 10,
) -> tuple[HybridWeights, list[dict[str, Any]]]:
    """
    Calibrate Hybrid weights strictly on the validation query set `val_queries`
    by maximizing NDCG@K (tie-breaking on Recall@K), never touching test queries.
    """
    if not val_queries:
        return HybridWeights(), []

    if candidate_weights is not None:
        grid = list(candidate_weights)
    elif collab_model is not None and collab_model.is_available:
        grid = [
            (0.85, 0.10, 0.05),
            (0.80, 0.12, 0.08),
            (0.75, 0.15, 0.10),
            (0.70, 0.15, 0.15),
            (0.65, 0.15, 0.20),
        ]
    else:
        grid = [
            (0.90, 0.10, 0.0),
            (0.85, 0.15, 0.0),
            (0.80, 0.20, 0.0),
            (0.75, 0.25, 0.0),
            (0.65, 0.35, 0.0),
        ]

    novelty_map = build_item_self_information_map(movies_df)
    catalog_size = len(movies_df)
    history: list[dict[str, Any]] = []
    best_weights = HybridWeights(content=grid[0][0], demographic=grid[0][1], collaborative=grid[0][2])
    best_score: tuple[float, float] = (-1.0, -1.0)

    for w_c, w_d, w_cf in grid:
        w = HybridWeights(content=w_c, demographic=w_d, collaborative=w_cf)
        hybrid = HybridRecommender(
            content_model=content_model,
            demographic_model=demographic_model,
            collaborative_model=collab_model,
            weights=w,
        ).fit(movies_df)

        pairs: list[tuple[list[int], set[int]]] = []
        for q_mid, rel_set in val_queries:
            items = hybrid.recommend(identifier=q_mid, top_n=k)
            pairs.append(([item.movie_id for item in items], rel_set))

        metrics = _compute_metrics_for_pairs(
            model_name=f"Val Hybrid ({w_c:.2f}/{w_d:.2f}/{w_cf:.2f})",
            pairs=pairs,
            k=k,
            catalog_size=catalog_size,
            content_model=content_model,
            novelty_map=novelty_map,
        )
        row = metrics.to_dict()
        row["weights"] = {"content": w_c, "demographic": w_d, "collaborative": w_cf}
        history.append(row)

        score_key = (metrics.ndcg_at_k, metrics.recall_at_k)
        if score_key > best_score:
            best_score = score_key
            best_weights = w

    return best_weights, history


def compute_paired_bootstrap_comparison(
    baseline_pairs: Sequence[tuple[list[int], set[int]]],
    candidate_pairs: Sequence[tuple[list[int], set[int]]],
    k: int = 10,
    n_resamples: int = 5000,
    random_state: int = 42,
    content_model: ContentBasedRecommender | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Compute paired query-level bootstrap confidence intervals (95%) comparing a candidate
    model against a baseline model over the exact same test queries.
    """
    if len(baseline_pairs) != len(candidate_pairs) or len(baseline_pairs) == 0:
        return {}

    n_q = len(baseline_pairs)
    metric_fns = {
        "precision_at_k": lambda recs, rel: precision_at_k(recs, rel, k),
        "recall_at_k": lambda recs, rel: recall_at_k(recs, rel, k),
        "ndcg_at_k": lambda recs, rel: ndcg_at_k(recs, rel, k),
        "map_at_k": lambda recs, rel: average_precision_at_k(recs, rel, k),
        "hit_rate_at_k": lambda recs, rel: hit_rate_at_k(recs, rel, k),
    }

    base_arrays: dict[str, np.ndarray] = {
        name: np.array([fn(r, rel) for r, rel in baseline_pairs], dtype=float)
        for name, fn in metric_fns.items()
    }
    cand_arrays: dict[str, np.ndarray] = {
        name: np.array([fn(r, rel) for r, rel in candidate_pairs], dtype=float)
        for name, fn in metric_fns.items()
    }

    if content_model is not None:
        base_arrays["diversity"] = np.array(
            [1.0 - intra_list_similarity([r], content_model) for r, _ in baseline_pairs],
            dtype=float,
        )
        cand_arrays["diversity"] = np.array(
            [1.0 - intra_list_similarity([r], content_model) for r, _ in candidate_pairs],
            dtype=float,
        )

    rng = np.random.default_rng(random_state)
    boot_idx = rng.integers(0, n_q, size=(n_resamples, n_q))

    report: dict[str, dict[str, Any]] = {}
    for name in base_arrays:
        b_arr = base_arrays[name]
        c_arr = cand_arrays[name]
        diffs = c_arr - b_arr
        obs_diff = float(np.mean(diffs))
        boot_diffs = np.mean(diffs[boot_idx], axis=1)
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        p_two_sided = min(
            1.0,
            2.0 * min(float(np.mean(boot_diffs <= 0.0)), float(np.mean(boot_diffs >= 0.0))),
        )
        report[name] = {
            "baseline_mean": round(float(np.mean(b_arr)), 6),
            "candidate_mean": round(float(np.mean(c_arr)), 6),
            "mean_diff": round(obs_diff, 6),
            "bootstrap_se": round(float(np.std(boot_diffs, ddof=1)), 6),
            "ci_95_low": round(float(ci_low), 6),
            "ci_95_high": round(float(ci_high), 6),
            "statistically_significant_95": bool(ci_low > 0.0 or ci_high < 0.0),
            "bootstrap_two_sided_pvalue": round(p_two_sided, 6),
            "wins": int(np.sum(diffs > 1e-12)),
            "ties": int(np.sum(np.abs(diffs) <= 1e-12)),
            "losses": int(np.sum(diffs < -1e-12)),
            "sample_size": n_q,
        }
    return report


def evaluate_user_level_collaborative_holdout(
    movies_df: pd.DataFrame,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int = 200,
    min_positive_rating: float = 4.0,
    random_state: int = 42,
    hybrid_weights: HybridWeights | None = None,
) -> dict[str, Any]:
    """
    Evaluate observed user preferences using chronological/split test interactions.

    Each user's highest-rated movie in `train_ratings` acts as the query seed, and all
    movies rated >= `min_positive_rating` by that user in `test_ratings` act as ground-truth
    observed relevant items. `CollaborativeRecommender` is fitted strictly on `train_ratings`.
    """
    if train_ratings.empty or test_ratings.empty:
        return {}

    eval_weights = ContentFeatureWeights(overview=2, genres=3, tagline=2, collection=0, production=1, language=1)
    demo_model = DemographicRecommender(quantile=0.90).fit(movies_df)
    cb_model = ContentBasedRecommender(weights=eval_weights).fit(movies_df)
    cf_model = CollaborativeRecommender(random_state=random_state).fit(movies_df, ratings_df=train_ratings)
    if not cf_model.is_available:
        return {}

    hyb_weights = hybrid_weights or HybridWeights(content=0.75, demographic=0.15, collaborative=0.10)
    hyb_model = HybridRecommender(
        content_model=cb_model,
        demographic_model=demo_model,
        collaborative_model=cf_model,
        weights=hyb_weights,
    ).fit(movies_df, ratings_df=train_ratings)

    valid_ids = set(movies_df["movie_id"].astype(int).tolist())
    test_pos = test_ratings.loc[
        (test_ratings["rating"] >= min_positive_rating) & (test_ratings["movie_id"].isin(valid_ids))
    ]
    test_user_rel: dict[int, set[int]] = (
        test_pos.groupby("user_id")["movie_id"].apply(lambda s: {int(m) for m in s}).to_dict()
    )

    train_valid = train_ratings.loc[train_ratings["movie_id"].isin(valid_ids)]
    eligible_users: list[tuple[int, int, set[int]]] = []
    for uid, rel_set in sorted(test_user_rel.items()):
        u_train = train_valid.loc[train_valid["user_id"] == uid]
        if u_train.empty:
            continue
        u_sorted = u_train.sort_values(by=["rating"], ascending=False)
        seed_mid: int | None = None
        for cand_mid in u_sorted["movie_id"].astype(int).tolist():
            if cf_model.has_item(cand_mid):
                seed_mid = cand_mid
                break
        if seed_mid is None:
            continue
        cleaned_rel = {m for m in rel_set if m != seed_mid}
        if cleaned_rel:
            eligible_users.append((int(uid), seed_mid, cleaned_rel))

    if not eligible_users:
        return {}

    rng = np.random.default_rng(random_state)
    if len(eligible_users) > max_users:
        idxs = sorted(int(i) for i in rng.choice(len(eligible_users), size=max_users, replace=False))
        sampled = [eligible_users[i] for i in idxs]
    else:
        sampled = eligible_users

    demo_top_ids = [int(m) for m in demo_model.recommend(top_n=k + 1)["movie_id"].tolist()]
    pairs_by_model: dict[str, list[tuple[list[int], set[int]]]] = {
        "Demographic Filtering": [],
        "Content-Based Filtering": [],
        "Collaborative Filtering": [],
        "Hybrid Recommendation": [],
    }

    for _uid, seed_mid, rel_set in sampled:
        pairs_by_model["Demographic Filtering"].append(([m for m in demo_top_ids if m != seed_mid][:k], rel_set))
        cb_recs = [int(m) for m in cb_model.recommend(identifier=seed_mid, top_n=k)["movie_id"].tolist()]
        pairs_by_model["Content-Based Filtering"].append((cb_recs, rel_set))
        cf_recs = [int(m) for m in cf_model.recommend(movie_id=seed_mid, movies_df=movies_df, top_n=k)["movie_id"].tolist()]
        pairs_by_model["Collaborative Filtering"].append((cf_recs, rel_set))
        hyb_recs = [item.movie_id for item in hyb_model.recommend(identifier=seed_mid, top_n=k)]
        pairs_by_model["Hybrid Recommendation"].append((hyb_recs, rel_set))

    novelty_map = build_item_self_information_map(movies_df)
    catalog_size = len(movies_df)
    metrics_out = {
        name: _compute_metrics_for_pairs(
            model_name=name,
            pairs=pairs,
            k=k,
            catalog_size=catalog_size,
            content_model=cb_model,
            novelty_map=novelty_map,
        ).to_dict()
        for name, pairs in pairs_by_model.items()
    }
    cf_vs_cb = compute_paired_bootstrap_comparison(
        baseline_pairs=pairs_by_model["Content-Based Filtering"],
        candidate_pairs=pairs_by_model["Collaborative Filtering"],
        k=k,
        n_resamples=3000,
        random_state=random_state,
        content_model=cb_model,
    )
    hyb_vs_cb = compute_paired_bootstrap_comparison(
        baseline_pairs=pairs_by_model["Content-Based Filtering"],
        candidate_pairs=pairs_by_model["Hybrid Recommendation"],
        k=k,
        n_resamples=3000,
        random_state=random_state,
        content_model=cb_model,
    )
    return {
        "evaluation_type": "observed_user_interactions_holdout",
        "evaluated_users": len(sampled),
        "metrics": metrics_out,
        "paired_bootstrap_cf_vs_cb": cf_vs_cb,
        "paired_bootstrap_hybrid_vs_cb": hyb_vs_cb,
    }


def evaluate_catalog_models(
    movies_df: pd.DataFrame,
    k: int = 10,
    max_queries: int = 200,
    random_state: int = 42,
    ratings_df: pd.DataFrame | None = None,
    hybrid_weights: HybridWeights | None = None,
    queries_override: Sequence[tuple[int, set[int]]] | None = None,
    require_genre_overlap: bool = False,
) -> dict[str, ModelEvaluationMetrics]:
    """
    Run a leakage-free quantitative benchmark comparing Demographic, Content-Based,
    Collaborative (when real user ratings exist), and Hybrid models.

    Anti-Leakage Protocol (v2.1.0):
    - Relevance Oracle: Ground-truth sets R_meta(q) use held-out `collection_name` strictly
      (`require_genre_overlap=False` by default, removing `genres_list` overlap circularity).
    - Feature Holdout: Content-Based and Hybrid use `ContentFeatureWeights(collection=0)`.
    - Train/Val/Test Isolation: `ratings_df` is split strictly (`train`/`val`/`test`) so
      CollaborativeRecommender only ever fits on `train_ratings`. When `queries_override` is
      omitted, test queries are sampled strictly disjointly from the validation split.
    """
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

    collab_model: CollaborativeRecommender | None = None
    if ratings_df is not None and not ratings_df.empty:
        train_ratings, _val_ratings, _test_ratings = split_ratings_train_val_test(
            ratings_df, random_state=random_state
        )
        fitted_collab = CollaborativeRecommender(random_state=random_state).fit(
            movies_df, ratings_df=train_ratings
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
    ).fit(movies_df)

    if queries_override is not None:
        sampled_queries = list(queries_override)
    else:
        query_records = build_relevance_oracle(
            movies_df, min_group_size=3, require_genre_overlap=require_genre_overlap
        )
        if not query_records:
            raise ValueError("Dataset does not contain sufficient collection groups for holdout evaluation.")
        _val_q, sampled_queries = split_oracle_queries(
            query_records,
            val_size=min(100, max(1, len(query_records) // 4)),
            test_size=max_queries,
            random_state=random_state,
        )
        if not sampled_queries:
            sampled_queries = _val_q

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
    novelty_map = build_item_self_information_map(movies_df)
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
            novelty_map=novelty_map,
        )

    return summary


def evaluate_cold_start_stratification(
    movies_df: pd.DataFrame,
    test_queries: Sequence[tuple[int, set[int]]],
    k: int = 10,
    random_state: int = 42,
    ratings_df: pd.DataFrame | None = None,
    hybrid_weights: HybridWeights | None = None,
) -> dict[str, Any]:
    """
    Compare Content-Based and Hybrid performance stratified into cold-start vs warm query regimes.
    When ratings exist, cold-start queries are items without collaborative coverage; otherwise,
    items in the bottom quartile of `vote_count` among test queries.
    """
    if not test_queries:
        return {}

    eval_weights = ContentFeatureWeights(overview=2, genres=3, tagline=2, collection=0, production=1, language=1)
    demo_model = DemographicRecommender(quantile=0.90).fit(movies_df)
    cb_model = ContentBasedRecommender(weights=eval_weights).fit(movies_df)
    collab_model: CollaborativeRecommender | None = None
    if ratings_df is not None and not ratings_df.empty:
        train_r, _, _ = split_ratings_train_val_test(ratings_df, random_state=random_state)
        fitted_cf = CollaborativeRecommender(random_state=random_state).fit(movies_df, ratings_df=train_r)
        if fitted_cf.is_available:
            collab_model = fitted_cf

    hyb_weights = hybrid_weights or HybridWeights(
        content=0.75, demographic=0.15, collaborative=0.10 if collab_model is not None else 0.0
    )
    hyb_model = HybridRecommender(
        content_model=cb_model,
        demographic_model=demo_model,
        collaborative_model=collab_model,
        weights=hyb_weights,
    ).fit(movies_df)

    vote_map = dict(zip(movies_df["movie_id"].astype(int), movies_df["vote_count"].astype(int), strict=True))
    if collab_model is not None:
        cold_queries = [(q, rel) for q, rel in test_queries if not collab_model.has_item(q)]
        warm_queries = [(q, rel) for q, rel in test_queries if collab_model.has_item(q)]
        criterion = "collaborative_coverage"
    else:
        q_votes = [vote_map.get(q, 0) for q, _ in test_queries]
        q25 = float(np.percentile(q_votes, 25))
        cold_queries = [(q, rel) for q, rel in test_queries if vote_map.get(q, 0) <= q25]
        warm_queries = [(q, rel) for q, rel in test_queries if vote_map.get(q, 0) > q25]
        criterion = "bottom_quartile_vote_count"

    novelty_map = build_item_self_information_map(movies_df)
    catalog_size = len(movies_df)

    def _eval_subset(subset: Sequence[tuple[int, set[int]]]) -> dict[str, Any]:
        if not subset:
            return {"queries": 0}
        cb_pairs = [
            ([int(m) for m in cb_model.recommend(identifier=q, top_n=k)["movie_id"].tolist()], rel)
            for q, rel in subset
        ]
        hy_pairs = [
            ([item.movie_id for item in hyb_model.recommend(identifier=q, top_n=k)], rel)
            for q, rel in subset
        ]
        return {
            "queries": len(subset),
            "content_based": _compute_metrics_for_pairs("CB", cb_pairs, k, catalog_size, cb_model, novelty_map).to_dict(),
            "hybrid": _compute_metrics_for_pairs("Hybrid", hy_pairs, k, catalog_size, cb_model, novelty_map).to_dict(),
        }

    return {
        "criterion": criterion,
        "cold_start": _eval_subset(cold_queries),
        "warm": _eval_subset(warm_queries),
    }


def evaluate_hybrid_weight_sensitivity(
    movies_df: pd.DataFrame,
    weight_configs: Sequence[tuple[float, float]] | None = None,
    k: int = 10,
    max_queries: int = 200,
    random_state: int = 42,
    queries_override: Sequence[tuple[int, set[int]]] | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate sensitivity of the Hybrid Recommender across multiple (w_content, w_demographic)
    weight ratios on a designated query set (validation queries by default).
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

    if queries_override is not None:
        sampled_queries = list(queries_override)
    else:
        query_records = build_relevance_oracle(movies_df, min_group_size=3, require_genre_overlap=False)
        if not query_records:
            return []
        sampled_queries, _ = split_oracle_queries(
            query_records,
            val_size=min(max_queries, len(query_records)),
            test_size=max(0, len(query_records) - max_queries),
            random_state=random_state,
        )

    catalog_size = len(movies_df)
    novelty_map = build_item_self_information_map(movies_df)
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
            novelty_map=novelty_map,
        )
        row = metrics.to_dict()
        row["weight_content"] = round(w_c, 2)
        row["weight_demographic"] = round(w_d, 2)
        sensitivity_rows.append(row)

    return sensitivity_rows
