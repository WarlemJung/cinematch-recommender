"""Hybrid Movie Recommendation Engine combining Content-Based, Demographic, and Collaborative models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd

from cinematch.models.collaborative import CollaborativeRecommender
from cinematch.models.content_based import ContentBasedRecommender
from cinematch.models.demographic import DemographicRecommender
from cinematch.utils.exceptions import InvalidParameterError, ModelNotTrainedError


@dataclass(frozen=True)
class HybridWeights:
    """Configurable base weights for the Hybrid Recommender."""

    content: float = 0.75
    demographic: float = 0.15
    collaborative: float = 0.10

    def normalize(self, include_collaborative: bool) -> dict[str, float]:
        """
        Return normalized weights summing to 1.0.
        Automatically redistributes collaborative weight when collaborative data is unavailable.
        """
        w_c = max(0.0, float(self.content))
        w_d = max(0.0, float(self.demographic))
        w_cf = max(0.0, float(self.collaborative)) if include_collaborative else 0.0

        total = w_c + w_d + w_cf
        if total <= 0.0:
            raise InvalidParameterError("At least one active hybrid recommendation weight must be positive.")

        return {
            "content": round(w_c / total, 4),
            "demographic": round(w_d / total, 4),
            "collaborative": round(w_cf / total, 4),
        }


@dataclass(frozen=True)
class HybridRecommendationItem:
    """Structured output record for a hybrid recommendation with explainability."""

    movie_id: int
    title: str
    release_year: int | None
    release_date: str
    vote_average: float
    vote_count: int
    genres: list[str]
    overview: str
    poster_path: str | None
    final_score: float
    individual_scores: dict[str, float | None]
    active_weights: dict[str, float]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HybridRecommender:
    """
    Combines Content-Based, Demographic, and (when available) Collaborative Filtering
    with query-relative score calibration, thematic relevance gating, dynamic weight
    normalization, and transparent natural-language explanations.
    """

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        demographic_model: DemographicRecommender,
        collaborative_model: CollaborativeRecommender | None = None,
        weights: HybridWeights | None = None,
    ) -> None:
        self.content_model = content_model
        self.demographic_model = demographic_model
        self.collaborative_model = collaborative_model
        self.weights = weights or HybridWeights()
        self._df: pd.DataFrame | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            self._df is not None
            and self.content_model.is_fitted
            and self.demographic_model.is_fitted
        )

    def fit(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame | None = None,
    ) -> HybridRecommender:
        """Fit or synchronize underlying models on the catalog DataFrame without duplicate fitting."""
        if not self.demographic_model.is_fitted:
            self.demographic_model.fit(movies_df)
        if not self.content_model.is_fitted:
            self.content_model.fit(movies_df)
        if self.collaborative_model is not None and not self.collaborative_model.is_fitted:
            self.collaborative_model.fit(movies_df, ratings_df=ratings_df)

        if isinstance(movies_df.index, pd.RangeIndex) and movies_df.index.start == 0 and movies_df.index.step == 1:
            self._df = movies_df
        else:
            self._df = movies_df.reset_index(drop=True)
        return self

    def _build_explanation(
        self,
        source_row: pd.Series,
        candidate_row: pd.Series,
        content_score: float,
        demographic_score: float,
        collaborative_score: float | None,
    ) -> str:
        """
        Generate a factual explanation strictly from the attributes and scores actually used.
        """
        reasons: list[str] = []

        source_col = str(source_row.get("collection_name", "") or "").strip()
        cand_col = str(candidate_row.get("collection_name", "") or "").strip()
        if source_col and cand_col and source_col.lower() == cand_col.lower():
            reasons.append(f"faz parte da mesma franquia ({cand_col})")

        if content_score >= 0.35:
            reasons.append("apresenta alta similaridade temática e narrativa com o filme selecionado")
        elif content_score >= 0.15:
            reasons.append("apresenta afinidade temática com o enredo selecionado")

        source_genres = set(source_row.get("genres_list") or [])
        cand_genres = list(candidate_row.get("genres_list") or [])
        shared_genres = [g for g in cand_genres if g in source_genres]
        if shared_genres:
            if len(shared_genres) == 1:
                reasons.append(f"compartilha o gênero {shared_genres[0]}")
            else:
                joined = ", ".join(shared_genres[:3])
                reasons.append(f"compartilha os gêneros {joined}")

        source_studios = set(source_row.get("production_list") or [])
        cand_studios = [s for s in (candidate_row.get("production_list") or []) if s in source_studios]
        if cand_studios:
            reasons.append(f"foi produzido pelo mesmo estúdio ({cand_studios[0]})")

        if collaborative_score is not None and collaborative_score >= 0.25:
            reasons.append("demonstra forte padrão de preferência entre usuários com histórico similar")

        vote_avg = float(candidate_row.get("vote_average", 0.0))
        vote_cnt = int(candidate_row.get("vote_count", 0))
        if demographic_score >= 0.65 and vote_avg >= 7.0:
            reasons.append(f"possui excelente avaliação geral ({vote_avg:.1f}/10 em {vote_cnt} votos)")
        elif vote_avg >= 6.5:
            reasons.append(f"conta com boa avaliação geral ({vote_avg:.1f}/10)")

        if not reasons:
            reasons.append("combina relevância temática no catálogo e avaliação ponderada consistente")

        if len(reasons) == 1:
            body = reasons[0]
        else:
            body = ", ".join(reasons[:-1]) + " e " + reasons[-1]

        return f"Recomendado porque {body}."

    def recommend(
        self,
        identifier: int | str,
        top_n: int = 12,
        genre: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_rating: float | None = None,
    ) -> list[HybridRecommendationItem]:
        """
        Compute hybrid recommendations for a target movie_id or title.

        Score Calibration & Gating:
        - Raw TF-IDF cosine similarities have a compressed upper bound (~0.25-0.55) compared to
          global normalized Weighted Ratings (0.0-1.0). We scale query content scores by the
          query's maximum non-self similarity so both signals share a commensurate [0, 1] range.
        - A thematic relevance gate attenuates the demographic popularity boost for candidates
          with near-zero content and collaborative similarity, preventing unrelated global
          blockbusters from dominating personalized movie-to-movie recommendations.
        """
        if not self.is_fitted or self._df is None:
            raise ModelNotTrainedError("HybridRecommender")
        if top_n < 1:
            raise InvalidParameterError("top_n must be at least 1.")

        target_idx = self.content_model.resolve_movie_index(identifier)
        source_row = self._df.iloc[target_idx]
        target_movie_id = int(source_row["movie_id"])

        raw_content_scores = self.content_model.compute_similarity_for_index(target_idx)
        raw_content_scores[target_idx] = 0.0

        max_content = float(np.max(raw_content_scores))
        if max_content > 1e-6:
            scaled_content_scores = raw_content_scores / max_content
        else:
            scaled_content_scores = raw_content_scores.copy()

        demographic_scores = self.demographic_model.get_normalized_scores()

        collab_scores: np.ndarray | None = None
        if self.collaborative_model is not None and self.collaborative_model.has_item(target_movie_id):
            collab_scores = self.collaborative_model.predict_catalog_scores(target_movie_id)

        has_collab = collab_scores is not None
        active_weights = self.weights.normalize(include_collaborative=has_collab)

        # Thematic relevance gate: demographic score boosts relevant candidates rather than
        # promoting unrelated blockbusters with 0.0 content/collaborative similarity.
        if max_content > 1e-6:
            personalized_signal = scaled_content_scores.copy()
            if has_collab and collab_scores is not None:
                personalized_signal = np.maximum(personalized_signal, collab_scores)
            relevance_gate = np.clip(personalized_signal / 0.25, 0.0, 1.0)
        else:
            relevance_gate = np.ones_like(demographic_scores)

        gated_demographic_scores = demographic_scores * relevance_gate

        final_scores = (
            active_weights["content"] * scaled_content_scores
            + active_weights["demographic"] * gated_demographic_scores
        )
        if has_collab and collab_scores is not None:
            final_scores = final_scores + active_weights["collaborative"] * collab_scores

        # Exclude queried movie itself
        final_scores[target_idx] = -1.0

        # Apply optional user filters
        valid_mask = np.ones(len(self._df), dtype=bool)
        valid_mask[target_idx] = False

        if genre:
            g_low = genre.strip().lower()
            genre_mask = self._df["genres_list"].apply(
                lambda gl: any(str(g).lower() == g_low for g in gl)
            ).to_numpy(dtype=bool)
            valid_mask &= genre_mask

        if min_year is not None and "release_year" in self._df.columns:
            valid_mask &= (self._df["release_year"].to_numpy() >= min_year)

        if max_year is not None and "release_year" in self._df.columns:
            years = self._df["release_year"].to_numpy()
            valid_mask &= (years > 0) & (years <= max_year)

        if min_rating is not None:
            valid_mask &= (self._df["vote_average"].to_numpy() >= min_rating)

        masked_scores = np.where(valid_mask, final_scores, -1.0)
        vote_counts = self._df["vote_count"].to_numpy(dtype=float)
        order = np.lexsort((-vote_counts, -masked_scores))
        top_indices = [int(i) for i in order[:top_n] if masked_scores[i] >= 0.0]

        recommendations: list[HybridRecommendationItem] = []
        for idx in top_indices:
            cand = self._df.iloc[idx]
            c_score = round(float(raw_content_scores[idx]), 4)
            d_score = round(float(demographic_scores[idx]), 4)
            cf_score = round(float(collab_scores[idx]), 4) if (has_collab and collab_scores is not None) else None
            f_score = round(float(final_scores[idx]), 4)

            year_val = int(cand.get("release_year", 0))
            explanation = self._build_explanation(
                source_row=source_row,
                candidate_row=cand,
                content_score=c_score,
                demographic_score=d_score,
                collaborative_score=cf_score,
            )

            recommendations.append(
                HybridRecommendationItem(
                    movie_id=int(cand["movie_id"]),
                    title=str(cand["title"]),
                    release_year=year_val if year_val > 0 else None,
                    release_date=str(cand.get("release_date", "")),
                    vote_average=float(cand["vote_average"]),
                    vote_count=int(cand["vote_count"]),
                    genres=list(cand.get("genres_list") or []),
                    overview=str(cand.get("overview", "")),
                    poster_path=cand.get("poster_path"),
                    final_score=f_score,
                    individual_scores={
                        "content": c_score,
                        "demographic": d_score,
                        "collaborative": cf_score,
                    },
                    active_weights=active_weights,
                    explanation=explanation,
                )
            )

        return recommendations
