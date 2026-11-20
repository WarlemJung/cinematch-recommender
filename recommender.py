"""
Backward-compatible adapter for MovieRecommenderSystem powered by the modular CineMatch engine.

Uses sparse TF-IDF matrices and on-demand cosine similarity (never materializing a dense N x N matrix)
and enforces full data validation and reproducible sampling.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cinematch.data.loader import MovieDataLoader  # noqa: E402
from cinematch.models.content_based import ContentBasedRecommender  # noqa: E402
from cinematch.models.demographic import DemographicRecommender  # noqa: E402
from cinematch.models.hybrid import HybridRecommendationItem, HybridRecommender  # noqa: E402
from cinematch.utils.exceptions import MovieNotFoundError  # noqa: E402


class MovieRecommenderSystem:
    """
    Facade preserving the original `MovieRecommenderSystem` API while delegating to the
    memory-safe, modular `cinematch` package.
    """

    def __init__(
        self,
        filepath: str | Path,
        sample_size: int | None = None,
        random_state: int = 42,
    ) -> None:
        loader = MovieDataLoader(
            dataset_path=filepath,
            sample_size=sample_size,
            random_state=random_state,
        )
        self.loaded = loader.load()
        self.df = self.loaded.movies_df
        self.quality_report = self.loaded.quality_report

        self.demographic_model = DemographicRecommender().fit(self.df)
        self.content_model = ContentBasedRecommender()
        self.hybrid_model: HybridRecommender | None = None

    def get_demographic_recommendations(self, top_n: int = 10) -> pd.DataFrame:
        """Return top_n movies ranked by the IMDB Weighted Rating formula."""
        ranked = self.demographic_model.recommend(top_n=top_n)
        cols = [c for c in ("movie_id", "title", "vote_count", "vote_average", "score", "release_year") if c in ranked.columns]
        return ranked[cols]

    def compute_content_similarity(self) -> None:
        """
        Fit the sparse TF-IDF matrix without materializing a dense N x N similarity matrix.
        """
        self.content_model.fit(self.df)
        self.hybrid_model = HybridRecommender(
            content_model=self.content_model,
            demographic_model=self.demographic_model,
        ).fit(self.df)

    def get_content_recommendations(self, title: str | int, top_n: int = 10) -> pd.Series:
        """
        Return top_n most similar movie titles using on-demand sparse linear_kernel.
        Handles duplicate titles safely without crashing.
        """
        if not self.content_model.is_fitted:
            self.compute_content_similarity()

        recs_df = self.content_model.recommend(identifier=title, top_n=top_n)
        return recs_df["title"]

    def get_hybrid_recommendations(self, identifier: str | int, top_n: int = 10) -> list[HybridRecommendationItem]:
        """Return explainable hybrid recommendations combining content and demographic signals."""
        if not self.content_model.is_fitted or self.hybrid_model is None:
            self.compute_content_similarity()
        assert self.hybrid_model is not None
        return self.hybrid_model.recommend(identifier=identifier, top_n=top_n)


if __name__ == "__main__":
    default_csv = Path(__file__).resolve().parent / "movies_metadata.csv"
    recsys = MovieRecommenderSystem(default_csv)
    print("Data Quality Report:", recsys.quality_report.to_dict())
    print("\n--- Top Demographic Recommendations ---")
    print(recsys.get_demographic_recommendations(top_n=10))
    recsys.compute_content_similarity()
    try:
        print("\n--- Hybrid Recommendations for 'Toy Story' ---")
        for item in recsys.get_hybrid_recommendations("Toy Story", top_n=5):
            print(f"- {item.title} ({item.release_year}) [Score: {item.final_score:.3f}]: {item.explanation}")
    except MovieNotFoundError as err:
        print(err.message)
