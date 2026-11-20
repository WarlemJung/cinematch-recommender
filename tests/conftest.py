"""Shared pytest fixtures using small synthetic datasets exclusively for fast, isolated testing."""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cinematch.config import AppConfig  # noqa: E402
from cinematch.data.loader import LoadedDataset  # noqa: E402
from cinematch.data.preprocessing import preprocess_movies_dataframe  # noqa: E402


@pytest.fixture()
def raw_movies_dataframe() -> pd.DataFrame:
    """Return a small synthetic raw DataFrame including edge cases (duplicate titles, missing overview, etc.)."""
    records = [
        {
            "id": "862",
            "imdb_id": "tt0114709",
            "title": "Toy Story",
            "overview": "Led by Woody, Andy's toys live happily in his room until Buzz Lightyear arrives.",
            "tagline": "The adventure takes off!",
            "genres": "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]",
            "belongs_to_collection": "{'id': 10194, 'name': 'Toy Story Collection'}",
            "production_companies": "[{'name': 'Pixar Animation Studios', 'id': 3}]",
            "release_date": "1995-10-30",
            "vote_average": 7.7,
            "vote_count": 5415,
            "popularity": 21.95,
            "runtime": 81.0,
            "original_language": "en",
            "poster_path": "/rhIRbceoE9lR4veEXuwCC2wARtG.jpg",
        },
        {
            "id": "863",
            "imdb_id": "tt0120363",
            "title": "Toy Story 2",
            "overview": "Andy heads off to Cowboy Camp, leaving his toys to their own devices. Woody is kidnapped.",
            "tagline": "The toys are back!",
            "genres": "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]",
            "belongs_to_collection": "{'id': 10194, 'name': 'Toy Story Collection'}",
            "production_companies": "[{'name': 'Pixar Animation Studios', 'id': 3}]",
            "release_date": "1999-10-30",
            "vote_average": 7.3,
            "vote_count": 3914,
            "popularity": 17.55,
            "runtime": 92.0,
            "original_language": "en",
            "poster_path": "/3CmK3XurcLeUyMifCR28ibzupbB.jpg",
        },
        {
            "id": "10193",
            "imdb_id": "tt0435761",
            "title": "Toy Story 3",
            "overview": "Woody, Buzz, and the rest of Andy's toys are mistakenly delivered to a day-care center.",
            "tagline": "No toy gets left behind.",
            "genres": "[{'id': 16, 'name': 'Animation'}, {'id': 10751, 'name': 'Family'}, {'id': 35, 'name': 'Comedy'}]",
            "belongs_to_collection": "{'id': 10194, 'name': 'Toy Story Collection'}",
            "production_companies": "[{'name': 'Pixar Animation Studios', 'id': 3}]",
            "release_date": "2010-06-16",
            "vote_average": 7.6,
            "vote_count": 4710,
            "popularity": 16.96,
            "runtime": 103.0,
            "original_language": "en",
            "poster_path": "/mMltbSxwEdNE4Cv8QYLpzkHWTDo.jpg",
        },
        {
            "id": "949",
            "imdb_id": "tt0113277",
            "title": "Heat",
            "overview": "Obsessive master thief Neil McCauley leads a top-notch crew on high-stakes heists in Los Angeles.",
            "tagline": "A Los Angeles Crime Saga",
            "genres": "[{'id': 28, 'name': 'Action'}, {'id': 80, 'name': 'Crime'}, {'id': 18, 'name': 'Drama'}]",
            "belongs_to_collection": "{'id': 9900, 'name': 'Crime Classics Collection'}",
            "production_companies": "[{'name': 'Warner Bros.', 'id': 6194}]",
            "release_date": "1995-12-15",
            "vote_average": 7.7,
            "vote_count": 1886,
            "popularity": 17.92,
            "runtime": 170.0,
            "original_language": "en",
            "poster_path": "/zMyfPUelumio3tiDKPffaUpsQTD.jpg",
        },
        {
            "id": "20776",
            "imdb_id": "tt0068688",
            "title": "Heat",
            "overview": "An ex-child star attempts to revive his acting career in Hollywood.",
            "tagline": "",
            "genres": "[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}]",
            "belongs_to_collection": "",
            "production_companies": "[]",
            "release_date": "1972-10-05",
            "vote_average": 5.6,
            "vote_count": 18,
            "popularity": 1.45,
            "runtime": 100.0,
            "original_language": "en",
            "poster_path": "/invalid_script_tag.exe",
        },
        {
            "id": "500",
            "imdb_id": "tt0105236",
            "title": "Reservoir Dogs",
            "overview": "A botched robbery indicates a police informant, and the pressure mounts in the aftermath at a warehouse.",
            "tagline": "Every dog has his day.",
            "genres": "[{'id': 80, 'name': 'Crime'}, {'id': 53, 'name': 'Thriller'}]",
            "belongs_to_collection": "{'id': 9900, 'name': 'Crime Classics Collection'}",
            "production_companies": "[{'name': 'Live Entertainment', 'id': 285}]",
            "release_date": "1992-09-02",
            "vote_average": 8.1,
            "vote_count": 3821,
            "popularity": 12.22,
            "runtime": 99.0,
            "original_language": "en",
            "poster_path": "/AjTtJNumZyUDz33VtMlF1K8JPsE.jpg",
        },
        {
            "id": "680",
            "imdb_id": "tt0110912",
            "title": "Pulp Fiction",
            "overview": "A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge in Los Angeles crime stories.",
            "tagline": "You won't know the facts until you've seen the fiction.",
            "genres": "[{'id': 53, 'name': 'Thriller'}, {'id': 80, 'name': 'Crime'}]",
            "belongs_to_collection": "{'id': 9900, 'name': 'Crime Classics Collection'}",
            "production_companies": "[{'name': 'Miramax Films', 'id': 14}]",
            "release_date": "1994-09-10",
            "vote_average": 8.3,
            "vote_count": 8670,
            "popularity": 140.95,
            "runtime": 154.0,
            "original_language": "en",
            "poster_path": "/dM2w364MScsjFf8pfMbaWUcWrR.jpg",
        },
        {
            "id": "777",
            "imdb_id": "tt0000777",
            "title": "Silent Mystery Without Synopsis",
            "overview": None,
            "tagline": None,
            "genres": "[{'id': 9648, 'name': 'Mystery'}]",
            "belongs_to_collection": None,
            "production_companies": "[]",
            "release_date": "invalid-date",
            "vote_average": 6.8,
            "vote_count": 45,
            "popularity": 2.1,
            "runtime": 88.0,
            "original_language": "fr",
            "poster_path": None,
        },
    ]
    return pd.DataFrame(records)


@pytest.fixture()
def cleaned_dataset(raw_movies_dataframe: pd.DataFrame) -> LoadedDataset:
    """Return a preprocessed LoadedDataset built from the synthetic fixture."""
    movies_df, report = preprocess_movies_dataframe(raw_movies_dataframe)
    return LoadedDataset(movies_df=movies_df, quality_report=report, ratings_df=None)


@pytest.fixture()
def synthetic_ratings_dataframe() -> pd.DataFrame:
    """Return real-structured user-item interaction logs for testing Collaborative Filtering."""
    interactions = [
        {"user_id": 1, "movie_id": 862, "rating": 5.0},
        {"user_id": 1, "movie_id": 863, "rating": 4.5},
        {"user_id": 1, "movie_id": 10193, "rating": 5.0},
        {"user_id": 1, "movie_id": 949, "rating": 2.0},
        {"user_id": 2, "movie_id": 862, "rating": 4.5},
        {"user_id": 2, "movie_id": 863, "rating": 5.0},
        {"user_id": 2, "movie_id": 10193, "rating": 4.5},
        {"user_id": 2, "movie_id": 500, "rating": 1.5},
        {"user_id": 3, "movie_id": 862, "rating": 5.0},
        {"user_id": 3, "movie_id": 863, "rating": 4.0},
        {"user_id": 3, "movie_id": 680, "rating": 2.0},
        {"user_id": 5, "movie_id": 949, "rating": 5.0},
        {"user_id": 5, "movie_id": 500, "rating": 4.5},
        {"user_id": 5, "movie_id": 680, "rating": 5.0},
        {"user_id": 5, "movie_id": 862, "rating": 2.0},
        {"user_id": 6, "movie_id": 949, "rating": 4.5},
        {"user_id": 6, "movie_id": 500, "rating": 5.0},
        {"user_id": 6, "movie_id": 680, "rating": 4.5},
        {"user_id": 6, "movie_id": 863, "rating": 1.5},
        {"user_id": 7, "movie_id": 949, "rating": 5.0},
        {"user_id": 7, "movie_id": 680, "rating": 5.0},
        {"user_id": 7, "movie_id": 10193, "rating": 2.0},
    ]
    return pd.DataFrame(interactions)


@pytest.fixture()
def test_app_config(tmp_path: Path) -> AppConfig:
    """Return a clean test configuration."""
    return AppConfig(
        env="testing",
        debug=False,
        dataset_path=tmp_path / "movies_metadata.csv",
        demographic_quantile=0.50,
    )
