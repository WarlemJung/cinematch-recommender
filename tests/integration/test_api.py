"""Integration tests for all CineMatch Flask REST API endpoints and error states."""

from __future__ import annotations

from pathlib import Path
import pytest
from flask import Flask
from flask.testing import FlaskClient

from cinematch.config import AppConfig
from cinematch.data.loader import LoadedDataset
from cinematch.main import create_app


@pytest.fixture()
def app(test_app_config: AppConfig, cleaned_dataset: LoadedDataset) -> Flask:
    """Create a Flask application instance wired with the synthetic test dataset."""
    return create_app(config=test_app_config, loaded_dataset=cleaned_dataset)


@pytest.fixture()
def client(app: Flask) -> FlaskClient:
    """Return a Flask test client."""
    return app.test_client()


def test_index_web_interface_and_security_headers(client: FlaskClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "CineMatch" in html
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"


def test_health_endpoint_healthy(client: FlaskClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["catalog_size"] == 8
    assert data["models"]["demographic_fitted"] is True
    assert data["models"]["content_based_fitted"] is True
    assert data["models"]["tfidf_sparse_memory_mb"] >= 0.0


def test_health_and_api_when_dataset_missing(tmp_path: Path) -> None:
    missing_config = AppConfig(
        env="testing",
        debug=False,
        dataset_path=tmp_path / "does_not_exist.csv",
    )
    degraded_app = create_app(config=missing_config, loaded_dataset=None)
    degraded_client = degraded_app.test_client()

    health_res = degraded_client.get("/api/health")
    assert health_res.status_code == 503
    assert health_res.get_json()["status"] == "degraded"

    trending_res = degraded_client.get("/api/trending")
    assert trending_res.status_code == 503
    assert trending_res.get_json()["error_code"] == "SYSTEM_NOT_INITIALIZED"


def test_trending_endpoint_pagination_and_filters(client: FlaskClient) -> None:
    res = client.get("/api/trending?top_n=4&page=1&page_size=2")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["items"]) == 2
    assert data["page"] == 1
    assert data["page_size"] == 2
    assert data["total"] >= 2

    # Filter by genre
    anim_res = client.get("/api/trending?genre=Animation")
    assert anim_res.status_code == 200
    anim_items = anim_res.get_json()["items"]
    assert all("Animation" in item["genres"] for item in anim_items)

    # Invalid year range
    bad_res = client.get("/api/trending?min_year=2020&max_year=1990")
    assert bad_res.status_code == 400
    assert bad_res.get_json()["error_code"] == "INVALID_PARAMETER"


def test_genres_endpoint(client: FlaskClient) -> None:
    res = client.get("/api/genres")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] > 0
    names = [g["name"] for g in data["items"]]
    assert "Animation" in names
    assert "Crime" in names


def test_search_movies_endpoint(client: FlaskClient) -> None:
    res = client.get("/api/movies/search?q=Toy+Story")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] == 3
    assert data["items"][0]["title"] == "Toy Story"

    # Invalid page parameter
    bad_res = client.get("/api/movies/search?page=0")
    assert bad_res.status_code == 400
    assert bad_res.get_json()["error_code"] == "VALIDATION_ERROR"


def test_movie_details_endpoint(client: FlaskClient) -> None:
    res = client.get("/api/movies/862")
    assert res.status_code == 200
    data = res.get_json()
    assert data["movie_id"] == 862
    assert data["title"] == "Toy Story"
    assert data["release_year"] == 1995

    not_found = client.get("/api/movies/404404")
    assert not_found.status_code == 404
    assert not_found.get_json()["error_code"] == "MOVIE_NOT_FOUND"


def test_content_recommend_endpoints_by_id_and_title(client: FlaskClient) -> None:
    # By movie_id
    res_id = client.get("/api/recommend/862?top_n=3")
    assert res_id.status_code == 200
    data_id = res_id.get_json()
    assert data_id["source_movie"]["movie_id"] == 862
    assert len(data_id["items"]) == 3
    assert all(item["movie_id"] != 862 for item in data_id["items"])

    # By duplicate title 'Heat' (must return 200 instead of 500!)
    res_dup = client.get("/api/recommend?title=Heat&top_n=3")
    assert res_dup.status_code == 200
    data_dup = res_dup.get_json()
    assert data_dup["source_movie"]["movie_id"] == 949

    # Missing parameter
    res_missing = client.get("/api/recommend")
    assert res_missing.status_code == 400

    # Excessive top_n (> 50)
    res_limit = client.get("/api/recommend/862?top_n=500")
    assert res_limit.status_code == 400
    assert res_limit.get_json()["error_code"] == "VALIDATION_ERROR"


def test_hybrid_recommendations_endpoints(client: FlaskClient) -> None:
    res = client.get("/api/recommendations/hybrid/862?top_n=3")
    assert res.status_code == 200
    data = res.get_json()
    assert data["algorithm"] == "hybrid_weighted_ensemble"
    assert data["source_movie"]["movie_id"] == 862
    assert len(data["items"]) == 3

    first_item = data["items"][0]
    assert "final_score" in first_item
    assert "individual_scores" in first_item
    assert "explanation" in first_item
    assert first_item["explanation"].startswith("Recomendado porque")

    # Hybrid by query title
    res_q = client.get("/api/recommendations/hybrid?title=Pulp+Fiction&top_n=2")
    assert res_q.status_code == 200
    assert res_q.get_json()["source_movie"]["movie_id"] == 680
