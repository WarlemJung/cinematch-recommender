"""Integration tests for CineMatch Flask Application Factory, WSGI entrypoint, REST endpoints, and resilience."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from flask import Flask
from flask.testing import FlaskClient

from cinematch.config import AppConfig
from cinematch.data.loader import LoadedDataset
from cinematch.main import create_app, main as run_dev_main
import cinematch.wsgi as wsgi_module


@pytest.fixture()
def app(test_app_config: AppConfig, cleaned_dataset: LoadedDataset) -> Flask:
    """Create a Flask application instance wired with the synthetic test dataset."""
    return create_app(config=test_app_config, loaded_dataset=cleaned_dataset)


@pytest.fixture()
def client(app: Flask) -> FlaskClient:
    """Return a Flask test client."""
    return app.test_client()


def test_application_factory_environments_and_wsgi_entrypoint(
    tmp_path: Path,
    raw_movies_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "movies_metadata.csv"
    raw_movies_dataframe.to_csv(csv_path, index=False)

    # 1. Production environment initialization from disk
    prod_cfg = AppConfig(
        env="production",
        debug=False,
        dataset_path=csv_path,
        cors_origins=["https://cinematch.example.com"],
    )
    prod_app = create_app(config=prod_cfg)
    assert prod_app.config["RECOMMENDATION_SERVICE"] is not None
    prod_client = prod_app.test_client()
    res = prod_client.get("/api/health", headers={"Origin": "https://cinematch.example.com"})
    assert res.status_code == 200
    assert res.headers.get("Access-Control-Allow-Origin") == "https://cinematch.example.com"

    # 2. Development environment initialization
    dev_cfg = AppConfig(
        env="development",
        debug=True,
        log_json=True,
        dataset_path=csv_path,
    )
    dev_app = create_app(config=dev_cfg)
    assert dev_app.config["APP_CONFIG"].debug is True

    # 3. WSGI entrypoint singleton behavior (no duplicate model loading in same process)
    monkeypatch.setenv("CINEMATCH_ENV", "production")
    monkeypatch.setenv("CINEMATCH_DEBUG", "false")
    monkeypatch.setenv("CINEMATCH_DATASET_PATH", str(csv_path))

    wsgi_app_1 = wsgi_module.get_wsgi_app(force_reload=True)
    wsgi_app_2 = wsgi_module.get_wsgi_app(force_reload=False)
    wsgi_attr_app = wsgi_module.app
    assert wsgi_app_1 is wsgi_app_2
    assert wsgi_attr_app is wsgi_app_1
    assert wsgi_app_1.test_client().get("/api/health").status_code == 200

    with pytest.raises(AttributeError):
        _ = wsgi_module.nonexistent_attribute

    # 4. Local development main() runner (mocking Flask.run so no blocking socket starts)
    ran_args: dict[str, object] = {}

    def fake_run(self: Flask, host: str, port: int, debug: bool) -> None:
        ran_args["host"] = host
        ran_args["port"] = port
        ran_args["debug"] = debug

    monkeypatch.setattr(Flask, "run", fake_run)
    run_dev_main()
    assert ran_args["debug"] is False


def test_application_factory_resilience_when_model_init_fails(
    test_app_config: AppConfig,
    cleaned_dataset: LoadedDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_init(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Simulated matrix allocation failure")

    monkeypatch.setattr("cinematch.main.RecommendationService.__init__", broken_init)
    degraded_app = create_app(config=test_app_config, loaded_dataset=cleaned_dataset)
    assert degraded_app.config["RECOMMENDATION_SERVICE"] is None

    res = degraded_app.test_client().get("/api/health")
    assert res.status_code == 503
    assert res.get_json()["status"] == "degraded"


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
    assert data["version"] == "2.1.0"
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

    anim_res = client.get("/api/trending?genre=Animation")
    assert anim_res.status_code == 200
    anim_items = anim_res.get_json()["items"]
    assert all("Animation" in item["genres"] for item in anim_items)

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
    res_id = client.get("/api/recommend/862?top_n=3")
    assert res_id.status_code == 200
    data_id = res_id.get_json()
    assert data_id["source_movie"]["movie_id"] == 862
    assert len(data_id["items"]) == 3
    assert all(item["movie_id"] != 862 for item in data_id["items"])

    res_dup = client.get("/api/recommend?title=Heat&top_n=3")
    assert res_dup.status_code == 200
    data_dup = res_dup.get_json()
    assert data_dup["source_movie"]["movie_id"] == 949

    res_missing = client.get("/api/recommend")
    assert res_missing.status_code == 400

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

    res_q = client.get("/api/recommendations/hybrid?title=Pulp+Fiction&top_n=2")
    assert res_q.status_code == 200
    assert res_q.get_json()["source_movie"]["movie_id"] == 680


def test_concurrent_wsgi_singleton_and_api_requests(
    tmp_path: Path,
    raw_movies_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    csv_path = tmp_path / "movies_metadata.csv"
    raw_movies_dataframe.to_csv(csv_path, index=False)
    monkeypatch.setenv("CINEMATCH_ENV", "production")
    monkeypatch.setenv("CINEMATCH_DEBUG", "false")
    monkeypatch.setenv("CINEMATCH_DATASET_PATH", str(csv_path))

    wsgi_module.get_wsgi_app(force_reload=True)

    def _call_concurrent(idx: int) -> tuple[int, int]:
        w_app = wsgi_module.get_wsgi_app(force_reload=False)
        res = w_app.test_client().get(
            "/api/recommendations/hybrid/862?top_n=3"
            if idx % 2 == 0
            else "/api/movies/search?q=Heat"
        )
        return id(w_app), res.status_code

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_call_concurrent, range(18)))

    app_ids = {app_id for app_id, _ in results}
    statuses = [status for _, status in results]
    assert len(app_ids) == 1
    assert all(status == 200 for status in statuses)
