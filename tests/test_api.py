from fastapi.testclient import TestClient

from app.main import app

VALID_MATCH = {
    "abstract": "We study transformer architectures for low-resource neural machine translation and evaluation benchmarks.",
    "top_n": 5,
}


def test_health_returns_ok():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_returns_name():
    with TestClient(app) as client:
        response = client.get("/api/v1/")
    assert response.status_code == 200
    assert response.json()["name"] == "ReviewerMatch API"


def test_health_full():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "authors_in_db" in body
    assert "index_built" in body
    assert "faiss_vectors" in body


def test_match_requires_api_key():
    with TestClient(app) as client:
        response = client.post("/api/v1/match", json=VALID_MATCH)
    assert response.status_code == 403


def test_match_wrong_api_key():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/match",
            json=VALID_MATCH,
            headers={"X-API-Key": "wrong"},
        )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid API key"


def test_match_abstract_too_short():
    with TestClient(app) as client:
        bad = {**VALID_MATCH, "abstract": "too short"}
        response = client.post(
            "/api/v1/match", json=bad, headers={"X-API-Key": "changeme"}
        )
    assert response.status_code == 422


def test_match_valid_request_returns_results():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/match", json=VALID_MATCH, headers={"X-API-Key": "changeme"}
        )
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert "query_summary" in body
    assert "total_matched" in body
    assert "processing_time_ms" in body
    assert "data_freshness" in body
    assert isinstance(body["results"], list)
    for row in body["results"]:
        assert 0.0 <= row["score"] <= 100.0
        assert 0.0 <= row["similarity"] <= 1.0
        assert "matching_works" in row
        assert "profile_url" in row


def test_match_accepts_query_field():
    payload = {
        "query": (
            "We study transformer architectures for low-resource neural machine "
            "translation and evaluation benchmarks."
        ),
        "top_n": 5,
    }
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/match", json=payload, headers={"X-API-Key": "changeme"}
        )
    assert response.status_code == 200


def test_match_respects_top_n():
    with TestClient(app) as client:
        payload = {**VALID_MATCH, "top_n": 3}
        response = client.post(
            "/api/v1/match", json=payload, headers={"X-API-Key": "changeme"}
        )
    assert response.status_code == 200
    assert len(response.json()["results"]) <= 3


def test_author_detail_not_found():
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/author/99999", headers={"X-API-Key": "changeme"}
        )
    assert response.status_code == 404


def test_author_requires_api_key():
    with TestClient(app) as client:
        response = client.get("/api/v1/author/1")
    assert response.status_code == 403
