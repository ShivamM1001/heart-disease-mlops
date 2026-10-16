
"""Automated tests for the FastAPI prediction service."""

from fastapi.testclient import TestClient

from src.api import app


VALID_INPUT = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def test_health_endpoint():
    """Health endpoint reports a loaded model."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    """Prediction endpoint returns a valid prediction."""
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["prediction_label"] in ("Absence", "Presence")
    assert 0 <= result["probability_absence"] <= 1
    assert 0 <= result["probability_presence"] <= 1
    assert abs(
        result["probability_absence"]
        + result["probability_presence"] - 1
    ) < 1e-6



def test_predict_rejects_invalid_age():
    """An age outside the allowed range returns HTTP 422."""
    invalid_input = {**VALID_INPUT, "age": 150}

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_predict_rejects_unexpected_field():
    """Unexpected input fields return HTTP 422."""
    invalid_input = {**VALID_INPUT, "unexpected_field": 123}

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422

