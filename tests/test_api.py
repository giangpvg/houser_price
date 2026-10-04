"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthcheck():
    """Verify /health returns HTTP 200 with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["service"] == "House Price Prediction AI"


def test_get_model_info():
    """Verify /api/model/info returns model metrics and coefficients."""
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "features" in data
    assert "metadata" in data
    assert "r2_score" in data["metadata"]["metrics"]


def test_predict_single_success():
    """Verify valid house prediction returns expected fields."""
    payload = {
        "area_m2": 85.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "distance_to_center_km": 4.5,
        "house_age_years": 3.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_billion_vnd" in data
    assert data["predicted_price_billion_vnd"] > 0
    assert "price_per_m2_million_vnd" in data
    assert "feature_contributions" in data
    assert "prediction_interval" in data


def test_predict_validation_error():
    """Verify negative or invalid area triggers 422 Unprocessable Entity."""
    invalid_payload = {
        "area_m2": -50.0,  # Invalid: gt=10.0 required
        "bedrooms": 3,
        "bathrooms": 2,
        "distance_to_center_km": 4.5,
        "house_age_years": 3.0,
    }
    response = client.post("/api/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_batch():
    """Verify batch predictions endpoint."""
    batch_payload = {
        "properties": [
            {
                "area_m2": 60.0,
                "bedrooms": 2,
                "bathrooms": 1,
                "distance_to_center_km": 8.0,
                "house_age_years": 5.0,
            },
            {
                "area_m2": 150.0,
                "bedrooms": 4,
                "bathrooms": 3,
                "distance_to_center_km": 2.0,
                "house_age_years": 1.0,
            },
        ]
    }
    response = client.post("/api/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
