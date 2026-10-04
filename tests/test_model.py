"""Unit tests for the Machine Learning Linear Regression pipeline."""

import pytest
from app.ml.predictor import HousePricePredictor
from app.ml.train import generate_synthetic_housing_data, train_and_save_model


def test_generate_data():
    """Verify synthetic data generation dimensions and constraints."""
    X, y = generate_synthetic_housing_data(n_samples=200, random_state=123)
    assert X.shape == (200, 5)
    assert y.shape == (200,)
    assert (y > 0).all(), "Tất cả giá nhà phải lớn hơn 0"


def test_model_training_and_metrics():
    """Verify model training produces valid R2 and error metrics."""
    metadata = train_and_save_model(n_samples=300, random_state=42)
    assert "metrics" in metadata
    assert metadata["metrics"]["r2_score"] > 0.85, "R2 score phải đạt trên 0.85"
    assert metadata["metrics"]["mae_billion_vnd"] > 0


def test_predictor_single_prediction():
    """Verify single prediction output and feature contributions."""
    predictor = HousePricePredictor.get_instance()
    res = predictor.predict_one(
        area_m2=100.0,
        bedrooms=3,
        bathrooms=2,
        distance_to_center_km=5.0,
        house_age_years=2.0,
    )

    assert "predicted_price_billion_vnd" in res
    assert res["predicted_price_billion_vnd"] > 0
    assert "price_per_m2_million_vnd" in res
    assert "feature_contributions" in res

    contributions = res["feature_contributions"]
    assert "base_intercept_billion" in contributions
    assert "area_contribution_billion" in contributions
    assert contributions["area_contribution_billion"] > 0


def test_predictor_economic_logic():
    """Verify monotonic relationships: more area = more expensive; further = cheaper."""
    predictor = HousePricePredictor.get_instance()

    small_house = predictor.predict_one(
        area_m2=50.0,
        bedrooms=2,
        bathrooms=1,
        distance_to_center_km=5.0,
        house_age_years=5.0,
    )
    large_house = predictor.predict_one(
        area_m2=120.0,
        bedrooms=2,
        bathrooms=1,
        distance_to_center_km=5.0,
        house_age_years=5.0,
    )

    assert (
        large_house["predicted_price_billion_vnd"]
        > small_house["predicted_price_billion_vnd"]
    ), "Nhà diện tích lớn hơn phải có giá cao hơn"

    near_house = predictor.predict_one(
        area_m2=80.0,
        bedrooms=2,
        bathrooms=1,
        distance_to_center_km=2.0,
        house_age_years=5.0,
    )
    far_house = predictor.predict_one(
        area_m2=80.0,
        bedrooms=2,
        bathrooms=1,
        distance_to_center_km=20.0,
        house_age_years=5.0,
    )

    assert (
        near_house["predicted_price_billion_vnd"]
        > far_house["predicted_price_billion_vnd"]
    ), "Nhà gần trung tâm hơn phải có giá cao hơn nhà xa trung tâm"
