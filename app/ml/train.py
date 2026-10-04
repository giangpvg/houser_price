"""Training script for Housing Price Prediction using Linear Regression.

Generates realistic housing data, trains an Ordinary Least Squares (OLS)
Linear Regression model, evaluates performance metrics, and serializes
the trained model artifact using joblib.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_NAMES = [
    "area_m2",
    "bedrooms",
    "bathrooms",
    "distance_to_center_km",
    "house_age_years",
]


def generate_synthetic_housing_data(
    n_samples: int = 1500, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic housing price data in Vietnam metro context.

    Features:
    - area_m2: 30m2 to 250m2
    - bedrooms: 1 to 5 rooms
    - bathrooms: 1 to 4 rooms
    - distance_to_center_km: 1km to 25km
    - house_age_years: 0 to 30 years

    Target:
    - price_billion_vnd: Price in billion VND
      Formula base: 0.8 + 0.052*area + 0.25*bed + 0.18*bath - 0.10*dist - 0.035*age + noise
    """
    rng = np.random.RandomState(random_state)

    area = rng.uniform(30.0, 250.0, size=n_samples)
    bedrooms = rng.randint(1, 6, size=n_samples)
    bathrooms = rng.randint(1, 5, size=n_samples)
    distance = rng.uniform(0.8, 25.0, size=n_samples)
    age = rng.uniform(0.0, 30.0, size=n_samples)

    # Realistic relationship with stochastic Gaussian error term
    noise = rng.normal(0, 0.22, size=n_samples)
    price = (
        0.85
        + (0.054 * area)
        + (0.28 * bedrooms)
        + (0.20 * bathrooms)
        - (0.11 * distance)
        - (0.038 * age)
        + noise
    )

    # Ensure all prices remain strictly positive
    price = np.clip(price, a_min=0.5, a_max=None)

    X = np.column_stack([area, bedrooms, bathrooms, distance, age])
    y = price

    return X, y


def train_and_save_model(
    n_samples: int = 1500, random_state: int = 42
) -> Dict[str, Any]:
    """Train Linear Regression model and serialize artifact."""
    X, y = generate_synthetic_housing_data(
        n_samples=n_samples, random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # Predictions and metrics evaluation
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    coefficients = {
        name: round(float(coef), 5)
        for name, coef in zip(FEATURE_NAMES, model.coef_)
    }
    intercept = round(float(model.intercept_), 5)

    metadata: Dict[str, Any] = {
        "model_type": "LinearRegression (Ordinary Least Squares)",
        "features": FEATURE_NAMES,
        "coefficients": coefficients,
        "intercept": intercept,
        "metrics": {
            "r2_score": round(r2, 4),
            "mae_billion_vnd": round(mae, 4),
            "rmse_billion_vnd": round(rmse, 4),
            "test_sample_size": len(y_test),
            "train_sample_size": len(y_train),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save model binary and metadata
    artifact = {
        "model": model,
        "metadata": metadata,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Model saved successfully to: {MODEL_PATH}")
    print(
        f"Metrics: R2={r2:.4f}, MAE={mae:.4f} tỷ VNĐ, RMSE={rmse:.4f} tỷ VNĐ"
    )
    print(f"Intercept: {intercept}, Coefficients: {coefficients}")

    return metadata


if __name__ == "__main__":
    train_and_save_model()
