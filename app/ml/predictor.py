"""Inference and feature decomposition engine for House Price Prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from app.ml.train import (
    FEATURE_NAMES,
    METRICS_PATH,
    MODEL_PATH,
    train_and_save_model,
)


class HousePricePredictor:
    """Predictor class managing model loading, inference, and feature contribution."""

    _instance: Optional["HousePricePredictor"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.load_or_train()

    @classmethod
    def get_instance(cls) -> "HousePricePredictor":
        """Singleton accessor for efficient reuse in web requests."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_or_train(self) -> None:
        """Load trained artifact from disk, or train a new model if absent."""
        if not self.model_path.exists():
            print(
                f"[Predictor] Artifact {self.model_path} not found. Training model now..."
            )
            self.metadata = train_and_save_model()

        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.metadata = artifact.get("metadata", {})

    def predict_one(
        self,
        area_m2: float,
        bedrooms: int,
        bathrooms: int,
        distance_to_center_km: float,
        house_age_years: float,
    ) -> Dict[str, Any]:
        """Compute predicted house price and linear feature decomposition."""
        if self.model is None:
            self.load_or_train()

        input_vector = np.array(
            [[area_m2, bedrooms, bathrooms, distance_to_center_km, house_age_years]]
        )

        raw_price = float(self.model.predict(input_vector)[0])
        # Ensure realistic minimum bounds
        predicted_price_billion = max(0.1, round(raw_price, 3))
        predicted_price_vnd = int(predicted_price_billion * 1_000_000_000)

        # Price per square meter (in million VND)
        price_per_m2_million = round(
            (predicted_price_billion * 1000) / max(area_m2, 1.0), 2
        )

        # Decompose into linear components: y = intercept + sum(w_i * x_i)
        intercept = float(self.model.intercept_)
        coefs = self.model.coef_

        contributions = {
            "base_intercept_billion": round(intercept, 4),
            "area_contribution_billion": round(float(coefs[0] * area_m2), 4),
            "bedrooms_contribution_billion": round(float(coefs[1] * bedrooms), 4),
            "bathrooms_contribution_billion": round(float(coefs[2] * bathrooms), 4),
            "distance_contribution_billion": round(
                float(coefs[3] * distance_to_center_km), 4
            ),
            "age_contribution_billion": round(
                float(coefs[4] * house_age_years), 4
            ),
        }

        # 95% approximate prediction interval based on model RMSE
        rmse = self.metadata.get("metrics", {}).get("rmse_billion_vnd", 0.22)
        margin = round(1.96 * rmse, 3)
        price_range = {
            "low_billion_vnd": max(0.1, round(predicted_price_billion - margin, 3)),
            "high_billion_vnd": round(predicted_price_billion + margin, 3),
        }

        return {
            "predicted_price_billion_vnd": predicted_price_billion,
            "predicted_price_vnd": predicted_price_vnd,
            "formatted_price": f"{predicted_price_billion:,.3f} Tỷ VNĐ",
            "price_per_m2_million_vnd": price_per_m2_million,
            "formatted_price_per_m2": f"{price_per_m2_million:,.2f} Triệu/m²",
            "prediction_interval": price_range,
            "feature_contributions": contributions,
            "input_features": {
                "area_m2": area_m2,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "distance_to_center_km": distance_to_center_km,
                "house_age_years": house_age_years,
            },
        }

    def predict_batch(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform batch predictions for multiple properties."""
        results = []
        for item in items:
            res = self.predict_one(
                area_m2=item["area_m2"],
                bedrooms=item["bedrooms"],
                bathrooms=item["bathrooms"],
                distance_to_center_km=item["distance_to_center_km"],
                house_age_years=item["house_age_years"],
            )
            results.append(res)
        return results

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata, coefficients, and performance metrics."""
        return {
            "status": "ready" if self.model is not None else "not_loaded",
            "features": FEATURE_NAMES,
            "metadata": self.metadata,
        }

    def retrain(
        self, n_samples: int = 1500, random_state: int = 42
    ) -> Dict[str, Any]:
        """Retrain model and reload in-memory instance."""
        new_metadata = train_and_save_model(
            n_samples=n_samples, random_state=random_state
        )
        self.load_or_train()
        return new_metadata
