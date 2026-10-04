"""API endpoints for House Price Prediction service."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status

from app.ml.predictor import HousePricePredictor
from app.models.schemas import (
    BatchHouseFeatures,
    BatchPredictionResponse,
    HealthResponse,
    HouseFeatures,
    ModelInfoResponse,
    PredictionResponse,
    RetrainRequest,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check Endpoint",
    description="Used by Docker and CI/CD pipelines to verify service vitality.",
)
def health_check() -> HealthResponse:
    """Verify health status and model availability."""
    predictor = HousePricePredictor.get_instance()
    is_ready = predictor.model is not None
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        service="House Price Prediction AI",
        model_loaded=is_ready,
        version="1.0.0",
    )


@router.get(
    "/api/model/info",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Retrieve trained weights, intercept, feature names, and R2/MAE/RMSE evaluation metrics.",
)
def get_model_info() -> ModelInfoResponse:
    """Return model coefficients and evaluation metrics."""
    predictor = HousePricePredictor.get_instance()
    info = predictor.get_info()
    return ModelInfoResponse(
        status=info["status"],
        features=info["features"],
        metadata=info["metadata"],
    )


@router.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Predict Single House Price",
    description="Takes property specifications and calculates predicted price alongside linear component contributions.",
)
def predict_house_price(features: HouseFeatures) -> PredictionResponse:
    """Predict price for a single property."""
    predictor = HousePricePredictor.get_instance()
    try:
        result = predictor.predict_one(
            area_m2=features.area_m2,
            bedrooms=features.bedrooms,
            bathrooms=features.bathrooms,
            distance_to_center_km=features.distance_to_center_km,
            house_age_years=features.house_age_years,
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi dự đoán giá: {str(e)}",
        )


@router.post(
    "/api/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict Batch House Prices",
    description="Accepts up to 100 properties and returns predictions in a single batch request.",
)
def predict_batch_house_prices(
    batch: BatchHouseFeatures,
) -> BatchPredictionResponse:
    """Run batch prediction for multiple properties."""
    predictor = HousePricePredictor.get_instance()
    try:
        items = [item.model_dump() for item in batch.properties]
        results = predictor.predict_batch(items)
        parsed = [PredictionResponse(**r) for r in results]
        return BatchPredictionResponse(count=len(parsed), results=parsed)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý dự đoán hàng loạt: {str(e)}",
        )


@router.post(
    "/api/model/retrain",
    summary="Retrain Linear Regression Model",
    description="Retrains model with specified sample size and instantly swaps in-memory weights.",
)
def retrain_model(request: RetrainRequest) -> Dict[str, Any]:
    """Trigger model retraining."""
    predictor = HousePricePredictor.get_instance()
    try:
        new_metadata = predictor.retrain(
            n_samples=request.n_samples, random_state=request.random_state
        )
        return {
            "success": True,
            "message": "Mô hình đã được tái huấn luyện thành công!",
            "metadata": new_metadata,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tái huấn luyện thất bại: {str(e)}",
        )
