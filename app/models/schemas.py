"""Pydantic schemas for request validation and response serialization."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """Input features for predicting house price."""

    area_m2: float = Field(
        ...,
        gt=10.0,
        le=1000.0,
        description="Diện tích sàn sử dụng (m2)",
        examples=[85.0],
    )
    bedrooms: int = Field(
        ...,
        ge=1,
        le=20,
        description="Số lượng phòng ngủ",
        examples=[3],
    )
    bathrooms: int = Field(
        ...,
        ge=1,
        le=10,
        description="Số lượng phòng tắm/vệ sinh",
        examples=[2],
    )
    distance_to_center_km: float = Field(
        ...,
        ge=0.1,
        le=100.0,
        description="Khoảng cách tới trung tâm thành phố (km)",
        examples=[4.5],
    )
    house_age_years: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Tuổi đời công trình xây dựng (năm)",
        examples=[3.0],
    )


class PredictionInterval(BaseModel):
    """Estimated confidence interval range."""

    low_billion_vnd: float
    high_billion_vnd: float


class FeatureContributions(BaseModel):
    """Linear decomposition of predicted price by features."""

    base_intercept_billion: float
    area_contribution_billion: float
    bedrooms_contribution_billion: float
    bathrooms_contribution_billion: float
    distance_contribution_billion: float
    age_contribution_billion: float


class PredictionResponse(BaseModel):
    """Detailed response containing predicted price and explanatory breakdown."""

    predicted_price_billion_vnd: float
    predicted_price_vnd: int
    formatted_price: str
    price_per_m2_million_vnd: float
    formatted_price_per_m2: str
    prediction_interval: PredictionInterval
    feature_contributions: FeatureContributions
    input_features: HouseFeatures


class BatchHouseFeatures(BaseModel):
    """Request model for batch predictions."""

    properties: List[HouseFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Danh sách các bất động sản cần định giá",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    count: int
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str
    service: str
    model_loaded: bool
    version: str


class RetrainRequest(BaseModel):
    """Parameters for model retraining."""

    n_samples: int = Field(
        default=1500,
        ge=100,
        le=20000,
        description="Số lượng mẫu huấn luyện tổng hợp",
    )
    random_state: int = Field(
        default=42,
        ge=0,
        le=10000,
        description="Random seed cho tính tái lập",
    )


class ModelInfoResponse(BaseModel):
    """Information about active ML model."""

    status: str
    features: List[str]
    metadata: Dict[str, Any]
