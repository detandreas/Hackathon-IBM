"""
FastAPI routes for XGBoost model predictions and risk scoring
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predict import predict_era5, predict_satellite_indices
from models.risk_score import get_current_risk_score, calculate_risk_score, get_risk_tier

router = APIRouter(prefix="/api/v1", tags=["predictions"])


# ─── Schemas ─────────────────────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """Response for prediction endpoints"""
    success: bool
    data: Dict
    message: Optional[str] = None


class RiskScoreResponse(BaseModel):
    """Response for risk scoring"""
    overall_score: float
    tier: str
    pricing_recommendation: str
    individual_scores: Dict[str, float]
    metrics: Dict[str, float]


# ─── Endpoints ───────────────────────────────────────────────────────────


@router.get("/predict/era5", response_model=PredictionResponse)
async def get_era5_predictions(days_ahead: int = Query(1, ge=1, le=30)):
    """
    Predict ERA5 climate features

    - **days_ahead**: Number of days to forecast (1-30)
    """
    try:
        predictions = predict_era5(days_ahead=days_ahead)
        return PredictionResponse(
            success=True,
            data=predictions,
            message=f"Forecast for {days_ahead} days ahead"
        )
    except Exception as e:
        return PredictionResponse(
            success=False,
            data={},
            message=str(e)
        )


@router.get("/predict/satellite", response_model=PredictionResponse)
async def get_satellite_predictions():
    """
    Predict satellite indices (NDVI, NDBI, NDMI, BSI)
    """
    try:
        predictions = predict_satellite_indices()
        return PredictionResponse(
            success=True,
            data=predictions,
            message="Next satellite observation prediction"
        )
    except Exception as e:
        return PredictionResponse(
            success=False,
            data={},
            message=str(e)
        )


@router.get("/risk/current", response_model=RiskScoreResponse)
async def get_current_risk():
    """
    Get current area risk score (0-100)

    - **0**: Very safe
    - **50**: Moderate risk
    - **100**: Very risky
    """
    risk_data = get_current_risk_score()
    return RiskScoreResponse(**risk_data)


@router.post("/risk/calculate", response_model=RiskScoreResponse)
async def calculate_custom_risk(metrics: Dict[str, float]):
    """
    Calculate risk score for custom metrics

    Expected metrics:
    - NDVI (vegetation health)
    - NDBI (built-up areas)
    - NDMI (soil moisture)
    - BSI (bare soil)
    - precip_mm_day (daily precipitation)
    - temp_2m_C (temperature)
    """
    try:
        overall_score, individual_scores = calculate_risk_score(metrics)
        tier, pricing = get_risk_tier(overall_score)

        return RiskScoreResponse(
            overall_score=float(overall_score),
            tier=tier,
            pricing_recommendation=pricing,
            individual_scores={k: float(v) for k, v in individual_scores.items()},
            metrics=metrics
        )
    except Exception as e:
        raise ValueError(f"Error calculating risk: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "XGBoost Predictions API",
        "endpoints": {
            "era5_forecast": "/api/v1/predict/era5",
            "satellite_forecast": "/api/v1/predict/satellite",
            "current_risk": "/api/v1/risk/current",
            "custom_risk": "/api/v1/risk/calculate"
        }
    }
