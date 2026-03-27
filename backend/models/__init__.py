"""
XGBoost Models for Climate & Satellite Data Prediction
"""

from .train import train_era5_models, train_satellite_models
from .predict import predict_era5, predict_satellite_indices
from .risk_score import calculate_risk_score

__all__ = [
    'train_era5_models',
    'train_satellite_models',
    'predict_era5',
    'predict_satellite_indices',
    'calculate_risk_score'
]
