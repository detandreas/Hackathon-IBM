"""
Prediction functions for ERA5 climate and satellite indices
"""

import os
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'artifacts')


def load_era5_models():
    """Load trained ERA5 models"""
    return joblib.load(os.path.join(MODELS_DIR, 'era5_models.pkl')), \
           joblib.load(os.path.join(MODELS_DIR, 'scaler_era5.pkl')), \
           joblib.load(os.path.join(MODELS_DIR, 'features_era5.pkl'))


def load_satellite_models():
    """Load trained satellite models"""
    return joblib.load(os.path.join(MODELS_DIR, 'satellite_models.pkl')), \
           joblib.load(os.path.join(MODELS_DIR, 'scaler_satellite.pkl')), \
           joblib.load(os.path.join(MODELS_DIR, 'features_satellite.pkl'))


def predict_era5(days_ahead: int = 1) -> dict:
    """
    Predict ERA5 climate features for N days ahead

    Args:
        days_ahead: Number of days to forecast (default: 1)

    Returns:
        Dictionary with predictions for each climate feature
    """
    # Load data and models
    era5 = pd.read_csv(os.path.join(DATA_DIR, 'era5_land_features.csv'))
    sentinel2 = pd.read_csv(os.path.join(DATA_DIR, 'sentinel2_features.csv'))

    era5['time'] = pd.to_datetime(era5['time'])
    sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])

    models, scaler, features = load_era5_models()

    # Get latest data
    latest_era5 = era5.iloc[-1]
    latest_sentinel = sentinel2.iloc[-1]

    # Prepare features (simplified - use latest values)
    pred_data = {}
    for feature in ['NDVI', 'NDBI', 'NDMI', 'BSI', 'B02_blue', 'B03_green', 'B04_red',
                    'B08_nir', 'B11_swir16', 'B12_swir22', 'cloud_cover_pct']:
        if feature in latest_sentinel.index:
            pred_data[feature] = latest_sentinel[feature]

    # Add lag features
    for col in ['temp_2m_C', 'precip_mm_day', 'soil_water_vol_m3m3', 'net_solar_rad_Jm2', 'skin_temp_C']:
        for lag in [1, 2, 3, 7]:
            lag_col = f'{col}_lag_{lag}'
            if lag_col in features:
                pred_data[lag_col] = latest_era5[col] if lag <= 1 else 0

    # Create feature vector
    X_pred = pd.DataFrame([pred_data])
    X_pred = X_pred[features].fillna(0)
    X_pred_scaled = scaler.transform(X_pred)

    # Predict
    predictions = {}
    for feature, model in models.items():
        pred_value = model.predict(X_pred_scaled)[0]
        predictions[feature] = {
            'value': float(pred_value),
            'date': (datetime.now() + timedelta(days=days_ahead)).isoformat()
        }

    return predictions


def predict_satellite_indices() -> dict:
    """
    Predict satellite indices for next observation

    Returns:
        Dictionary with predictions for each satellite index
    """
    # Load data and models
    sentinel2 = pd.read_csv(os.path.join(DATA_DIR, 'sentinel2_features.csv'))
    era5 = pd.read_csv(os.path.join(DATA_DIR, 'era5_land_features.csv'))

    sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])
    era5['time'] = pd.to_datetime(era5['time'])

    models, scaler, features = load_satellite_models()

    # Get latest data
    latest_sentinel = sentinel2.iloc[-1]
    latest_date = latest_sentinel['datetime']

    # Prepare features
    pred_data = {}
    for feature in ['B02_blue', 'B03_green', 'B04_red', 'B08_nir', 'B11_swir16', 'B12_swir22']:
        pred_data[feature] = latest_sentinel[feature]

    # Add lags and climate
    for col in ['NDVI', 'NDBI', 'NDMI', 'BSI']:
        for lag in [1, 2, 3]:
            pred_data[f'{col}_lag_{lag}'] = latest_sentinel[col]

    climate_features = ['soil_water_vol_m3m3', 'net_solar_rad_Jm2', 'temp_2m_C', 'skin_temp_C', 'precip_mm_day']
    era5_latest = era5.iloc[-1]
    for col in climate_features:
        pred_data[f'{col}_monthly'] = era5_latest[col]

    # Create feature vector
    X_pred = pd.DataFrame([pred_data])
    X_pred = X_pred[features].fillna(0)
    X_pred_scaled = scaler.transform(X_pred)

    # Predict
    predictions = {}
    target_indices = ['NDVI', 'NDBI', 'NDMI', 'BSI']
    for idx in target_indices:
        if idx in models:
            pred_value = models[idx].predict(X_pred_scaled)[0]
            predictions[idx] = {
                'value': float(pred_value),
                'date': latest_date.isoformat()
            }

    return predictions


if __name__ == '__main__':
    print("ERA5 Predictions:")
    print(predict_era5())
    print("\nSatellite Predictions:")
    print(predict_satellite_indices())
