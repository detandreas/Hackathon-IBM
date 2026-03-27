"""
Training pipeline for XGBoost models
- ERA5 climate features (temperature, precipitation, soil moisture, radiation)
- Satellite indices (NDVI, NDBI, NDMI, BSI)
"""

import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Get base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'artifacts')

os.makedirs(MODELS_DIR, exist_ok=True)


def train_era5_models():
    """Train XGBoost models for ERA5 climate features"""
    print("Training ERA5 Climate Models...")

    # Load data
    era5_path = os.path.join(DATA_DIR, 'era5_land_features.csv')
    sentinel2_path = os.path.join(DATA_DIR, 'sentinel2_features.csv')

    era5 = pd.read_csv(era5_path)
    sentinel2 = pd.read_csv(sentinel2_path)

    era5['time'] = pd.to_datetime(era5['time'])
    sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])

    # Prepare data
    target_features = ['temp_2m_C', 'precip_mm_day', 'soil_water_vol_m3m3',
                       'net_solar_rad_Jm2', 'skin_temp_C']
    satellite_indices = ['NDVI', 'NDBI', 'NDMI', 'BSI']
    spectral_bands = ['B02_blue', 'B03_green', 'B04_red', 'B08_nir', 'B11_swir16', 'B12_swir22']

    # Create lags
    era5_with_lags = era5.copy()
    for feature in target_features:
        for lag in [1, 2, 3, 7]:
            era5_with_lags[f'{feature}_lag_{lag}'] = era5_with_lags[feature].shift(lag)

    # Merge with satellite
    era5_with_lags['date'] = era5_with_lags['time'].dt.normalize()
    sentinel2['date'] = sentinel2['datetime'].dt.normalize()

    merged = pd.merge_asof(
        era5_with_lags.sort_values('date'),
        sentinel2[['date'] + satellite_indices + spectral_bands + ['cloud_cover_pct']].sort_values('date'),
        on='date',
        direction='nearest'
    )

    # Fill missing
    for col in satellite_indices + spectral_bands + ['cloud_cover_pct']:
        merged[col] = merged[col].ffill().bfill()

    merged = merged.dropna(subset=[f'{target_features[0]}_lag_7'])

    # Features
    lag_features = [f'{feature}_lag_{lag}' for feature in target_features for lag in [1, 2, 3, 7]]
    input_features = satellite_indices + spectral_bands + ['cloud_cover_pct'] + lag_features

    X = merged[input_features].fillna(merged[input_features].mean())
    y = merged[target_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train models
    models = {}
    results = {}

    for feature in target_features:
        model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        model.fit(X_train, y_train[feature])

        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test[feature], y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test[feature], y_pred_test))
        mae = mean_absolute_error(y_test[feature], y_pred_test)

        models[feature] = model
        results[feature] = {'r2': r2, 'rmse': rmse, 'mae': mae}

        print(f"  {feature}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Save models
    joblib.dump(models, os.path.join(MODELS_DIR, 'era5_models.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_era5.pkl'))
    joblib.dump(input_features, os.path.join(MODELS_DIR, 'features_era5.pkl'))

    with open(os.path.join(MODELS_DIR, 'metadata_era5.pkl'), 'wb') as f:
        pickle.dump({
            'target_features': target_features,
            'input_features': input_features,
            'satellite_indices': satellite_indices,
            'spectral_bands': spectral_bands
        }, f)

    print("✅ ERA5 models saved")
    return models, scaler, input_features, results


def train_satellite_models():
    """Train XGBoost models for satellite indices"""
    print("Training Satellite Indices Models...")

    # Load data
    era5_path = os.path.join(DATA_DIR, 'era5_land_features.csv')
    sentinel2_path = os.path.join(DATA_DIR, 'sentinel2_features.csv')

    era5 = pd.read_csv(era5_path)
    sentinel2 = pd.read_csv(sentinel2_path)

    sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])
    era5['time'] = pd.to_datetime(era5['time'])

    # Prepare
    target_indices = ['NDVI', 'NDBI', 'NDMI', 'BSI']
    spectral_bands = ['B02_blue', 'B03_green', 'B04_red', 'B08_nir', 'B11_swir16', 'B12_swir22']
    climate_features = ['soil_water_vol_m3m3', 'net_solar_rad_Jm2', 'temp_2m_C', 'skin_temp_C', 'precip_mm_day']

    # Create lags
    sentinel2_with_lags = sentinel2.copy()
    for idx in target_indices:
        for lag in [1, 2, 3]:
            sentinel2_with_lags[f'{idx}_lag_{lag}'] = sentinel2_with_lags[idx].shift(lag)

    sentinel2_with_lags = sentinel2_with_lags.dropna()

    # Merge with climate
    sentinel2_with_lags['month_str'] = sentinel2_with_lags['month']
    era5['month_str'] = era5['time'].dt.strftime('%Y-%m')
    era5_monthly = era5.groupby('month_str')[climate_features].mean().reset_index()
    era5_monthly.columns = ['month_str'] + [f'{c}_monthly' for c in climate_features]

    merged = pd.merge(sentinel2_with_lags, era5_monthly, on='month_str', how='left').dropna()

    # Features
    lag_features = [f'{idx}_lag_{lag}' for idx in target_indices for lag in [1, 2, 3]]
    climate_monthly = [f'{c}_monthly' for c in climate_features]
    input_features = spectral_bands + lag_features + climate_monthly

    X = merged[input_features].fillna(merged[input_features].mean())
    y = merged[target_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train
    models = {}
    results = {}

    for idx in target_indices:
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                            random_state=42, verbosity=0)
        model.fit(X_train, y_train[idx])

        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test[idx], y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test[idx], y_pred_test))
        mae = mean_absolute_error(y_test[idx], y_pred_test)

        models[idx] = model
        results[idx] = {'r2': r2, 'rmse': rmse, 'mae': mae}

        print(f"  {idx}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Save
    joblib.dump(models, os.path.join(MODELS_DIR, 'satellite_models.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_satellite.pkl'))
    joblib.dump(input_features, os.path.join(MODELS_DIR, 'features_satellite.pkl'))

    with open(os.path.join(MODELS_DIR, 'metadata_satellite.pkl'), 'wb') as f:
        pickle.dump({
            'target_indices': target_indices,
            'input_features': input_features,
            'satellite_indices': target_indices,
            'spectral_bands': spectral_bands
        }, f)

    print("✅ Satellite models saved")
    return models, scaler, input_features, results


if __name__ == '__main__':
    train_era5_models()
    train_satellite_models()
