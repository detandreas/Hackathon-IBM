"""
Combined XGBoost model to predict satellite indices
- Short-term: Next satellite observation (NDVI, NDBI, NDMI, BSI)
- Long-term: Monthly trends using climate data
"""

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

print("="*70)
print("COMBINED MODEL: Satellite Indices Prediction (Next Obs + Long-term)")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
era5 = pd.read_csv('./data/era5_land_features.csv')
sentinel2 = pd.read_csv('./data/sentinel2_features.csv')

# Convert time columns
era5['time'] = pd.to_datetime(era5['time'])
sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])

era5 = era5.sort_values('time').reset_index(drop=True)
sentinel2 = sentinel2.sort_values('datetime').reset_index(drop=True)

print(f"ERA5: {era5.shape}")
print(f"Sentinel2: {sentinel2.shape}")

# Target indices to predict
target_indices = ['NDVI', 'NDBI', 'NDMI', 'BSI']
spectral_bands = ['B02_blue', 'B03_green', 'B04_red', 'B08_nir', 'B11_swir16', 'B12_swir22']
climate_features = ['soil_water_vol_m3m3', 'net_solar_rad_Jm2', 'temp_2m_C', 'skin_temp_C', 'precip_mm_day']

# ============================================================================
# MODEL 1: SHORT-TERM - Predict NEXT satellite observation
# ============================================================================
print("\n" + "="*70)
print("MODEL 1: SHORT-TERM (Next Observation)")
print("="*70)

print("\n[2/5] Preparing short-term data...")

# Create lag features for indices
sentinel2_with_lags = sentinel2.copy()
for idx in target_indices:
    for lag in [1, 2, 3]:
        sentinel2_with_lags[f'{idx}_lag_{lag}'] = sentinel2_with_lags[idx].shift(lag)

sentinel2_with_lags = sentinel2_with_lags.dropna()

# Merge with climate data (by month)
sentinel2_with_lags['month_str'] = sentinel2_with_lags['month']
era5['month_str'] = era5['time'].dt.strftime('%Y-%m')

# Take average climate data per month
era5_monthly = era5.groupby('month_str')[climate_features].mean().reset_index()
era5_monthly.columns = ['month_str'] + [f'{c}_monthly' for c in climate_features]

# Merge
short_term_data = pd.merge(sentinel2_with_lags, era5_monthly, on='month_str', how='left')
short_term_data = short_term_data.dropna()

print(f"Short-term training data: {short_term_data.shape}")

# Features and targets for short-term model
lag_features = [f'{idx}_lag_{lag}' for idx in target_indices for lag in [1, 2, 3]]
climate_monthly = [f'{c}_monthly' for c in climate_features]
short_term_features = spectral_bands + lag_features + climate_monthly

X_short = short_term_data[short_term_features].fillna(short_term_data[short_term_features].mean())
y_short = short_term_data[target_indices]

print(f"Features: {len(short_term_features)}")
print(f"Targets: {target_indices}")

# Train separate models for each index
print("\n[3/5] Training short-term models...")
short_term_models = {}
scaler_short = StandardScaler()
X_short_scaled = scaler_short.fit_transform(X_short)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_short_scaled, y_short, test_size=0.2, random_state=42
)

print("\nShort-term Model Performance:")
print("-" * 70)

for idx in target_indices:
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_train_s, y_train_s[idx])

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    train_r2 = r2_score(y_train_s[idx], y_pred_train)
    test_r2 = r2_score(y_test_s[idx], y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_s[idx], y_pred_test))
    test_mae = mean_absolute_error(y_test_s[idx], y_pred_test)

    short_term_models[idx] = model

    print(f"{idx:6s} | Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")

# ============================================================================
# MODEL 2: LONG-TERM - Monthly trends using climate data
# ============================================================================
print("\n" + "="*70)
print("MODEL 2: LONG-TERM (Monthly Trends)")
print("="*70)

print("\n[4/5] Preparing long-term data...")

# Create monthly aggregates
sentinel2['year_month'] = sentinel2['datetime'].dt.to_period('M')
monthly_indices = sentinel2.groupby('year_month')[target_indices].mean().reset_index()
monthly_indices['year_month'] = monthly_indices['year_month'].astype(str)

# Create lag features for monthly data
long_term_data = monthly_indices.copy()
for idx in target_indices:
    for lag in [1, 2, 3, 6, 12]:
        long_term_data[f'{idx}_lag_{lag}'] = long_term_data[idx].shift(lag)

long_term_data = long_term_data.dropna()

# Merge with monthly climate
era5_monthly_all = era5.groupby('month_str')[climate_features].mean().reset_index()
era5_monthly_all.columns = ['year_month'] + climate_features

long_term_data = pd.merge(long_term_data, era5_monthly_all, on='year_month', how='left')
long_term_data = long_term_data.dropna()

print(f"Long-term training data: {long_term_data.shape}")

# Features and targets for long-term model
lag_features_lt = [f'{idx}_lag_{lag}' for idx in target_indices for lag in [1, 2, 3, 6, 12]]
long_term_features = lag_features_lt + climate_features

X_long = long_term_data[long_term_features].fillna(long_term_data[long_term_features].mean())
y_long = long_term_data[target_indices]

print(f"Features: {len(long_term_features)}")
print(f"Targets: {target_indices}")

# Train long-term models
print("\nLong-term Model Performance:")
print("-" * 70)

long_term_models = {}
scaler_long = StandardScaler()
X_long_scaled = scaler_long.fit_transform(X_long)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_long_scaled, y_long, test_size=0.2, random_state=42
)

for idx in target_indices:
    model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X_train_l, y_train_l[idx])

    y_pred_train = model.predict(X_train_l)
    y_pred_test = model.predict(X_test_l)

    train_r2 = r2_score(y_train_l[idx], y_pred_train)
    test_r2 = r2_score(y_test_l[idx], y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_l[idx], y_pred_test))
    test_mae = mean_absolute_error(y_test_l[idx], y_pred_test)

    long_term_models[idx] = model

    print(f"{idx:6s} | Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")

# ============================================================================
# SAVE MODELS AND PREPROCESSING
# ============================================================================
print("\n[5/5] Saving models...")

# Save short-term models
joblib.dump(short_term_models, './models/short_term_models.pkl')
joblib.dump(scaler_short, './models/scaler_short_term.pkl')
joblib.dump(short_term_features, './models/features_short_term.pkl')

# Save long-term models
joblib.dump(long_term_models, './models/long_term_models.pkl')
joblib.dump(scaler_long, './models/scaler_long_term.pkl')
joblib.dump(long_term_features, './models/features_long_term.pkl')

# Save metadata
metadata = {
    'target_indices': target_indices,
    'climate_features': climate_features,
    'spectral_bands': spectral_bands,
    'short_term_features': short_term_features,
    'long_term_features': long_term_features
}
with open('./models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n" + "="*70)
print("✅ ALL MODELS TRAINED AND SAVED!")
print("="*70)
print("\nModels saved to ./models/")
print("  - short_term_models.pkl (predicts next observation)")
print("  - long_term_models.pkl (predicts monthly trends)")
print("  - Scalers and feature lists included")
print("\nUse predict_indices.py to make predictions!")
