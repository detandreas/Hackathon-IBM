"""
XGBoost model to predict ERA5 climate features (daily)
Predicts all 5 climate variables using satellite data + lag features
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
print("ERA5 CLIMATE FEATURES PREDICTION MODEL")
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

# Target features to predict
target_features = ['temp_2m_C', 'precip_mm_day', 'soil_water_vol_m3m3', 'net_solar_rad_Jm2', 'skin_temp_C']
satellite_indices = ['NDVI', 'NDBI', 'NDMI', 'BSI']
spectral_bands = ['B02_blue', 'B03_green', 'B04_red', 'B08_nir', 'B11_swir16', 'B12_swir22']
cloud_cover = 'cloud_cover_pct'

print(f"\nTarget features to predict: {target_features}")
print(f"Satellite indices available: {satellite_indices}")

# ============================================================================
# PREPARE DATA: Merge ERA5 with Sentinel2
# ============================================================================
print("\n[2/5] Preparing data...")

# Create lag features for ERA5 targets
era5_with_lags = era5.copy()
for feature in target_features:
    for lag in [1, 2, 3, 7]:
        era5_with_lags[f'{feature}_lag_{lag}'] = era5_with_lags[feature].shift(lag)

# Convert to datetime for merging
era5_with_lags['date'] = era5_with_lags['time'].dt.normalize()
sentinel2_for_merge = sentinel2.copy()
sentinel2_for_merge['date'] = sentinel2_for_merge['datetime'].dt.normalize()

# Merge on date (each ERA5 daily record gets matched with available sentinel data)
merged = pd.merge_asof(
    era5_with_lags.sort_values('date'),
    sentinel2_for_merge[['date'] + satellite_indices + spectral_bands + [cloud_cover]].sort_values('date'),
    on='date',
    direction='nearest'
)

print(f"Merged data shape: {merged.shape}")
print(f"Missing values in satellite features: {merged[satellite_indices].isnull().sum().sum()}")

# Fill missing satellite data with forward/backward fill
for col in satellite_indices + spectral_bands + [cloud_cover]:
    merged[col] = merged[col].ffill().bfill()

# Drop rows with lag NaNs (first 7 days)
merged = merged.dropna(subset=[f'{target_features[0]}_lag_7'])

print(f"After removing incomplete lag features: {merged.shape}")

# ============================================================================
# BUILD AND TRAIN MODELS
# ============================================================================
print("\n[3/5] Training models for each ERA5 feature...")

# Prepare features
lag_features = [f'{feature}_lag_{lag}' for feature in target_features for lag in [1, 2, 3, 7]]
input_features = satellite_indices + spectral_bands + [cloud_cover] + lag_features

X = merged[input_features].fillna(merged[input_features].mean())
y = merged[target_features]

print(f"Input features: {len(input_features)}")
print(f"Samples: {len(X)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train models
models = {}
print("\nModel Performance:")
print("-" * 90)
print(f"{'Feature':<20} {'Train R²':<12} {'Test R²':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 90)

for feature in target_features:
    model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train[feature])

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train[feature], y_pred_train)
    test_r2 = r2_score(y_test[feature], y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test[feature], y_pred_test))
    test_mae = mean_absolute_error(y_test[feature], y_pred_test)

    models[feature] = model

    print(f"{feature:<20} {train_r2:<12.4f} {test_r2:<12.4f} {test_rmse:<12.4f} {test_mae:<12.4f}")

print("-" * 90)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n[4/5] Top features for each target...")
print("-" * 70)

for feature in target_features:
    feature_importance = pd.DataFrame({
        'feature': input_features,
        'importance': models[feature].feature_importances_
    }).sort_values('importance', ascending=False)

    top_5 = feature_importance.head(5)
    print(f"\n{feature}:")
    for idx, row in top_5.iterrows():
        print(f"  {row['feature']:.<40} {row['importance']:.4f}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n[5/5] Saving models...")

joblib.dump(models, './models/era5_models.pkl')
joblib.dump(scaler, './models/scaler_era5.pkl')
joblib.dump(input_features, './models/features_era5.pkl')

metadata = {
    'target_features': target_features,
    'input_features': input_features,
    'satellite_indices': satellite_indices,
    'spectral_bands': spectral_bands,
    'lag_features': lag_features
}
with open('./models/metadata_era5.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n" + "="*70)
print("✅ ERA5 MODELS TRAINED AND SAVED!")
print("="*70)
print("\nModels saved to ./models/")
print("  - era5_models.pkl (5 models for each climate feature)")
print("  - scaler_era5.pkl (feature scaler)")
print("  - features_era5.pkl (feature names)")
print("  - metadata_era5.pkl (metadata)")
print("\nUse predict_era5.py to make daily predictions!")
