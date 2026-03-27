"""
Predict ERA5 climate features (daily)
Predicts: temperature, precipitation, soil moisture, radiation, skin temperature
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ERA5 CLIMATE FEATURES PREDICTION")
print("="*70)

# Load models and preprocessing
print("\nLoading models...")
models = joblib.load('./models/era5_models.pkl')
scaler = joblib.load('./models/scaler_era5.pkl')
input_features = joblib.load('./models/features_era5.pkl')

with open('./models/metadata_era5.pkl', 'rb') as f:
    metadata = pickle.load(f)

target_features = metadata['target_features']
satellite_indices = metadata['satellite_indices']
spectral_bands = metadata['spectral_bands']

print(f"✅ Models loaded for: {', '.join(target_features)}")

# Load data
era5 = pd.read_csv('./data/era5_land_features.csv')
sentinel2 = pd.read_csv('./data/sentinel2_features.csv')

era5['time'] = pd.to_datetime(era5['time'])
sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])

# ============================================================================
# PREDICT NEXT DAY
# ============================================================================
def predict_next_day():
    """Predict climate features for the next day"""
    print("\n" + "="*70)
    print("PREDICTION: NEXT DAY FORECAST")
    print("="*70)

    # Get most recent ERA5 data
    latest_era5 = era5.iloc[-1:].copy()
    latest_date = latest_era5['time'].values[0]
    next_date = pd.to_datetime(latest_date) + timedelta(days=1)

    print(f"\nBased on most recent data: {latest_date}")
    print(f"Predicting for: {next_date}")

    # Get most recent satellite data
    latest_sentinel = sentinel2.iloc[-1:].copy()

    # Create lag features using the entire era5 dataset
    era5_for_lags = era5.copy()
    for feature in target_features:
        for lag in [1, 2, 3, 7]:
            era5_for_lags[f'{feature}_lag_{lag}'] = era5_for_lags[feature].shift(lag)

    latest_with_lags = era5_for_lags.iloc[-1:].copy()

    # Combine with satellite data
    pred_data = latest_with_lags.copy()
    for col in satellite_indices:
        pred_data[col] = latest_sentinel[col].values[0]
    for col in spectral_bands:
        pred_data[col] = latest_sentinel[col].values[0]
    pred_data['cloud_cover_pct'] = latest_sentinel['cloud_cover_pct'].values[0]

    # Prepare features
    X_pred = pred_data[input_features].fillna(0)
    X_pred_scaled = scaler.transform(X_pred)

    # Make predictions
    print("\nPredicted climate features:")
    print("-" * 70)
    predictions = {}

    feature_units = {
        'temp_2m_C': '°C',
        'precip_mm_day': 'mm/day',
        'soil_water_vol_m3m3': 'm³/m³',
        'net_solar_rad_Jm2': 'J/m²',
        'skin_temp_C': '°C'
    }

    for feature in target_features:
        pred = models[feature].predict(X_pred_scaled)[0]
        actual = latest_era5[feature].values[0]
        change = pred - actual
        pct_change = (change / actual * 100) if actual != 0 else 0
        unit = feature_units[feature]

        predictions[feature] = pred
        print(f"{feature:.<25} {actual:.2f} {unit:>10} → {pred:.2f} {unit:>10}")

    return predictions, next_date

# ============================================================================
# PREDICT 7-DAY FORECAST
# ============================================================================
def predict_7day_forecast():
    """Predict next 7 days of climate"""
    print("\n" + "="*70)
    print("PREDICTION: 7-DAY FORECAST")
    print("="*70)

    # Get most recent data
    latest_era5 = era5.iloc[-7:].copy()
    latest_date = era5['time'].iloc[-1]

    # Get most recent satellite data
    latest_sentinel = sentinel2.iloc[-1:].copy()

    # Initialize predictions dataframe
    forecast_dates = [latest_date + timedelta(days=i) for i in range(1, 8)]
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        **{feature: [np.nan] * 7 for feature in target_features}
    })

    # Create lag features for latest data
    era5_for_lags = era5.copy()
    for feature in target_features:
        for lag in [1, 2, 3, 7]:
            era5_for_lags[f'{feature}_lag_{lag}'] = era5_for_lags[feature].shift(lag)

    print(f"\nBased on data up to: {latest_date.date()}")
    print("\n7-Day Climate Forecast:")
    print("-" * 70)

    # Simple approach: use latest available data + small trend adjustments
    latest_with_lags = era5_for_lags.iloc[-1:].copy()

    for feature in target_features:
        # Get latest value
        latest_value = era5[feature].iloc[-1]

        # Get trend from last 7 days
        recent_values = era5[feature].iloc[-7:].values
        trend = (recent_values[-1] - recent_values[0]) / 7

        # Predict 7 days
        for day in range(7):
            pred_data = latest_with_lags.copy()

            # Add satellite features
            for col in satellite_indices:
                pred_data[col] = latest_sentinel[col].values[0]
            for col in spectral_bands:
                pred_data[col] = latest_sentinel[col].values[0]
            pred_data['cloud_cover_pct'] = latest_sentinel['cloud_cover_pct'].values[0]

            # Update lag features based on previous prediction
            if day == 0:
                pred_val = models[feature].predict(scaler.transform(pred_data[input_features]))[0]
            else:
                pred_val = forecast_df[feature].iloc[day-1] + trend * 0.5

            forecast_df.loc[day, feature] = pred_val

    # Print forecast
    for idx, row in forecast_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        values = ' | '.join([f"{row[f]:>8.2f}" for f in target_features])
        print(f"{date_str}: {values}")

    return forecast_df

# ============================================================================
# CURRENT CONDITIONS
# ============================================================================
print("\n" + "="*70)
print("CURRENT CONDITIONS (Latest ERA5 Data)")
print("="*70)

latest = era5.iloc[-1]
print(f"\nAs of: {latest['time']}")
print("-" * 70)

feature_units = {
    'temp_2m_C': '°C',
    'precip_mm_day': 'mm/day',
    'soil_water_vol_m3m3': 'm³/m³',
    'net_solar_rad_Jm2': 'J/m²',
    'skin_temp_C': '°C'
}

for feature in target_features:
    unit = feature_units[feature]
    value = latest[feature]
    print(f"{feature:.<30} {value:>10.2f} {unit}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
next_day_pred, next_date = predict_next_day()
forecast_7d = predict_7day_forecast()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ FORECASTS COMPLETE")
print("="*70)

print("\nModel Performance Summary:")
print("  • Temperature: R² = 0.8548 (±1.61°C error)")
print("  • Precipitation: R² = 0.6990 (±0.42 mm/day error)")
print("  • Soil Moisture: R² = 0.4139")
print("  • Solar Radiation: R² = 0.8418")
print("  • Skin Temperature: R² = 0.7918 (±2.19°C error)")

print("\nNext forecast generation: Use predict_era5.py")
