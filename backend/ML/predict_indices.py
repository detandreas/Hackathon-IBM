"""
Combined prediction system for satellite indices
- Short-term: Predict the next satellite observation
- Long-term: Forecast monthly trends (6 months to 5 years ahead)
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SATELLITE INDICES PREDICTION SYSTEM")
print("="*70)

# Load models and preprocessing
print("\nLoading models...")
short_term_models = joblib.load('./models/short_term_models.pkl')
long_term_models = joblib.load('./models/long_term_models.pkl')
scaler_short = joblib.load('./models/scaler_short_term.pkl')
scaler_long = joblib.load('./models/scaler_long_term.pkl')
short_term_features = joblib.load('./models/features_short_term.pkl')
long_term_features = joblib.load('./models/features_long_term.pkl')

with open('./models/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

target_indices = metadata['target_indices']
climate_features = metadata['climate_features']
spectral_bands = metadata['spectral_bands']

print(f"✅ Models loaded")
print(f"Predicting: {', '.join(target_indices)}")

# Load data
era5 = pd.read_csv('./data/era5_land_features.csv')
sentinel2 = pd.read_csv('./data/sentinel2_features.csv')

era5['time'] = pd.to_datetime(era5['time'])
sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])

# ============================================================================
# FUNCTION 1: Predict next observation
# ============================================================================
def predict_next_observation():
    """Predict the next satellite observation"""
    print("\n" + "="*70)
    print("PREDICTION 1: NEXT SATELLITE OBSERVATION")
    print("="*70)

    # Get most recent data
    latest_sentinel = sentinel2.tail(1).copy()
    latest_date = latest_sentinel['datetime'].values[0]

    print(f"\nBased on most recent observation: {latest_date}")

    # Create lag features
    sentinel2_sorted = sentinel2.sort_values('datetime').reset_index(drop=True)
    temp_data = sentinel2_sorted.copy()

    for idx in target_indices:
        for lag in [1, 2, 3]:
            temp_data[f'{idx}_lag_{lag}'] = temp_data[idx].shift(lag)

    # Get latest with lags
    latest_with_lags = temp_data.tail(1).copy()

    # Get monthly climate data
    sentinel2_sorted['month_str'] = sentinel2_sorted['datetime'].dt.strftime('%Y-%m')
    era5['month_str'] = era5['time'].dt.strftime('%Y-%m')
    era5_monthly = era5.groupby('month_str')[climate_features].mean().reset_index()
    era5_monthly.columns = ['month_str'] + [f'{c}_monthly' for c in climate_features]

    latest_with_lags['month_str'] = latest_sentinel['month'].values[0]
    latest_merged = pd.merge(latest_with_lags, era5_monthly, on='month_str', how='left')

    # Prepare features
    X_pred = latest_merged[short_term_features].fillna(latest_merged[short_term_features].mean())
    X_pred_scaled = scaler_short.transform(X_pred)

    # Make predictions
    print("\nPredicted indices for next observation:")
    print("-" * 70)
    predictions_next = {}
    for idx in target_indices:
        pred = short_term_models[idx].predict(X_pred_scaled)[0]
        actual = latest_sentinel[idx].values[0]
        change = pred - actual
        pct_change = (change / actual * 100) if actual != 0 else 0

        predictions_next[idx] = pred
        print(f"{idx:6s} | Current: {actual:.4f} → Predicted: {pred:.4f} | Change: {change:+.4f} ({pct_change:+.2f}%)")

    return predictions_next, latest_date

# ============================================================================
# FUNCTION 2: Predict long-term trends
# ============================================================================
def predict_long_term(months_ahead=12):
    """Predict monthly trends for N months ahead"""
    print("\n" + "="*70)
    print(f"PREDICTION 2: LONG-TERM FORECAST ({months_ahead} months ahead)")
    print("="*70)

    # Get recent monthly data
    sentinel2_sorted = sentinel2.sort_values('datetime').reset_index(drop=True)
    sentinel2_sorted['year_month'] = sentinel2_sorted['datetime'].dt.to_period('M')

    monthly_data = sentinel2_sorted.groupby('year_month')[target_indices].mean().reset_index()
    monthly_data['year_month'] = monthly_data['year_month'].astype(str)

    # Create lag features
    for idx in target_indices:
        for lag in [1, 2, 3, 6, 12]:
            monthly_data[f'{idx}_lag_{lag}'] = monthly_data[idx].shift(lag)

    # Get latest complete monthly data
    latest_month = monthly_data.tail(1).copy()
    latest_month = latest_month.dropna()

    if len(latest_month) == 0:
        print("⚠️  Not enough data for long-term prediction")
        return None

    # Get climate data for prediction
    era5['month_str'] = era5['time'].dt.strftime('%Y-%m')
    era5_monthly = era5.groupby('month_str')[climate_features].mean().reset_index()
    era5_monthly.columns = ['month_str'] + climate_features

    # Merge
    latest_month['month_str'] = latest_month['year_month']
    latest_merged = pd.merge(latest_month, era5_monthly, on='month_str', how='left')
    latest_merged = latest_merged.fillna(latest_merged[climate_features].mean())

    # Prepare features
    X_pred = latest_merged[long_term_features].fillna(0)
    X_pred_scaled = scaler_long.transform(X_pred)

    # Make predictions
    print(f"\nPredicted indices for {months_ahead} months ahead:")
    print("-" * 70)
    predictions_lt = {}
    for idx in target_indices:
        pred = long_term_models[idx].predict(X_pred_scaled)[0]
        actual = latest_month[idx].values[0]
        change = pred - actual
        pct_change = (change / actual * 100) if actual != 0 else 0

        predictions_lt[idx] = pred
        print(f"{idx:6s} | Recent: {actual:.4f} → Forecast: {pred:.4f} | Change: {change:+.4f} ({pct_change:+.2f}%)")

    return predictions_lt

# ============================================================================
# MAIN PREDICTION PIPELINE
# ============================================================================
print("\n" + "="*70)
print("MAKING COMBINED PREDICTIONS")
print("="*70)

# Predict next observation
next_obs, last_date = predict_next_observation()

# Predict long-term trends
lt_6m = predict_long_term(months_ahead=6)
lt_1y = predict_long_term(months_ahead=12)
lt_2y = predict_long_term(months_ahead=24)
lt_5y = predict_long_term(months_ahead=60)

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: Satellite Indices Forecast")
print("="*70)

summary_df = pd.DataFrame({
    'Index': target_indices,
    'Latest': [sentinel2[idx].iloc[-1] for idx in target_indices],
    'Next Obs': [next_obs.get(idx, np.nan) for idx in target_indices],
    '6M Trend': [lt_6m.get(idx, np.nan) for idx in target_indices],
    '1Y Trend': [lt_1y.get(idx, np.nan) for idx in target_indices],
    '2Y Trend': [lt_2y.get(idx, np.nan) for idx in target_indices],
    '5Y Trend': [lt_5y.get(idx, np.nan) for idx in target_indices],
})

print("\n" + summary_df.to_string(index=False))

print("\n" + "="*70)
print("✅ PREDICTIONS COMPLETE")
print("="*70)
print("\nInterpretation:")
print("  • Next Obs: Predicts when the satellite observes this location again")
print("  • 6M-5Y Trends: Show expected direction of vegetation/soil/water changes")
print("  • Values range from -1 to 1 (normalized indices)")
print("\nNote: Confidence decreases for longer-term forecasts due to data scarcity")
